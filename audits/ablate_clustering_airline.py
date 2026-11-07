"""Clustering ablation on airline level-0 skills.

For each clustering strategy in STRATEGIES (LLM, Embedding x3 thresholds, Tag,
Hybrid, Reverse-Hybrid), this script:
  1. Clusters the 265 airline level-0 skills.
  2. For every multi-cluster (|C| >= 2), synthesises one evolved skill via the
     existing evolve_skill_cluster() pipeline (all 4 gates enabled, default
     defaults for retries / thresholds).
  3. Scores every successfully evolved skill via evaluate_skill().
  4. Writes per-cluster rows to <OUT>/per_cluster.csv incrementally so the run
     can be Ctrl-C'd and resumed without losing work.

Resume: a row is considered "done" if (strategy_key, cluster_idx) already
appears in per_cluster.csv. Already-done entries are skipped on next launch.

Outputs:
  output/clustering_ablation_airline/
      per_cluster.csv     one row per (strategy, cluster)
      per_strategy.csv    one row per strategy (structural metrics)
      summary.json        aggregate stats per strategy
      run.log             tee'd stdout/stderr (written by the caller)
"""

from __future__ import annotations

import argparse
import csv
import json
import io
import random
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Force UTF-8 stdout/stderr so unicode labels (e.g. greek delta) don't crash on Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fedskill.clustering import get_clustering_strategy  # noqa: E402
from fedskill.models.skill import Skill  # noqa: E402
from fedskill.server.evolver import (  # noqa: E402
    evolve_skill_cluster,
    evaluate_skill,
)
from fedskill.utils.run_manifest import (  # noqa: E402
    atomic_write_json, atomic_write_text, bind_state, cluster_identity,
    code_identity, file_digest, read_json,
)

ARTIFACT_DIR = REPO_ROOT / "output" / "tau2_airline_train50" / "skills_from_train"
OUT_DIR = REPO_ROOT / "output" / "clustering_ablation_airline"
SKILLS_PATH = ARTIFACT_DIR / "level0_all.json"
SESSIONS_PATH = ARTIFACT_DIR / "sessions.json"
PER_CLUSTER_CSV = OUT_DIR / "per_cluster.csv"
PER_STRATEGY_CSV = OUT_DIR / "per_strategy.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"

# Limit task descriptions used for scoring to keep judge prompt size bounded.
# Score gate is identical for all strategies so this only fixes the scoring
# basis, not the relative comparison.
MAX_TASKS_FOR_SCORING = 30
SEED = 42


# --------------------------------------------------------------------------
# Strategy registry (order = paper Table order)
# --------------------------------------------------------------------------

@dataclass
class StrategySpec:
    key: str            # short id used in CSV / summary
    name: str           # registry name passed to get_clustering_strategy
    label: str          # human label for tables
    kwargs: dict = field(default_factory=dict)


STRATEGIES: list[StrategySpec] = [
    StrategySpec("llm",         "llm",            "LLM one-shot"),
    StrategySpec("emb_loose",   "embedding",      "Embedding (δ=0.4)",
                 kwargs=dict(distance_threshold=0.4, max_cluster_size=8)),
    StrategySpec("emb_default", "embedding",      "Embedding (δ=0.6)",
                 kwargs=dict(distance_threshold=0.6, max_cluster_size=8)),
    StrategySpec("emb_tight",   "embedding",      "Embedding (δ=0.8)",
                 kwargs=dict(distance_threshold=0.8, max_cluster_size=8)),
    StrategySpec("tag",         "tag",            "Tag-based",
                 kwargs=dict(max_cluster_size=8)),
    StrategySpec("hybrid",      "hybrid",         "Hybrid (default)",
                 kwargs=dict(distance_threshold=0.5, max_cluster_size=8)),
    StrategySpec("rev_hybrid",  "reverse_hybrid", "Reverse-Hybrid",
                 kwargs=dict(distance_threshold=0.5, max_cluster_size=8)),
]


PER_CLUSTER_COLS = [
    "strategy_key",
    "strategy_label",
    "cluster_idx",
    "cluster_id",
    "fingerprint",
    "cluster_size",
    "best_parent_score",
    "evolved",         # 1 if evolution returned a skill, else 0
    "evolved_score",   # float or empty
    "wall_s",
]


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def load_skills() -> list[Skill]:
    with SKILLS_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: list[Skill] = []
    for d in raw:
        try:
            out.append(Skill.from_dict(d))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not parse skill: {e}", flush=True)
    return out


def load_task_descriptions(limit: int = MAX_TASKS_FOR_SCORING) -> list[str]:
    with SESSIONS_PATH.open("r", encoding="utf-8") as f:
        sessions = json.load(f)
    out: list[str] = []
    for s in sessions:
        td = s.get("task_description")
        if td:
            out.append(td)
    # deterministic truncation
    return out[:limit]


# --------------------------------------------------------------------------
# Resume support
# --------------------------------------------------------------------------

def load_done_set() -> set[tuple[str, str]]:
    """Return stable completed identities; reject unbound/partial legacy rows."""
    done: set[tuple[str, str]] = set()
    if not PER_CLUSTER_CSV.exists():
        return done
    with PER_CLUSTER_CSV.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if not row.get("cluster_id") or not row.get("fingerprint") or row.get("evolved") not in {"0", "1"}:
                raise ValueError(f"Unbound or corrupt CSV row in {PER_CLUSTER_CSV}; use a new output directory.")
            key = row["strategy_key"], row["cluster_id"]
            if key in done:
                raise ValueError(f"Duplicate cluster row {key}")
            done.add(key)
    return done


def append_row(row: dict) -> None:
    text = PER_CLUSTER_CSV.read_text(encoding="utf-8") if PER_CLUSTER_CSV.exists() else ""
    buffer = io.StringIO(newline="")
    wr = csv.DictWriter(buffer, fieldnames=PER_CLUSTER_COLS)
    if not text:
        wr.writeheader()
    wr.writerow(row)
    atomic_write_text(PER_CLUSTER_CSV, text + buffer.getvalue())


# --------------------------------------------------------------------------
# Main per-strategy pass
# --------------------------------------------------------------------------

def run_strategy(
    spec: StrategySpec,
    skills: list[Skill],
    task_descs: list[str],
    done: set[tuple[str, str]],
) -> dict:
    print(f"\n{'='*70}", flush=True)
    print(f"[strategy] {spec.key} ({spec.label})", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    snapshot = OUT_DIR / f"{spec.key}_clusters.json"
    manifest_path = OUT_DIR / f"{spec.key}_manifest.json"
    if snapshot.exists() and not manifest_path.exists():
        raise ValueError(f"Unbound cluster snapshot {snapshot}")
    state = bind_state(manifest_path, {
        "skills": [s.to_dict() for s in skills], "tasks": task_descs,
        "strategy": spec.name, "kwargs": spec.kwargs, "seed": SEED,
        "code": code_identity(REPO_ROOT),
        "model": os.getenv("SKILLLADDER_MODEL") or os.getenv("OPENAI_MODEL"),
    })
    if snapshot.exists():
        if file_digest(snapshot) != state.get("clusters_digest"):
            raise ValueError(f"Changed cluster snapshot {snapshot}")
        clusters = [[Skill.from_dict(d) for d in c] for c in read_json(snapshot)]
    else:
        strat = get_clustering_strategy(spec.name, **spec.kwargs)
        clusters = strat.cluster([Skill.from_dict(s.to_dict()) for s in skills])
        atomic_write_json(snapshot, [[s.to_dict() for s in c] for c in clusters])
        state["clusters_digest"] = file_digest(snapshot)
        atomic_write_json(manifest_path, state)
    if PER_CLUSTER_CSV.exists():
        with PER_CLUSTER_CSV.open(encoding="utf-8", newline="") as stream:
            allowed = {cluster_identity(c) for c in clusters}
            for row in csv.DictReader(stream):
                if row["strategy_key"] == spec.key and (
                    row["fingerprint"] != state["fingerprint"] or row["cluster_id"] not in allowed
                ):
                    raise ValueError(f"Mismatched saved row for {spec.key}; use a new output directory.")
    cluster_time = time.time() - t0

    sizes = [len(c) for c in clusters]
    multi = [(i, c) for i, c in enumerate(clusters) if len(c) >= 2]
    print(
        f"  clusters: {len(clusters)} total, {len(multi)} multi  "
        f"(avg size {sum(sizes)/len(sizes):.2f}, max {max(sizes) if sizes else 0}, "
        f"cluster_time {cluster_time:.1f}s)",
        flush=True,
    )

    n_evolved = 0
    evolved_scores: list[float] = []

    for k, (cluster_idx, cluster) in enumerate(multi, start=1):
        cluster_id = cluster_identity(cluster)
        if (spec.key, cluster_id) in done:
            print(f"  [skip] [{k:>3}/{len(multi)}] cluster_idx={cluster_idx} already done", flush=True)
            continue

        best_parent_score = max((s.score for s in cluster), default=0.0)
        t1 = time.time()
        evolved = None
        score_val: float | str = ""
        try:
            evolved = evolve_skill_cluster(
                cluster, task_descriptions=task_descs, rng=random.Random(f"{SEED}:{cluster_id}")
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [err evolve] cluster_idx={cluster_idx}: {e}", flush=True)

        if evolved is not None:
            try:
                score_val = float(evaluate_skill(evolved, task_descs))
                evolved.score = score_val
                n_evolved += 1
                evolved_scores.append(score_val)
            except Exception as e:  # noqa: BLE001
                print(f"  [err score] cluster_idx={cluster_idx}: {e}", flush=True)
                score_val = ""

        wall = time.time() - t1
        accepted = evolved is not None
        sc_print = f"{score_val:.2f}" if isinstance(score_val, float) else "  --"
        print(
            f"  [{k:>3}/{len(multi)}] cluster_idx={cluster_idx:>3} size={len(cluster):>2} "
            f"evolved={accepted} score={sc_print} wall={wall:.1f}s",
            flush=True,
        )

        append_row({
            "strategy_key":      spec.key,
            "strategy_label":    spec.label,
            "cluster_idx":       cluster_idx,
            "cluster_id":        cluster_id,
            "fingerprint":       state["fingerprint"],
            "cluster_size":      len(cluster),
            "best_parent_score": round(best_parent_score, 4),
            "evolved":           int(accepted),
            "evolved_score":     f"{score_val:.4f}" if isinstance(score_val, float) else "",
            "wall_s":            round(wall, 2),
        })

    return {
        "strategy_key":      spec.key,
        "strategy_label":    spec.label,
        "params":            spec.kwargs,
        "num_skills":        len(skills),
        "num_clusters":      len(clusters),
        "num_multi":         len(multi),
        "num_singletons":    len(clusters) - len(multi),
        "avg_cluster_size":  round(sum(sizes) / len(sizes), 3) if sizes else 0.0,
        "max_cluster_size":  max(sizes) if sizes else 0,
        "cluster_wall_s":    round(cluster_time, 2),
        # NB: counts below combine fresh + resumed rows from per_cluster.csv
        "num_evolved_fresh": n_evolved,
        "evolved_scores_fresh": [round(x, 4) for x in evolved_scores],
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def aggregate_from_csv(per_strategy_meta: list[dict]) -> dict:
    """Merge fresh structural metrics with resumed evolved rows from CSV."""
    rows: dict[str, list[dict]] = {sp.key: [] for sp in STRATEGIES}
    if PER_CLUSTER_CSV.exists():
        with PER_CLUSTER_CSV.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r["strategy_key"] in rows:
                    rows[r["strategy_key"]].append(r)

    out: dict[str, dict] = {}
    for meta in per_strategy_meta:
        key = meta["strategy_key"]
        rs = rows.get(key, [])
        evolved_rows = [r for r in rs if r["evolved"] == "1"]
        scores = [float(r["evolved_score"]) for r in evolved_rows if r["evolved_score"]]
        meta_out = dict(meta)
        meta_out["num_evolved"] = len(evolved_rows)
        meta_out["accept_rate"] = (
            round(len(evolved_rows) / max(1, len(rs)), 4) if rs else 0.0
        )
        meta_out["mean_evolved_score"] = (
            round(sum(scores) / len(scores), 4) if scores else 0.0
        )
        meta_out["median_evolved_score"] = (
            round(sorted(scores)[len(scores) // 2], 4) if scores else 0.0
        )
        meta_out["n_cluster_rows"] = len(rs)
        out[key] = meta_out
    return out


def write_per_strategy_csv(agg: dict) -> None:
    cols = [
        "strategy_key",
        "strategy_label",
        "num_clusters",
        "num_multi",
        "num_singletons",
        "avg_cluster_size",
        "max_cluster_size",
        "cluster_wall_s",
        "num_evolved",
        "accept_rate",
        "mean_evolved_score",
        "median_evolved_score",
    ]
    with PER_STRATEGY_CSV.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for sp in STRATEGIES:
            m = agg.get(sp.key, {})
            wr.writerow({c: m.get(c, "") for c in cols})


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    global ARTIFACT_DIR, OUT_DIR, SKILLS_PATH, SESSIONS_PATH
    global PER_CLUSTER_CSV, PER_STRATEGY_CSV, SUMMARY_JSON
    global SEED

    parser = argparse.ArgumentParser(description="Run the paper's clustering ablation.")
    parser.add_argument(
        "--artifact-dir",
        default=str(ARTIFACT_DIR),
        help="Directory containing level0_all.json and sessions.json.",
    )
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    SEED = args.seed

    ARTIFACT_DIR = Path(args.artifact_dir)
    OUT_DIR = Path(args.output_dir)
    SKILLS_PATH = ARTIFACT_DIR / "level0_all.json"
    SESSIONS_PATH = ARTIFACT_DIR / "sessions.json"
    PER_CLUSTER_CSV = OUT_DIR / "per_cluster.csv"
    PER_STRATEGY_CSV = OUT_DIR / "per_strategy.csv"
    SUMMARY_JSON = OUT_DIR / "summary.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    skills = load_skills()
    task_descs = load_task_descriptions()
    print(f"[load] {len(skills)} L0 skills, {len(task_descs)} task descriptions", flush=True)

    done = load_done_set()
    print(f"[resume] {len(done)} cluster rows already in CSV; skipping those", flush=True)

    # Load previous summary so partial reruns (ABLATE_CLUSTERING_ONLY) don't drop
    # the structural metrics (avg_cluster_size, cluster_wall_s, ...) of strategies
    # we are not re-running this invocation.
    prev_meta_by_key: dict[str, dict] = {}
    if SUMMARY_JSON.exists():
        try:
            prev = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                for k, v in prev.items():
                    if isinstance(v, dict) and "strategy_key" in v:
                        prev_meta_by_key[k] = v
        except Exception as e:  # noqa: BLE001
            if isinstance(e, ValueError):
                raise
            print(f"[warn] could not load prev summary: {e}", flush=True)

    per_strategy_meta: list[dict] = []
    only = os.environ.get("ABLATE_CLUSTERING_ONLY")  # comma-sep keys for partial reruns
    only_set = {x.strip() for x in only.split(",") if x.strip()} if only else None

    for spec in STRATEGIES:
        if only_set and spec.key not in only_set:
            # Carry forward previous structural metrics for strategies we skip.
            prev = prev_meta_by_key.get(spec.key)
            if prev and prev.get("num_clusters", 0) > 0 and "error" not in prev:
                print(f"[carry] strategy {spec.key} (using prev summary)", flush=True)
                per_strategy_meta.append({
                    k: prev[k] for k in (
                        "strategy_key", "strategy_label", "params", "num_skills",
                        "num_clusters", "num_multi", "num_singletons",
                        "avg_cluster_size", "max_cluster_size", "cluster_wall_s",
                        "num_evolved_fresh", "evolved_scores_fresh",
                    ) if k in prev
                })
            else:
                print(f"[skip] strategy {spec.key} (not in ABLATE_CLUSTERING_ONLY)", flush=True)
            continue
        try:
            meta = run_strategy(spec, skills, task_descs, done)
        except Exception as e:  # noqa: BLE001
            if isinstance(e, ValueError):
                raise
            print(f"[err strategy {spec.key}] {type(e).__name__}: {e}", flush=True)
            meta = {
                "strategy_key": spec.key,
                "strategy_label": spec.label,
                "params": spec.kwargs,
                "num_skills": len(skills),
                "num_clusters": 0,
                "num_multi": 0,
                "num_singletons": 0,
                "avg_cluster_size": 0.0,
                "max_cluster_size": 0,
                "cluster_wall_s": 0.0,
                "num_evolved_fresh": 0,
                "evolved_scores_fresh": [],
                "error": f"{type(e).__name__}: {e}",
            }
        per_strategy_meta.append(meta)

    agg = aggregate_from_csv(per_strategy_meta)
    SUMMARY_JSON.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    write_per_strategy_csv(agg)
    print(f"\n[done] summary -> {SUMMARY_JSON}", flush=True)
    print(f"[done] per_strategy -> {PER_STRATEGY_CSV}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'Strategy':<22}{'Clust':>6}{'Multi':>6}{'AvgSz':>7}{'MaxSz':>6}{'Acc':>6}{'MeanScr':>9}")
    print("-" * 78)
    for sp in STRATEGIES:
        m = agg.get(sp.key, {})
        if not m:
            continue
        print(
            f"{sp.label[:22]:<22}"
            f"{m.get('num_clusters',0):>6}"
            f"{m.get('num_multi',0):>6}"
            f"{m.get('avg_cluster_size',0):>7.2f}"
            f"{m.get('max_cluster_size',0):>6}"
            f"{m.get('accept_rate',0):>6.2f}"
            f"{m.get('mean_evolved_score',0):>9.2f}"
        )


if __name__ == "__main__":
    main()
