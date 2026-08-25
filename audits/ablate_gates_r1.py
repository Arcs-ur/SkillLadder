"""Gate-level ablation on airline-50 R1.

Re-runs cluster-level synthesis for every cluster that the full pipeline
accepted in airline R1, under four flag combos:
    full           : all 4 gates on (control)
    no_consistency : Gate 1 off
    no_nondegrad   : Gate 3 off
    no_critical    : Gate 4 off

For each cluster x condition, records:
    accepted (bool), evolved.score, margin (evolved - best_parent), tokens.

Outputs per-cluster CSV + aggregate JSON. Uses Substrate Opus (no $ billing,
but rate-limited).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fedskill.models.skill import Skill
from fedskill.server.evolver import evolve_skill_cluster  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "output" / "tau2_airline_train50" / "skills_from_train"
OUT_DIR = REPO_ROOT / "output" / "gate_ablation_airline_r1"


def _skill_from_dict(d: dict) -> Skill:
    # Skill is a dataclass; construct from compatible kwargs.
    valid_fields = {
        "name", "description", "when_to_use", "procedure", "examples", "scripts",
        "category", "source_task", "level", "parent_ids", "score", "inject_count",
        "positive_count", "negative_count", "effectiveness", "version",
        "quality_score", "quality_history", "last_quality_update", "retired",
        "retirement_reason", "id",
    }
    kwargs = {k: v for k, v in d.items() if k in valid_fields}
    return Skill(**kwargs)


def load_clusters_from_artifact() -> tuple[list[list[Skill]], list[str]]:
    """Return (list of clusters, list of cluster_ids).

    Each cluster = list of L0 parent Skill objects, reconstructed from the
    parent_ids of every L1 skill in round1_skills.json. L0 skills are loaded
    from level0_all.json (round1_skills only contains the carried-forward
    subset, plus the newly synthesized L1 skills).
    """
    l0_path = ARTIFACT_DIR / "level0_all.json"
    with l0_path.open("r", encoding="utf-8") as f:
        l0_list = json.load(f)
    id_to_skill: dict[str, Skill] = {}
    for d in l0_list:
        sk = _skill_from_dict(d)
        id_to_skill[sk.id] = sk

    r1_path = ARTIFACT_DIR / "round1_skills.json"
    with r1_path.open("r", encoding="utf-8") as f:
        round1 = json.load(f)
    # Also add any L0 carried-forward in round1_skills (rare overlap, safe to merge).
    for d in round1:
        if int(d.get("level") or 0) == 0:
            sk = _skill_from_dict(d)
            id_to_skill.setdefault(sk.id, sk)

    clusters: list[list[Skill]] = []
    ids: list[str] = []
    for d in round1:
        if int(d.get("level") or 0) != 1:
            continue
        parents = []
        for pid in (d.get("parent_ids") or []):
            sk = id_to_skill.get(pid)
            if sk is None:
                continue
            parents.append(sk)
        if len(parents) >= 2:
            clusters.append(parents)
            ids.append(d.get("id") or f"unknown_{len(ids)}")
    return clusters, ids


def load_task_descriptions() -> list[str]:
    sess_path = ARTIFACT_DIR / "sessions.json"
    with sess_path.open("r", encoding="utf-8") as f:
        sessions = json.load(f)
    out: list[str] = []
    for s in sessions:
        td = s.get("task_description")
        if td:
            out.append(td)
    return out


CONDITIONS = {
    "full":            dict(),
    "no_consistency":  dict(disable_consistency=True),
    "no_nondegrad":    dict(disable_nondegradation=True),
    "no_critical":     dict(disable_critical_step=True),
}


def run(limit: int | None = None, sleep_s: float = 0.0) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] clusters from {ARTIFACT_DIR}", flush=True)
    clusters, cluster_ids = load_clusters_from_artifact()
    if limit is not None:
        clusters = clusters[:limit]
        cluster_ids = cluster_ids[:limit]
    print(f"[load] {len(clusters)} clusters", flush=True)
    task_descs = load_task_descriptions()
    print(f"[load] {len(task_descs)} task descriptions", flush=True)

    # subsample task descs to control cost: per Step 2 scoring call uses ALL tasks
    # in the prompt, so trim to 10 representative descriptions.
    task_descs = task_descs[:10]
    print(f"[scoring] using first {len(task_descs)} task descriptions for judge", flush=True)

    csv_path = OUT_DIR / "per_cluster.csv"
    summary: dict[str, dict] = {c: defaultdict(list) for c in CONDITIONS}

    # Resume support
    done: set[tuple[str, str]] = set()
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((row["cluster_id"], row["condition"]))
                cond = row["condition"]
                if row["accepted"] == "True":
                    summary[cond]["accepted"].append(1)
                    summary[cond]["score"].append(float(row["score"]))
                    summary[cond]["margin"].append(float(row["margin"]))
                else:
                    summary[cond]["accepted"].append(0)
        print(f"[resume] loaded {len(done)} prior (cluster,condition) pairs", flush=True)

    write_header = not csv_path.exists()
    f_csv = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.writer(f_csv)
    if write_header:
        writer.writerow(["cluster_id", "n_parents", "condition", "accepted",
                         "score", "best_parent_score", "margin"])
        f_csv.flush()

    t0 = time.time()
    n_total_calls = len(clusters) * len(CONDITIONS)
    n_done = len(done)
    print(f"[run] total (cluster,condition) = {n_total_calls}, already done = {n_done}", flush=True)

    for ci, (cluster, cid) in enumerate(zip(clusters, cluster_ids)):
        best_parent = max((s.score for s in cluster), default=0.0)
        for cond, flags in CONDITIONS.items():
            if (cid, cond) in done:
                continue
            try:
                evolved = evolve_skill_cluster(
                    cluster,
                    task_descriptions=task_descs,
                    **flags,
                )
            except Exception as e:
                print(f"  [error] cluster={cid} cond={cond}: {e}", flush=True)
                evolved = None
            accepted = evolved is not None
            score = float(evolved.score) if accepted else float("nan")
            margin = (score - best_parent) if accepted else float("nan")
            writer.writerow([cid, len(cluster), cond, accepted,
                             f"{score:.3f}" if accepted else "",
                             f"{best_parent:.3f}",
                             f"{margin:.3f}" if accepted else ""])
            f_csv.flush()
            summary[cond]["accepted"].append(1 if accepted else 0)
            if accepted:
                summary[cond]["score"].append(score)
                summary[cond]["margin"].append(margin)
            n_done += 1
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            eta = (n_total_calls - n_done) / max(rate, 1e-6)
            print(f"  [{n_done}/{n_total_calls}] cluster={cid[:8]} cond={cond:>16} "
                  f"accepted={accepted} score={'%.2f'%score if accepted else '--':>5} "
                  f"margin={'%+.2f'%margin if accepted else '--':>6} "
                  f"(ETA {eta:.0f}s)", flush=True)
            if sleep_s > 0:
                time.sleep(sleep_s)

    f_csv.close()

    # Aggregate
    agg: dict[str, dict] = {}
    for cond in CONDITIONS:
        acc_list = summary[cond]["accepted"]
        scores = summary[cond]["score"]
        margins = summary[cond]["margin"]
        n_total = len(acc_list) if acc_list else 0
        n_acc = sum(acc_list) if acc_list else 0
        agg[cond] = {
            "n_clusters": n_total,
            "n_accepted": n_acc,
            "acceptance_rate": round(n_acc / n_total, 3) if n_total else None,
            "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
            "mean_margin": round(sum(margins) / len(margins), 3) if margins else None,
            "median_margin": (sorted(margins)[len(margins) // 2] if margins else None),
            "frac_margin_ge_0": round(sum(1 for m in margins if m >= 0) / len(margins), 3) if margins else None,
            "frac_margin_ge_05": round(sum(1 for m in margins if m >= 0.5) / len(margins), 3) if margins else None,
        }

    (OUT_DIR / "summary.json").write_text(json.dumps(agg, indent=2))
    print("\n" + "=" * 80)
    for cond, s in agg.items():
        if s.get("acceptance_rate") is None:
            print(f"{cond:>16}  (no data)")
            continue
        print(f"{cond:>16}  acc={s['acceptance_rate']:.2f}  "
              f"mean_score={s['mean_score']}  "
              f"mean_margin={s['mean_margin']}  "
              f"median_margin={s['median_margin']}  "
              f"%margin>=0:{s['frac_margin_ge_0']}  "
              f"%margin>=0.5:{s['frac_margin_ge_05']}")
    print(f"\nWrote {OUT_DIR / 'summary.json'}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    p.add_argument("--output-dir", default=str(OUT_DIR))
    p.add_argument("--limit", type=int, default=None, help="Limit number of clusters (debug).")
    p.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()
    ARTIFACT_DIR = Path(args.artifact_dir)
    OUT_DIR = Path(args.output_dir)
    run(limit=args.limit, sleep_s=args.sleep)
