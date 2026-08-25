"""Compute per-evolved-skill margin = evolved.score - max(parent.score).

Reads round{0,1,2,3}_skills.json from saved artifact directories, builds a
score lookup by skill id, and reports margin distribution per round and
overall. No API calls.

Output: console table plus a caller-selected JSON file under ``output/``.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"

# Same artifact set as Table 10 (process audit). Each entry contributes
# round1_skills.json / round2_skills.json / round3_skills.json.
ARTIFACT_DIRS = [
    OUTPUT_ROOT / "tau2_airline_train50" / "skills_from_train",
    OUTPUT_ROOT / "tau2_banking_knowledge_all97_max50" / "skills_from_train",
    OUTPUT_ROOT / "tau2_retail_all114" / "skills_from_train",
    OUTPUT_ROOT / "tau2_telecom_remaining200_max50" / "skills_from_train",
]


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "skills" in data:
        data = data["skills"]
    return [s for s in data if isinstance(s, dict)]


def collect(artifact_dirs: list[Path]):
    """For each artifact dir, build id->score map across all rounds, then
    iterate evolved skills (level>=1) and compute margin against best parent.
    """
    rows = []  # list of dicts
    for d in artifact_dirs:
        if not d.exists():
            print(f"[skip] {d} (missing)")
            continue
        # union over rounds: a child in round r has parents in r-1 (and earlier)
        score_map: dict[str, float] = {}
        per_round = {}
        for r in (1, 2, 3):
            path = d / f"round{r}_skills.json"
            per_round[r] = _load(path)
            for s in per_round[r]:
                sid = s.get("id")
                if sid is None:
                    continue
                score_map[sid] = float(s.get("score") or 0.0)
        # We also need round 0 (parents of round-1 children). round1 file
        # actually contains BOTH new L1 skills AND carried-forward L0 skills,
        # so score_map already covers L0 ids.

        for r in (1, 2, 3):
            for s in per_round[r]:
                if int(s.get("level") or 0) < 1:
                    continue
                if s.get("level") != r:
                    continue  # carried-forward skill from earlier round
                parent_ids = s.get("parent_ids") or []
                if not parent_ids:
                    continue
                parent_scores = [score_map.get(pid) for pid in parent_ids]
                parent_scores = [p for p in parent_scores if p is not None]
                if not parent_scores:
                    continue
                child_score = float(s.get("score") or 0.0)
                best_parent = max(parent_scores)
                rows.append({
                    "domain": d.parts[-2],
                    "round": r,
                    "child_id": s.get("id"),
                    "n_parents": len(parent_ids),
                    "best_parent_score": best_parent,
                    "child_score": child_score,
                    "margin": child_score - best_parent,
                })
    return rows


def summarize(rows: list[dict]):
    def stats(vs: list[float]) -> dict:
        if not vs:
            return {"n": 0}
        return {
            "n": len(vs),
            "mean": round(statistics.mean(vs), 3),
            "median": round(statistics.median(vs), 3),
            "p10": round(statistics.quantiles(vs, n=10)[0], 3) if len(vs) >= 10 else min(vs),
            "p90": round(statistics.quantiles(vs, n=10)[-1], 3) if len(vs) >= 10 else max(vs),
            "min": round(min(vs), 3),
            "max": round(max(vs), 3),
            "frac_ge_0": round(sum(1 for v in vs if v >= 0) / len(vs), 3),
            "frac_ge_05": round(sum(1 for v in vs if v >= 0.5) / len(vs), 3),
            "frac_ge_1": round(sum(1 for v in vs if v >= 1.0) / len(vs), 3),
            "frac_ge_2": round(sum(1 for v in vs if v >= 2.0) / len(vs), 3),
        }

    out = {
        "overall": stats([r["margin"] for r in rows]),
        "by_round": {
            r: stats([row["margin"] for row in rows if row["round"] == r])
            for r in (1, 2, 3)
        },
        "by_domain": {
            d: stats([row["margin"] for row in rows if row["domain"] == d])
            for d in sorted({row["domain"] for row in rows})
        },
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="Compute child-vs-parent gate margins.")
    parser.add_argument(
        "--artifact-dir",
        action="append",
        default=[],
        help="A skills_from_train directory. Repeat for multiple domains.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_ROOT / "gate_margins.json"),
        help="Destination JSON file.",
    )
    args = parser.parse_args()
    artifact_dirs = [Path(path) for path in args.artifact_dir] or ARTIFACT_DIRS

    if not args.artifact_dir and not any(path.exists() for path in ARTIFACT_DIRS):
        parser.error(
            "No artifact directories found. Pass one or more --artifact-dir paths "
            "from a local reproduction run."
        )
    rows = collect(artifact_dirs)
    print(f"\nCollected {len(rows)} evolved skill->parent records.\n")

    summary = summarize(rows)

    def fmt(s):
        return (f"n={s['n']:>4} mean={s.get('mean','--'):>6} med={s.get('median','--'):>6} "
                f"p10={s.get('p10','--'):>6} p90={s.get('p90','--'):>6} "
                f">=0:{s.get('frac_ge_0','--'):>5} >=0.5:{s.get('frac_ge_05','--'):>5} "
                f">=1:{s.get('frac_ge_1','--'):>5} >=2:{s.get('frac_ge_2','--'):>5}")

    print("=" * 110)
    print(f"{'OVERALL':<25}{fmt(summary['overall'])}")
    print("-" * 110)
    print("By round:")
    for r in (1, 2, 3):
        s = summary["by_round"][r]
        print(f"  R{r:<23}{fmt(s)}")
    print("-" * 110)
    print("By domain:")
    for d, s in summary["by_domain"].items():
        print(f"  {d:<23}{fmt(s)}")
    print("=" * 110)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
