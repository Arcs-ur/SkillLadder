# SkillLadder

Official code release for **SkillLadder: Multi-User Skill Evolution with
Compliance-Oriented Synchronization and Quality-Gated Synthesis**, accepted to
Findings of EMNLP 2026.

SkillLadder extracts local L0 skills from agent traces, clusters related skills
across contributors, synthesizes an L0-to-L3 hierarchy under strongest-parent
quality gates, and releases only sanitized L1+ skills. The release boundary is
an empirical compliance layer, not a cryptographic privacy guarantee.

## Release Contents

- Core extraction, clustering, synthesis, quality-gating, sanitization, and
  runtime-quality code in `src/fedskill/`.
- Main tau2 experiment drivers and the AWM baseline in `scripts/`.
- Entity-retention, source-attribution, non-Claude judge, cost, clustering, and
  gate-ablation programs in `audits/`.
- Exact paper hyperparameters and task IDs in `configs/`.
- The tau2 agent/retrieval integration in `integrations/tau2/`.
- All prompts, either as named constants in the implementation or in the tau2
  integration patch; `prompts/README.md` is the prompt index.

In accordance with the paper's release statement, this repository does **not**
contain raw traces, generated skill artifacts, experimental results, logs, or
human-audit annotations.

## Installation

Python 3.12 is the reference environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the direct dependency versions used to verify this artifact, install
`requirements-paper.txt` before the editable package. tau2-bench manages its
own dependency set at the pinned revision.

Use `.env.example` as a template and load its values into your shell.
Skill extraction and evolution use an OpenAI-compatible endpoint through
`SKILLLADDER_API_KEY`, `SKILLLADDER_BASE_URL`, and `SKILLLADDER_MODEL`.
Secrets should never be committed.

## tau2 Reproduction

The integration patch targets tau2-bench commit
`01e812d1d1c4df6d8d2299bf175e482527112244`. Prepare tau2 as described in
`integrations/tau2/README.md`, then run a domain with its exact task-ID file:

```bash
python scripts/run_tau2_comparison.py \
  --tau2-root /path/to/tau2-bench \
  --domain airline \
  --task-ids-file configs/task_splits/airline_50.json \
  --agent-model YOUR_LITELLM_AGENT_MODEL \
  --user-model YOUR_LITELLM_USER_MODEL \
  --output-dir output/airline
```

Equivalent split files are provided for banking knowledge (97), retail (114),
and telecom (200). Generated outputs stay local under `output/` and are ignored
by Git. See `docs/REPRODUCIBILITY.md` for the full workflow and
`docs/COSTS.md` before launching API-backed runs.

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/skillladder/` | Public package surface (`Skill`, `SkillLadderServer`) |
| `src/fedskill/` | Reference implementation; the historical namespace is retained for artifact compatibility |
| `scripts/` | Extraction/evolution, baseline/static/dynamic comparison, and AWM entry points |
| `audits/` | Paper audit and ablation programs; they consume locally generated artifacts |
| `configs/paper.yaml` | Main-paper configuration |
| `configs/task_splits/` | Ordered, exact tau2 task IDs |
| `integrations/tau2/` | Pinned tau2 patch and setup instructions |
| `prompts/` | Index of disclosed prompt templates |
| `docs/` | Reproduction, costs, and release-scope documentation |

## License

SkillLadder is released under the Apache License 2.0. The tau2 integration
includes MIT-licensed material; see `THIRD_PARTY_NOTICES.md`.

## Citation

The final ACL Anthology entry is not yet available. Citation metadata is in
`CITATION.cff` and will be updated with the proceedings URL and BibTeX after
publication.
