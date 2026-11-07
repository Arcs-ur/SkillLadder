# Paper-to-Code Map

Reference: [SkillLadder, Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.218/),
pages 4553–4573. Table numbers below follow the final PDF.

The commands implement corrected reruns. **Runnable** means the required
entry point is connected, with user-generated inputs and provider access where
specified. **Partial** means analysis code exists but the complete table,
detector, historical sampling configuration or artifact evidence is missing.
Offline tests establish implementation behavior, not numerical reproduction
of the paper tables.

## Method and parameters

| Paper element | Implementation | Inputs / outputs and definition |
| --- | --- | --- |
| §2.2 dual-mode extraction | `src/fedskill/extractor/skill_extractor.py` | Success/failure sessions → original scrubbed L0, each linked to its source session locally |
| §2.3 hybrid clustering | `src/fedskill/clustering/hybrid_clustering.py` | Category discovery, local MiniLM assignment, oversized-cluster subdivision at distance 0.5, max size 8 |
| §2.5, Table 7 gates | `src/fedskill/server/evolver.py` | Score-gap context demotion; consistency; minimum score; strongest-parent non-degradation; critical steps; invalid synthesis/judge schemas fail the attempt |
| Eq. 6, Appendix G score | `evaluate_skill` in the evolver | Complete task-wise finite 0–10 applicability/relevance; `sum(relevance for applicability≥3) / len(all tasks)` |
| Appendix A Algorithm 1 | `scripts/run_tau2_phase1.py`, `src/fedskill/server/aggregator.py` | Fixed sessions/L0 → fixed round candidates → accepted children or retained originals; input-bound atomic checkpoints |
| §2.4, Appendix M release | `src/fedskill/utils/privacy.py`, `src/fedskill/middleware.py` | Level≥1, sanitized allowlist, new recipient IDs; feedback maps back to internal IDs locally |
| Appendix D source mixing | `compute_source_mixing` in `src/fedskill/eval/privacy_eval.py` | µ is the size of the union of distinct ancestral source sessions; missing parents/cycles mark the value incomplete |
| Appendix H Eq. 10 retrieval | `integrations/tau2/fedskill_tau2_integration.patch` | Non-retired/deduplicated candidates ranked by token overlap + runtime quality + 0.1×level; actual selections logged |
| Appendix H Eqs. 11–12 quality | `src/fedskill/models/skill.py`, `src/fedskill/server/quality.py` | Independent reward EMA, q=0.7×EMA+0.3×success fraction; alpha=0.35, success reward>0.5; retire below 0.15 after ≥5 injections |
| Appendix I Algorithm 2 | Same tau2 patch | Hidden controller query, 8 candidates, judge selects top-3, at most 2 refinement rounds; ephemeral injection |
| Table 18 settings | `configs/paper.yaml`, `src/fedskill/utils/config.py` | CLI > YAML > defaults. Hybrid=0.5 follows §2.3/Appendix F; Table 18's 0.60 is a separate disclosed value. Discovery/librarian prompt temperatures remain in their implementation constants |

The paper's R1 L0 parents have score=0 (Table 14). The code preserves that
protocol rather than introducing a new L0 scoring phase. Multi-session reruns
use `source_session_id`; the paper's one-session-per-task artifacts can use
`source_task`. Counting leaves or repeated ancestry does not increase µ.
Actual mixing distributions may differ from a bound based only on cluster
size when parent source sets overlap.

## Tables and analyses

Commands C1–C9 are detailed below. `SOURCE_ARTIFACTS` means the printed
`skills_from_train` directory from a source run. Local lineage is required for
audits; a sanitized peer payload intentionally cannot recover source ancestry.

| Final paper table / analysis | Entry / command | Inputs → outputs | Model roles | Status |
| --- | --- | --- | --- | --- |
| Table 1 ER by domain/level | `compute_entity_retention`, privacy module | All lineage + source trace texts → per-level ER | Local detector | Partial: regex analysis exists; exact historical detector configuration is not fully connected |
| Table 2 µ, ER, SA | C7; `run_privacy_evaluation` | Full lineage + trace text map → mixing/ER/SA report | SA adversary supplied through core client | Runnable µ; ER/SA exact table remains partial |
| Table 3 library statistics | C1 source phase, round JSON | Original L0 and final/round skills → counts, categories, scores, script presence | Sonnet producer/judge | Runnable artifacts; exact token-statistics/table renderer missing |
| Table 4 baseline, L0, L1+, Full static/dynamic | C1/C2 | Exact tasks + source library → per-simulation rewards, mean/SE and actual selection logs | Sonnet source; Sonnet/Qwen consumer; Sonnet simulator | Runnable five conditions; exact Mix/delta/table rendering missing |
| Table 4 AWM | C3 | Original successful L0 + successful source sessions → MiniLM static baseline | Same consumer/simulator as compared row | Runnable corrected candidate pool |
| Table 4 B1 raw-trace concatenation | No connected reward runner | Raw segments ranked using static scorer with quality/level terms zeroed | Same consumer/simulator | Missing: segment construction, runner and historical corpus configuration; native no-skill baseline is a different condition |
| Table 5 retail failures; Appendix N cases | C1 simulations and actual static usage logs | Reward-relevant behavioral/max-step outcomes and task 66/71 actions | Sonnet agent/simulator | Partial: traces/logs exist; exact failure decomposition and case renderer missing |
| Table 6 no-scrub/regex/Presidio | Privacy regex and human-audit detector profiles | Source/entity spans + skill content → ER | Local regex / Presidio NER | Partial: exact Presidio engine, language model, recognizers, thresholds and aligned runner missing |
| Tables 7–8 gates/threat map | Method rows above | Protocol descriptions | — | Documentation; ER/SA evaluate only the stated measurements |
| Table 9 literature comparison | Final paper | Published framework descriptions | — | No experiment runner |
| Table 10 human entity audit | C8 and annotation tools | Stratified documents/candidates + two reviews + missed spans → detector and retention statistics | Human gold; optional model prelabels | Runnable review/scoring tooling; historical annotations/sample unavailable; detector setup partial |
| Table 11 hard-negative SA sensitivity | C9 | Same-domain shared-tool/task-type neighbors + repeated 5/10-way lineups → skill-bootstrap CI, paired differences, normalized advantage | DeepSeek audit | Runnable analysis with configured paths; exact historical sample/weights not bundled |
| Table 12 clustering ablation | C4 | Original airline L0 + source tasks → per-cluster/per-strategy CSV and summary | Sonnet clustering/synthesis/judge; MiniLM | Runnable, fixed candidate snapshots; paper cells need a model-backed rerun |
| Table 13 process and margin diagnostics | Round clusters/skills; `audits/compute_gate_margins.py` | Candidate/accepted/preserved counts; child−strongest-parent margins | Saved default judge scores | Partial: margin utility covers locally resolvable saved parents; complete process-table renderer/retry-cause history missing |
| Table 14 R1 gate ablation | C5 | Every saved size≥2 R1 candidate, first 10 source descriptions → per-cluster acceptance/score/margin | Configured Sonnet synthesis/judge | Runnable corrected population; parents remain score=0 |
| Table 15 independent rescoring | C6 | Accepted child/strongest-parent pairs + sampled task descriptions → paired scores and margins | DeepSeek; paper also reports historical GLM audit | Runnable independent judge; exact published pair/task samples need original artifacts |
| Table 16 R2/R3 full non-Claude replication | `audits/non_claude_gate_ablation.py` | Saved candidates, held-fixed prompts/gates and task subset → per-condition CSV/JSON | DeepSeek synthesis **and** judge | Runnable per artifact directory; complete four-domain table merge missing |
| Table 17 evolution/online costs | `audits/evolution_token_audit.py`; C1 traces | Provider token usage per stage, turn/prompt/runtime records | DeepSeek evolution audit; Sonnet online audit | Evolution measurement runnable; exact online-cost table aggregation missing |
| Table 18 hyperparameters | YAML, prompt constants | Effective settings → manifests | Role-specific provider IDs | Runnable configuration; original weights/provider revisions remain unavailable |
| Table 19 telecom lineage | C1 internal round artifacts | Named L0→L3 parent chain | Sonnet synthesis/judge | Runnable new lineage; original IDs/content require historical artifacts |

## Commands and artifact contracts

All commands run from the repository root in the installed Python environment.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for variable setup, the source run,
task set selection, resume semantics, and the validation environment.
Commands without `--dry-run` may make model requests.

**C1 — five implemented comparison conditions.**

```bash
python scripts/run_tau2_comparison.py \
  --tau2-root "$TAU2_ROOT" --config configs/paper.yaml --domain airline \
  --agent-model "$SONNET_TAU2_MODEL" --user-model "$SONNET_TAU2_MODEL" \
  --skill-model "$SONNET_SKILL_MODEL" \
  --conditions baseline,l0-static,l1plus-static,full-static,full-dynamic \
  --output-dir output/airline --dry-run
```

Writes `comparison_manifest.json`, internal source/round artifacts, condition
libraries, and `summary.json` after removing `--dry-run`. Actual tau2 result
paths are recorded in settings. Reward mean/SE uses valid simulations; each
row retains task/trial/seed/status and missing rewards. This supports the
paper's single-trial task sample and explicitly labels multi-trial sample units.
Slot percentages in Mix require counting the actual selected IDs/levels in
`skill_usage.jsonl`; candidate library percentages are a different quantity.

**C2 — fixed Sonnet library transfer.**

Use C1 with `--source-artifacts-dir "$SOURCE_ARTIFACTS"`,
`--source-model "$SONNET_TAU2_MODEL"`, and a Qwen `--agent-model`. Keep the
Sonnet `--user-model` and remove the need for new skill generation.
Output paths and fingerprints distinguish consumer models.

**C3 — original success-only AWM.**

```bash
python scripts/run_awm_baseline.py \
  --tau2-root "$TAU2_ROOT" --config configs/paper.yaml --domain airline \
  --train-dir "$(dirname "$SOURCE_ARTIFACTS")" \
  --skills-out output/awm/airline_success_l0.json --run-name awm_airline \
  --agent-model "$SONNET_TAU2_MODEL" --user-model "$SONNET_TAU2_MODEL" \
  --dry-run
```

Inputs are `skills_from_train/sessions.json` and `level0_all.json`.
Success matches the source session, never membership in `evolved_all.json`.
Outputs are the deduplicated candidate JSON, tau2 results, selection logs, and
`output/baselines_awm/logs/<domain>/<run>.summary.json`.

**C4 — clustering ablation.**

```bash
python audits/ablate_clustering_airline.py \
  --artifact-dir "$SOURCE_ARTIFACTS" \
  --output-dir output/clustering_airline --seed 42
```

Reads original L0 and session descriptions; stores fixed per-strategy cluster
snapshots/manifests, `per_cluster.csv`, `per_strategy.csv`, and `summary.json`.
The local seed fixes member randomization. It does not fix LLM category output.

**C5 — all original R1 candidate clusters.**

```bash
python audits/ablate_gates_r1.py \
  --artifact-dir "$SOURCE_ARTIFACTS" --output-dir output/gates_airline_r1
```

Reads `round1_clusters.json` and `sessions.json`. Stable candidate IDs include
rejected full-gate clusters. `protocol.json` records candidate IDs, first ten
task descriptions and gate flags; results are `per_cluster.csv` and
`summary.json`. Use a separate output directory when changing the provider or
task/candidate scope; this audit's CSV is not a fully fingerprinted cross-model
resume store.

**C6 — independent child/parent judge.**

```bash
python audits/non_claude_judge_review.py \
  --provider deepseek --artifact-dir "$SOURCE_ARTIFACTS" \
  --levels 2,3 --tasks-per-domain 10 --seed 13 \
  --output-dir output/independent_judge --dry-run
```

Repeat `--artifact-dir` for each domain. Reads original/round skills and
sessions; rejects incomplete lineage or an empty pair set. Writes per-pair
CSV/JSONL reviews and an aggregate summary under the selected output directory.
The strongest parent is selected from saved source scores, then the independent
judge rates child and parent under the same tasks. Exact Table 15 pair counts
depend on the original artifacts and the requested limit/levels.

**C7 — distinct-source mixing without API calls.**

```bash
python - <<'PY'
import json, os
from pathlib import Path
from fedskill.models.skill import Skill
from fedskill.eval.privacy_eval import compute_source_mixing

root = Path(os.environ["SOURCE_ARTIFACTS"])
skills = {}
for name in ["level0_all.json", *[f"round{r}_skills.json" for r in (1, 2, 3)]]:
    for value in json.loads((root / name).read_text()):
        skills[value["id"]] = Skill.from_dict(value)
print(json.dumps(compute_source_mixing(list(skills.values())), indent=2))
PY
```

The union includes original L0 so parents absorbed by valid synthesis remain
resolvable. Incomplete lineage yields explicit errors and null level aggregates.
If the pipeline stopped early, include only existing round snapshots.
For ER/SA, `run_privacy_evaluation(skills, trace_texts, run_sa=False)` exposes
regex ER and mixing; enabling SA invokes the core LLM client. `trace_texts` is
source-task→trace text, so this historical audit API requires one source session
per task or an explicitly adapted multi-session protocol.

**C8 — human entity review.**

Set `DEFAULT_ARTIFACT_DIRS` in `audits/entity_detector_human_audit.py` to the
generated source paths before preparing:

```bash
python audits/entity_detector_human_audit.py prepare \
  --detector-profile paper-er --seed 42 --output-dir output/entity_audit
python audits/entity_detector_annotation_ui.py --audit-dir ACTUAL_AUDIT_DIR
python audits/entity_detector_human_audit.py score \
  --audit-dir ACTUAL_AUDIT_DIR --annotations REVIEW_A_CSV \
  --annotations-b REVIEW_B_CSV --additions ADJUDICATED_MISSED_SPANS_CSV
```

Preparation writes document/candidate sets; the local annotation UI supports
two reviewers and adjudication. Score consumes the reviewed candidates and
missed spans. Model prelabels from `entity_detector_model_prelabel.py` are
optional suggestions. `paper-er` is a released detector profile, not the
complete historical Presidio configuration.

**C9 — hard-negative attribution.**

Set `DEFAULT_ARTIFACT_DIRS` in `audits/source_attribution_sensitivity.py` to
generated source paths before running:

```bash
python audits/source_attribution_sensitivity.py \
  --provider deepseek --levels 3 --m-values 5,10 --repeats 5 \
  --seed 42 --output-dir output/source_attribution
```

This command performs embedding preparation and model judging. It writes
lineups in individual trial records and aggregated bootstrap/paired statistics.
SA is correct choices / evaluated trials; chance=1/m; normalized advantage
is `(SA−1/m)/(1−1/m)`. The paper's L3 original five-way and hard-negative
protocols use different judge/distractor settings and separate samples.

## Remaining numerical-reproduction work

The original private traces, skill snapshots, annotations and provider/weights
revisions are needed to assess historical table changes. Complete B1 reward
execution, exact ER/NER configuration, failure/cost/Mix table aggregation,
and historical sample selection remain explicit gaps. New corrected reruns
should report their own manifests and outputs rather than fill those gaps with
approximate values labeled as the published results.
