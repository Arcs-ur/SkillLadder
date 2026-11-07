# Reproducibility Guide

The current code implements corrected reruns of the
[published protocol](https://aclanthology.org/2026.findings-emnlp.218/).
Keep new results separate from historical paper artifacts. The
[paper-to-code map](PAPER_TO_CODE.md) identifies supported entries and remaining
gaps, including B1 raw-trace reward evaluation and the full Presidio setup.

## Environment and models

Use Python 3.12 and install from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q -p no:cacheprovider
```

`requirements-paper.txt` retains historical direct pins. It is not the complete
dependency lock for the current code or the independent tau2 environment.
Prepare the exact tau2 revision with its extras and native dependencies as
described in [the integration guide](../integrations/tau2/README.md).

Load `.env.example` values into your shell; the drivers do not load `.env`
automatically. Extraction, synthesis, and default judging use the
OpenAI-compatible `SKILLLADDER_MODEL`, `SKILLLADDER_BASE_URL`, and
`SKILLLADDER_API_KEY`. tau2 uses LiteLLM agent/user IDs and `OPENAI_*` settings.
The paper roles are Sonnet 4.6 for source agent, simulator, synthesizer and
default judge; transfer changes only the consumer to Qwen3-32B or Qwen3-8B.
Supply actual provider IDs rather than the display names in YAML.

For illustration, use these shell variables with values supported by your
provider:

```bash
export TAU2_ROOT=/absolute/path/to/tau2-bench
export SONNET_TAU2_MODEL=YOUR_LITELLM_SONNET_MODEL
export SONNET_SKILL_MODEL=YOUR_OPENAI_COMPATIBLE_SONNET_MODEL
```

## Configuration and exact tasks

The effective precedence is **explicit CLI > YAML > compatibility defaults**.
Pass `--config configs/paper.yaml` to comparison, phase1, and AWM. It selects
the domain's ordered IDs unless task IDs or start/count are explicitly supplied.
Model-role display names are descriptive; only `models.agent`, `models.user`,
and `models.skill`, when present, act as defaults for executable provider IDs.

| Domain | Tasks | File | Native task set / split |
| --- | ---: | --- | --- |
| airline | 50 | `configs/task_splits/airline_50.json` | airline / base |
| banking_knowledge | 97 | `configs/task_splits/banking_knowledge_97.json` | banking_knowledge / base |
| retail | 114 | `configs/task_splits/retail_114.json` | retail / base |
| telecom | 200 | `configs/task_splits/telecom_200.json` | telecom / full |

Task definitions, databases, policies and simulators come from pinned tau2.
The hybrid distance threshold remains **0.5**, following §2.3 and Appendix F.
Table 18 lists **0.60**; this paper inconsistency is recorded rather than
resolved by changing every strategy. The embedding-only ablation separately
sweeps 0.4/0.6/0.8.

Paper YAML enables strict embedding loading. For an API-backed rerun, resolve
the selected MiniLM repository to a weights commit and pass
`--embedding-revision <weights-commit>`. This revision is recorded in manifests
and forwarded to clustering and embedding retrieval. The paper's weights
revision is not available in the released configuration. Offline validation
uses no weights and therefore cannot establish embedding-result equivalence.

## Generate the Sonnet source library and run all implemented conditions

Preview the commands first:

```bash
python scripts/run_tau2_comparison.py \
  --tau2-root "$TAU2_ROOT" --config configs/paper.yaml --domain airline \
  --agent-model "$SONNET_TAU2_MODEL" --user-model "$SONNET_TAU2_MODEL" \
  --skill-model "$SONNET_SKILL_MODEL" \
  --conditions baseline,l0-static,l1plus-static,full-static,full-dynamic \
  --output-dir output/airline --dry-run
```

`--dry-run` validates local paths and effective parameters, prints commands, and
creates no run artifacts. It requests no model and downloads no weights.
Remove the flag for an API-backed run. Repeat with the other three domain
names; YAML selects their task lists. The default without `--conditions`
remains the historical three-condition baseline/Full static/Full dynamic path.

| Condition | Corpus and injection |
| --- | --- |
| `baseline` | No skills, native `llm_agent` |
| `l0-static` | All original L0 from `level0_all.json`, static top-10 |
| `l1plus-static` | Evolved level≥1 skills through the sharing sanitizer, static top-10 |
| `full-static` | Internal final library including retained L0, static top-10 |
| `full-dynamic` | Same internal library, hidden librarian: 8 candidates, top-3, up to 2 rounds |

The Sonnet run evaluates reuse on the same simulated tasks, as defined in §3.2.
Full rows follow Appendix L's internal comparison; L1+ implements the peer
release policy. The source baseline is also collected when skills must be
generated and `baseline` was omitted from the requested reporting conditions.

The driver prints the actual run directory and source artifact path. Use those
paths for subsequent commands; `<fingerprint>` below denotes that printed value:

```text
output/airline/runs/<fingerprint>/
  comparison_manifest.json
  summary.json, summary.md
  skills_from_train/
    phase1_manifest.json, sessions.json
    level0_all.json, level0_manifest.json
    round1_clusters.json, round1_skills.json, ...
    evolved_all.json
  condition_skills/
    l0-static.json, l1plus-static.json, full-static.json, full-dynamic.json
    private/l1plus_release_id_map.json
  logs/
```

Each session records `session_id`, `task_id`, `outcome`, `score`, traces and
task description. Internal skills retain IDs, source session and parent links.
Round cluster files store the fixed candidate population, including clusters
whose synthesis is rejected. Round skill files retain accepted children and
carried-forward originals. L1+ release payloads omit private lineage/history;
the local release-ID map remains inside `condition_skills/private/`.

tau2 simulations and actual selection logs are saved separately under
`$TAU2_ROOT/data/simulations/<run-name>/`, including
`results.json`, `skill_usage.jsonl`, and the runner's manifest.

## Fixed-library Qwen transfer

Set the source to the printed `skills_from_train` directory. Transfer must
reuse the Sonnet source library rather than extracting from Qwen's own traces:

```bash
export SOURCE_ARTIFACTS="output/airline/runs/<fingerprint>/skills_from_train"
export QWEN_CONSUMER_MODEL=YOUR_LITELLM_QWEN3_32B_MODEL

python scripts/run_tau2_comparison.py \
  --tau2-root "$TAU2_ROOT" --config configs/paper.yaml --domain airline \
  --source-artifacts-dir "$SOURCE_ARTIFACTS" \
  --source-model "$SONNET_TAU2_MODEL" --user-model "$SONNET_TAU2_MODEL" \
  --agent-model "$QWEN_CONSUMER_MODEL" \
  --conditions baseline,l0-static,l1plus-static,full-static,full-dynamic \
  --output-dir output/airline_qwen32b --dry-run
```

Quote real paths in shell assignments; replace `<fingerprint>` before execution.
Repeat with the Qwen3-8B consumer and a separate output directory.
The source manifest checks the saved simulator when available. Legacy source
artifacts can be read for transfer; `--source-model` explicitly records the
intended simulator. An external fixed source skips extraction/evolution.
`--skills-path <evolved_all.json>` also supports library reuse, but `l0-static`
additionally requires the original L0 via `--source-artifacts-dir`.

## AWM success-only baseline

AWM uses **original** L0 from successful Sonnet sessions, independently of which
L0 parents SkillLadder later retained. `--train-dir` is the run directory
containing `skills_from_train`, one level above `SOURCE_ARTIFACTS`:

```bash
python scripts/run_awm_baseline.py \
  --tau2-root "$TAU2_ROOT" --config configs/paper.yaml --domain airline \
  --train-dir "$(dirname "$SOURCE_ARTIFACTS")" \
  --skills-out output/awm/airline_success_l0.json --run-name awm_airline_sonnet \
  --agent-model "$SONNET_TAU2_MODEL" --user-model "$SONNET_TAU2_MODEL" \
  --dry-run
```

Remove `--dry-run` for a run. AWM uses MiniLM cosine retrieval, top-10, with
strict loading under paper YAML. Supply the same weights revision when running
the embedding baseline. Change only the consumer to reuse this pool for Qwen.
The standalone builder is
`python scripts/build_awm_skills.py <run-dir> <output-json>`.

## Resume and forced attempts

Manifests include code/content identity, tau2 revision, ordered task IDs,
effective parameters, role-specific models, prompt digests and input digests.
Session and cluster snapshots are fixed before processing. Atomic checkpoints
bind to input fingerprints and stable cluster IDs; changed input causes an
explicit error. Result reuse requires task/trial coverage, valid rewards/status,
and the matching manifest.

`--force-runs` and `--force-extract` create a separate attempt directory and
fresh run identity. They preserve earlier artifacts. Direct phase1 extraction
is available for an existing native baseline:

```bash
python scripts/run_tau2_phase1.py \
  --tau2-root "$TAU2_ROOT" --config configs/paper.yaml --domain airline \
  --run-name ACTUAL_TAU2_BASELINE_RUN \
  --skill-model "$SONNET_SKILL_MODEL" --output-dir output/airline_phase1
```

Use a fresh output directory for unbound legacy checkpoints. Seed 42 fixes
local randomization and saved membership, not commercial model responses.

Summaries preserve one row per simulation with task/trial/seed/reward/status.
They report both descriptive simulation-level reward SE and a separate SE over
task means. Missing reward is distinct from reward=0. Multiple trials whose
selection logs cannot identify a unique simulation pause quality feedback.
Runtime quality uses an independent reward EMA:
`q = 0.7 * reward_ema + 0.3 * success_rate` with success defined by reward>0.5.
Legacy runtime state needs complete history and a known EMA coefficient for
exact migration; otherwise the state is marked unresolved.

## Audits

Use the printed source artifact directory, with full private lineage available
locally. The following previews an independent DeepSeek rescoring run:

```bash
python audits/non_claude_judge_review.py \
  --provider deepseek --artifact-dir "$SOURCE_ARTIFACTS" \
  --levels 2,3 --tasks-per-domain 10 --seed 13 --dry-run
```

Repeat `--artifact-dir` for additional domains. Removing `--dry-run` requests
the configured auditor and writes child/strongest-parent pairs and summaries.
The other audit commands, their schemas and limitations are listed in
[PAPER_TO_CODE.md](PAPER_TO_CODE.md). The source-attribution and entity-audit
programs retain historical default input paths; set their module path lists to
the generated artifacts before use.

## Validated rerun environment

Local verification on **2026-11-07** used macOS arm64, Apple M5 Pro, 48 GiB RAM,
Python 3.12.14, and PortAudio 19.7. It checked package installation, all pytest
regressions, wheel/sdist build, and actual pinned tau2 imports/selectors/task
sets. CI defines equivalent offline jobs on Ubuntu 24.04/Python 3.12.

| Package | Local version |
| --- | --- |
| numpy | 2.5.4 |
| openai | 3.46.0 |
| pydantic | 2.14.3 |
| PyYAML | 6.0.3 |
| scikit-learn | 1.9.1 |
| sentence-transformers | 6.2.0 |
| torch | 2.15.0 |
| litellm | 1.108.2 |
| tau2 | 0.2.1-dev, pinned source revision |
| PyAudio | 0.2.14 |
| pytest | 9.2.0 |

The [full installed package snapshot](validation/2026-11-07-packages.txt)
records versions without credentials or local checkout paths. This is the
corrected-code validation environment, not a reconstructed paper environment.
No API calls or embedding weights were used; provider model IDs and weights
revision remain to be recorded during an authorized model-backed rerun.
