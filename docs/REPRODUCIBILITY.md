# Reproducibility Guide

## 1. Environment

Use Python 3.12, install this repository in editable mode, and configure the
variables in `.env.example`. The core client accepts any OpenAI-compatible chat
completion endpoint. The tau2 runner uses LiteLLM model identifiers; exact
provider identifiers differ by deployment and are therefore supplied on the
command line rather than hard-coded.

`requirements-paper.txt` pins the direct packages in the verified artifact
environment. Install that file first when reproducing the paper setup; use the
ranges in `pyproject.toml` when integrating SkillLadder into another project.

The paper used Claude Sonnet 4.6 for the main agent, user simulator,
synthesizer, and default judge. Qwen3-32B and Qwen3-8B were the transfer target
agents, while the user simulator remained Claude Sonnet 4.6. DeepSeek-v3.2 and
GLM-5.2 were used in the cross-family judge audits. `configs/paper.yaml` records
the provider-independent names and all shared hyperparameters.

## 2. Prepare tau2-bench

Follow `integrations/tau2/README.md`. Do not apply the patch to an arbitrary
tau2 version: its agent and registry interfaces are pinned to the disclosed
commit.

## 3. Select the Exact Tasks

Use `--task-ids-file` with one of the four ordered JSON lists:

```text
configs/task_splits/airline_50.json
configs/task_splits/banking_knowledge_97.json
configs/task_splits/retail_114.json
configs/task_splits/telecom_200.json
```

These lists contain IDs only. The benchmark's task definitions, policies,
databases, and simulators come from the pinned tau2 checkout.

## 4. Run Baseline, Extraction, and Skill Injection

`scripts/run_tau2_comparison.py` performs the restartable sequence used by the
main experiment:

1. Run the no-skill `llm_agent` baseline.
2. Parse the resulting simulations and extract success/failure-conditioned L0
   skills.
3. Run three rounds of hybrid clustering and quality-gated synthesis.
4. Run static top-10 reinjection.
5. Run the dynamic librarian with 8 candidates, top-3 return, and at most two
   retrieval rounds.

Example:

```bash
python scripts/run_tau2_comparison.py \
  --tau2-root "$TAU2_ROOT" \
  --domain retail \
  --task-ids-file configs/task_splits/retail_114.json \
  --agent-model "$SKILLLADDER_AGENT_MODEL" \
  --user-model "$SKILLLADDER_USER_MODEL" \
  --evolve-rounds 3 \
  --max-steps 30 \
  --output-dir output/retail
```

To rerun extraction/evolution from an existing tau2 simulation, invoke
`scripts/run_tau2_phase1.py` with the simulation's `--run-name`. The AWM
success-only L0 baseline is available through `scripts/run_awm_baseline.py`.

## 5. Audits and Ablations

The scripts under `audits/` intentionally take locally generated skill and
simulation files as inputs. No result files are bundled. Run each tool with
`--help` before use; API-backed judge tools accept provider, base URL, model,
and API-key environment options. The human entity-retention workflow separates
audit-set preparation, model pre-labeling, browser annotation, and scoring.

## Determinism and Interpretation

Task IDs and numeric defaults are fixed, but API model revisions, stochastic
sampling, hardware, provider routing, and benchmark dependency versions can
affect exact rewards and judge scores. The disclosed code reproduces the
protocol; it does not assert bit-for-bit reproduction across provider stacks.
