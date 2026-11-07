# tau2-bench Integration

The SkillLadder agent patch targets this exact tau2-bench revision:

```text
01e812d1d1c4df6d8d2299bf175e482527112244
```

Run these commands from the SkillLadder repository root with its Python
environment active. Use a separate tau2 checkout:

```bash
export TAU2_ROOT=/absolute/path/to/tau2-bench
git clone https://github.com/sierra-research/tau2-bench.git "$TAU2_ROOT"
git -C "$TAU2_ROOT" checkout 01e812d1d1c4df6d8d2299bf175e482527112244
git -C "$TAU2_ROOT" apply --check "$PWD/integrations/tau2/fedskill_tau2_integration.patch"
git -C "$TAU2_ROOT" apply "$PWD/integrations/tau2/fedskill_tau2_integration.patch"
```

The pinned tau2 imports its voice modules even for text-only runs. Install
PortAudio headers before its `voice` extra; the `knowledge` extra supports
banking knowledge tasks:

```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev
# macOS with Homebrew
brew install portaudio

python -m pip install -e "${TAU2_ROOT}[voice,knowledge]"
```

Use the native-package command for your OS. The local macOS validation used
PortAudio 19.7 built from source; Linux CI installs `portaudio19-dev`.

The patch registers `fedskill_llm_agent` and implements static and dynamic
skill injection, retirement filtering, deduplication, quality-aware retrieval,
and logs of the actual selection. Run the check without credentials:

```bash
python scripts/check_tau2_integration.py --tau2-root "$TAU2_ROOT"
```

It verifies the revision and applied patch, imports the registered agent, checks
retired/duplicate/zero-quality skills and actual selection logs, and loads all
four exact task sets with socket connections and model generation blocked.
It uses token-overlap retrieval and downloads no embedding weights.

Environment variables prefixed with `FEDSKILL_TAU2_` retain compatibility with
historical artifacts. The drivers set them from the effective configuration:
`SKILLS_PATH`, `INJECTION_MODE`, `SKILL_LOG_PATH`, `MAX_SKILLS`,
`EMBEDDING_MODEL`, `EMBEDDING_REVISION`, and `STRICT_EMBEDDINGS`, among others.
`--strict-embeddings` fails if the requested embedding model cannot load rather
than changing the retrieval/clustering method silently.

Telecom uses `--task-set-name telecom --task-split-name full`. At this revision,
the `telecom_full` registry alias has a wrapper incompatible with the runner's
split keyword; the explicit set/split pair loads the same 200 paper task IDs.
`configs/paper.yaml` supplies this pair automatically.

tau2-bench is not vendored here. Its MIT notice is reproduced in
`THIRD_PARTY_NOTICES.md`.
