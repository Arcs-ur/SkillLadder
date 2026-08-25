# tau2-bench Integration

The SkillLadder agent patch targets this exact tau2-bench revision:

```text
01e812d1d1c4df6d8d2299bf175e482527112244
```

Prepare a separate checkout and apply the patch:

```bash
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench
git checkout 01e812d1d1c4df6d8d2299bf175e482527112244
git apply /path/to/SkillLadder/integrations/tau2/fedskill_tau2_integration.patch
python -m pip install -e .
```

The patch registers `fedskill_llm_agent` and implements static and dynamic
skill injection. Environment variables prefixed with `FEDSKILL_TAU2_` are kept
for compatibility with the experiment artifact and are set by the scripts in
this repository.

tau2-bench is not vendored here. Its MIT notice is reproduced in
`THIRD_PARTY_NOTICES.md`.
