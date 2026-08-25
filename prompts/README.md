# Prompt Index

The authoritative prompt templates are disclosed in place so the code and
prompt used at runtime cannot drift apart.

| Stage | Location | Constants |
| --- | --- | --- |
| success/failure L0 extraction | `src/fedskill/extractor/skill_extractor.py` | `EXTRACT_SYSTEM`, `ERROR_ANALYST_SYSTEM`, `SUCCESS_ANALYST_SYSTEM`, `EXTRACT_USER` |
| open-category discovery | `src/fedskill/clustering/hybrid_clustering.py` | `DISCOVER_SYSTEM` |
| alternative clustering | `src/fedskill/clustering/llm_clustering.py`, `tag_clustering.py`, `reverse_hybrid.py` | module-level system templates |
| synthesis and five quality gates | `src/fedskill/server/evolver.py` | `EVOLVE_SYSTEM`, `CONSISTENCY_SYSTEM`, `STEP_PRESERVATION_SYSTEM`, `SCORE_SYSTEM`, user templates |
| runtime agent and dynamic librarian | `integrations/tau2/fedskill_tau2_integration.patch` | `FEDSKILL_*_PROMPT` constants |
| source-attribution and entity audits | `audits/` | `SYSTEM_PROMPT`, `USER_PROMPT`, and related module constants |

The paper protocol uses these fixed templates. Cross-family audits change the
model provider, not the synthesis/gate prompt text.
