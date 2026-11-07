# Cost and Runtime Notes

The values below are historical paper measurements from Appendix J/Table 17.
They were not remeasured during the corrected-code offline validation.

API-backed reproduction can be expensive. For an incremental window with
`M` new trace/L0 artifacts, at most `C` candidate clusters per round, and `R`
rounds, the dominant evolution-stage LLM-call count is `O(M + R*C)`. Token
volume also depends on artifact length and the gate evaluation set size.

The paper's DeepSeek-v3.2 audit of 10 round-stratified airline R2/R3 candidates,
scored against 10 task descriptions, used 30,236 tokens per candidate on
average (median 28,897; IQR 23,513-40,170), including retries. The score gate
accounted for 52.0% of tokens and synthesis for 34.3%.

On the sequential airline-50 Sonnet runs, the per-task measurements were:

| Condition | Turns | First-prompt tokens | Avg input tokens | Avg output tokens | Wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 8.9 | 5,578 | 67,503 | 1,406 | 190.8 |
| L0-only static | 9.3 | 7,864 | 95,053 | 1,547 | 195.3 |
| Full static | 9.3 | 14,240 | 156,353 | 1,724 | 196.7 |
| Dynamic librarian | 9.0 | 8,566 | 89,411 | 1,507 | 392.2 |

Static injection increases prompt mass because the skill block is resent with
the system prompt. Dynamic retrieval reduces prompt volume but the hidden
controller/judge calls approximately doubled wall time in this audit.
The implemented controller may skip retrieval, or request a second judge round;
budget for the actual consult path rather than assuming exactly two calls.
These numbers exclude one-off extraction and evolution.

The default five-condition driver evaluates five episodes per task and trial,
and reuses the no-skill source baseline for reporting. A fixed-library transfer
still evaluates a new consumer baseline when requested but skips new source
extraction/evolution. AWM is a separate run. Model-call retries can multiply
the cost of an extraction or synthesis attempt.

Start with the comparison driver's `--dry-run`, then use explicit task IDs or
a small `--task-ids-file` and low concurrency for a budgeted pilot. The pipeline
does not implement a hard monetary limit; enforce it at the provider.
Log provider IDs, request counts, input/output tokens, retry counts and wall
time with each rerun. Different task budgets, judges, gate task subsets or
clustering strategies define a different experiment and should be labeled.

Evolution token usage can be measured from saved candidate clusters:

```bash
python audits/evolution_token_audit.py \
  --provider deepseek --artifact-dir "$SOURCE_ARTIFACTS" \
  --rounds 2,3 --samples-per-round 5 --tasks 10 --seed 42 \
  --output-root output/evolution_token_audit --dry-run
```

This audit's preview reads local artifacts without API calls. Remove
`--dry-run` to collect provider-reported tokens. Offline tests and the pinned
tau2 integration check make no model requests and download no model weights.
