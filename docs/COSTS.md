# Cost and Runtime Notes

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
the system prompt. Dynamic retrieval reduces prompt volume but adds two hidden
LLM calls per agent turn and approximately doubled wall time in this audit.
These numbers exclude one-off extraction and evolution.

Before a full run, use a small task-ID file, low concurrency, and provider
budget limits. Smaller/local judges, fewer gate-evaluation tasks, cascaded
gates, caching, and embedding-only clustering are the main cost controls.
