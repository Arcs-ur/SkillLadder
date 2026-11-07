# Release Scope

This repository publishes the implementation code, parsers, audit scripts,
configuration, prompts, and exact task IDs/splits described in the paper.

It deliberately excludes:

- raw tau2 traces and simulation transcripts;
- extracted L0 skills and evolved L1-L3 skill artifacts;
- rewards, judge outputs, aggregate result tables, and logs;
- human-audit annotation files;
- provider credentials, internal endpoints, machine paths, and private model
  routing identifiers.

Reproduction creates these artifacts locally under ignored output directories.
The tau2 checkout also writes simulations under its own `data/simulations/`;
that external directory is not covered by this repository's ignore rules.

L0 skills are internal to extraction/evolution. Shared L1+ payloads contain
sanitized skill content and retrieval fields with independent release IDs.
Private source IDs, session IDs, parent links, runtime history, and release-ID
maps remain local. The `Full` paper comparison conditions may retrieve retained
internal L0, as defined in Appendix L; their files are not peer-release payloads.

The current source is for corrected reruns. Historical paper requirements and
reported values remain historical. New runs store their code/configuration,
model roles, task IDs, prompt/input digests and separate attempt identities.
The version snapshot in `docs/validation/` contains dependency metadata only.

CI publishes no experiment outputs and requires no provider credentials.
Wheel files contain the Python package; the source distribution also includes
scripts, configurations, prompts, documentation, tests and the tau2 patch.
