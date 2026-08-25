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
Do not commit them without conducting a separate privacy, licensing, and data
release review.
