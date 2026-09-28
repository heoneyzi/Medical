# Model smoke-test records

A model is marked runnable only after both checks below succeed:

1. registry preflight verifies the executable, pinned official repository commit, checkpoint and
   required runtime assets;
2. one real Replogle-derived control H5AD completes through the official model command and the
   standardized artifact validator.

Environment/checkpoint availability is machine-specific. `preflight.json`, per-context
`metadata.json`, and `run.log` under the selected artifact root are the authoritative run records;
documentation must not turn an unavailable model into an enrolled one.
