"""Project-wide constants kept in one place to avoid incompatible sentinels."""

SCHEMA_VERSION = "geoflowagent.processed.v2"
CACHE_VERSION = "geoflowagent.embedding-cache.v3"
CHECKPOINT_VERSION = "geoflowagent.checkpoint.v3"
# Bump this whenever serialization, missing-modality handling, or field/background
# pooling semantics change.  Embedding YAML hashes alone cannot detect those code
# changes, so caches and downstream checkpoints bind this value explicitly.
FEATURE_SPEC_VERSION = "geoflowagent.feature-spec.v1"
STOP_TOOL_ID = "<STOP>"
PAD_TOOL_ID = "<PAD>"
INVALID_REGRET = 1.0
