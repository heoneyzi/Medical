# Third-party data and code provenance

## FlowAgent / FlowPlan

- Paper: [Tools as Continuous Flow for Evolving Agentic Reasoning](https://arxiv.org/html/2605.07339v2)
- Code: [ssy166/FlowPlan](https://github.com/ssy166/FlowPlan)
- Audited commit: `c65155b85931e5d38c38573c8e4f84d31b056c45`
- Use here: conceptual and interface reference only; no source vendored.
- Release note: no root license file was visible at the audited commit, so copying was intentionally avoided.

## MARRVEL-MCP benchmark

- Dataset: [hjeong84/marrvel-mcp-benchmark-data](https://huggingface.co/datasets/hjeong84/marrvel-mcp-benchmark-data)
- Recommended pinned revision: `8e895924b1b19dc5ede19669bbf66c2d8d2eb5f0`
- Dataset card label: MIT
- Use here: optional download into a local curation queue.

## MARRVEL-MCP code

- Repository: [hyunhwan-bcm/MARRVEL_MCP](https://github.com/hyunhwan-bcm/MARRVEL_MCP)
- Audited commit: `6deda877b1720431e4e99ce4b7df67a090476251`
- Use here: tool names/concepts may guide independent typed wrappers after human review; no source vendored.
- Release note: confirm repository and upstream-database terms before redistribution.

## Frozen model families

Model names in the configs refer to Hugging Face models downloaded at run time; the configs pin these revisions:

- Qwen2.5-1.5B-Instruct (MVP default): `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`;
- Qwen2.5-7B-Instruct (capacity ablation): `a09a35458c702b33eeacc393d103063234e8bc28`;
- MedCPT Query: `d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc`;
- MedCPT Article: `d05a736da4bb84ee4057b7f7999485be6ed85465`;
- SapBERT: `090663c3ae57bf35ffe4d0d468a2a88d03051a4d`;
- DNABERT-2: `7bce263b15377fc15361f52cfab88f8b586abda0`.

Review each model card, license, and—especially for DNABERT-2—pinned remote code before downloading or publishing derived caches. A commit pin improves reproducibility; it is not a security or redistribution approval.

The DNABERT-2 remote implementation imports `einops`; the `hf` extra includes it explicitly. The
example environment constrains `transformers` to the 4.x line because this pinned remote code was
written against that API family. Still review the exact remote-code diff and run the encoder smoke
check in an isolated environment before a real extraction.

## Synthetic fixture

All records under `data/raw/smoke/` are invented, contain no patient data, and use synthetic gene/transcript/contig/HPO identifiers. They exist only to test code paths.
