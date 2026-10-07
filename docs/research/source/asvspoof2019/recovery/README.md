# ASVspoof 2019 LA recovery evidence

This directory records the small, read-only evidence used to recover a stalled
official LA.zip download. `zenodo-record-6906306.json` is the unmodified Zenodo
record API response. The other JSON files retain only recovery-relevant facts.

The Zenodo object must still be validated by a complete local MD5 before it is
treated as the original `LA.zip`. No file here proves a completed download or
turns a third-party source into an officially endorsed mirror.

Recovery completed on 2026-10-07 at 07:22:43 UTC. The exact executed
`range-transfer/recover_tail.py` and raw `range-transfer/execution.json` record
three validated tail ranges, preserved-prefix SHA-256, full size 7,640,952,520
bytes, and matching official MD5 `30c98f11d8b2bc21f2c257bfd78bb5c5`.
The completed archive SHA-256 is
`208a7e4e3913f8c75ae1afd19bf32a5b29ae68435e9e30e23e5e98b6a155e4ec`.
The earlier metadata/probe files remain unchanged historical evidence; their
pending status predates this transfer. Media extraction, protocol validation and
decode audits are separate downstream checks.
