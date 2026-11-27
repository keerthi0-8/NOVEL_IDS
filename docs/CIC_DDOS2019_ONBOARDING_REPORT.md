# CIC-DDoS2019 onboarding mapping report ? second run preflight

Mapping was prepared from CICFlowMeter documentation and unlabeled feature descriptors before any second-run predictions. No CIC labels or model performance were used to define the mapping.

- Source available locally: existing aggregate CIC sample from 13 CSV files, 162,276 rows. Individual original source filenames were not preserved in the aggregate.
- Mapping input: 512 unlabeled rows used to calculate descriptor statistics and similarity; raw values are not changed in the source files.
- Feature columns: 86 (original CIC column names, including spaces).
- Level 1 clear equivalents: 4 ? Flow Duration ? flow_duration (microseconds?seconds); Protocol ? transport_protocol (IANA number?name); Total Length of Fwd Packets ? source_to_destination_bytes; Total Length of Bwd Packets ? destination_to_source_bytes.
- Level 2 group-only: 72 ? packet counts, rates, inter-arrival times, TCP flags, window sizes, packet/segment/header length statistics, subflow/bulk features, and directional ratio. These are related groups only, never feature equivalences.
- Level 3 dataset-specific: 10 ? identifiers, endpoint metadata and other fields without an approved source mapping.
- Source common features missing: network_service (CIC has no direct service field).
- Compatibility score: 0.465116 = (4 + 0.5?72) / 86.
- Trust level from configured thresholds: Medium (0.40?0.75); this is a dataset-level compatibility description, not a model confidence threshold.
- Declared transforms: microseconds?seconds for duration/IAT/active/idle fields; IANA protocol number?protocol name.

Mapping SHA-256: 6cccab77a7892d469731955a519405a6eb1d4b370a3cacfbc7aa6a31f957eb59
Metadata SHA-256: 650644352304dc46270e1ae2a8167486460693117338faac22289a82a235d6c9
