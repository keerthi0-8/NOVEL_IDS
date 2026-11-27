# NOVEL-IDS taxonomy proposal

**Status: PROPOSED — NOT APPROVED FOR TRAINING OR EVALUATION**  
**Scope:** evidence review for a future harmonized UNSW-NB15 + NSL-KDD taxonomy. This document does not activate a taxonomy, alter labels, or authorize training. All candidate mappings below remain proposals.

## 1. Evidence and count provenance

Counts were read from the existing versioned unified artifacts, not estimated from published aggregate tables:

- NSL-KDD: `data/processed/unified_representation/nsl_kdd_unified_v2_no_kddtest_feature_overlap.joblib`.
- UNSW-NB15: `data/processed/unified_representation/unsw_nb15_unified_v2_no_test_feature_overlap.joblib`.
- Train and validation counts use each artifact's `train.labels` and `validation.labels`; test counts use `test_full.labels`.
- The artifacts retain the raw fine-grained source labels. No label grouping was applied to produce these counts.
- The NSL-KDD v2 validation set removes 121 validation rows whose exact 41-feature vectors overlap KDDTest+. The UNSW-NB15 v2 validation set removes 2,069 validation rows whose exact 42-feature vectors overlap the official test partition. These rows were excluded from validation, not moved into training.
- Thus the current v2 NSL-KDD train+validation rows total 125,852 (100,778 + 25,074), compared with 125,973 official KDDTrain+ rows. UNSW-NB15 train+validation totals 173,272 (140,273 + 32,999), compared with 175,341 official training rows. Keep these provenance differences visible when reporting metrics.
- Test counts are the full official partitions in the project artifacts: NSL-KDD KDDTest+ 22,544; UNSW-NB15 test 82,332. The configured file names are counter-intuitive: the local UNSW file named `UNSW_NB15_testing-set.csv` contains 175,341 canonical training rows, while `UNSW_NB15_training-set.csv` contains 82,332 canonical test rows. This follows the already confirmed project split assignment.

### 1.1 UNSW-NB15 fine-label counts

| Original attack_cat | Train | Validation | Test |
|---|---:|---:|---:|
| Normal | 44,800 | 10,871 | 37,000 |
| Generic | 32,000 | 7,281 | 18,871 |
| Exploits | 26,714 | 6,503 | 11,132 |
| Fuzzers | 14,548 | 3,494 | 6,062 |
| DoS | 9,811 | 2,357 | 4,089 |
| Reconnaissance | 8,393 | 1,594 | 3,496 |
| Analysis | 1,600 | 361 | 677 |
| Backdoor | 1,397 | 293 | 583 |
| Shellcode | 906 | 220 | 378 |
| Worms | 104 | 25 | 44 |
| **Total** | **140,273** | **32,999** | **82,332** |

No UNSW validation class is below 20 rows. Worms has only 25 validation rows and should nevertheless be interpreted cautiously.

### 1.2 NSL-KDD fine-label counts

| Original attack label | Train | Validation | KDDTest+ full |
|---|---:|---:|---:|
| normal | 53,875 | 13,446 | 9,711 |
| neptune | 32,971 | 8,212 | 4,657 |
| ipsweep | 2,879 | 706 | 141 |
| satan | 2,906 | 723 | 735 |
| portsweep | 2,345 | 582 | 157 |
| smurf | 2,117 | 495 | 665 |
| back | 764 | 191 | 359 |
| warezclient | 712 | 178 | 0 |
| teardrop | 714 | 175 | 12 |
| warezmaster | 16 | 4 | 944 |
| guess_passwd | 42 | 11 | 1,231 |
| pod | 161 | 33 | 41 |
| nmap | 1,195 | 298 | 73 |
| land | 15 | 2 | 7 |
| buffer_overflow | 24 | 6 | 20 |
| ftp_write | 6 | 2 | 3 |
| imap | 8 | 3 | 1 |
| multihop | 6 | 1 | 18 |
| loadmodule | 7 | 2 | 2 |
| rootkit | 8 | 2 | 13 |
| phf | 3 | 1 | 2 |
| spy | 1 | 1 | 0 |
| perl | 3 | 0 | 2 |
| apache2 | 0 | 0 | 737 |
| httptunnel | 0 | 0 | 133 |
| mailbomb | 0 | 0 | 293 |
| mscan | 0 | 0 | 996 |
| named | 0 | 0 | 17 |
| processtable | 0 | 0 | 685 |
| ps | 0 | 0 | 15 |
| saint | 0 | 0 | 319 |
| sendmail | 0 | 0 | 14 |
| snmpgetattack | 0 | 0 | 178 |
| snmpguess | 0 | 0 | 331 |
| sqlattack | 0 | 0 | 2 |
| udpstorm | 0 | 0 | 2 |
| worm | 0 | 0 | 2 |
| xlock | 0 | 0 | 9 |
| xsnoop | 0 | 0 | 4 |
| xterm | 0 | 0 | 13 |
| **Total rows** | **100,778** | **25,074** | **22,544** |

**Validation support warning:** 11 present fine labels have fewer than 20 validation rows: buffer_overflow (6), ftp_write (2), guess_passwd (11), imap (3), land (2), loadmodule (2), multihop (1), phf (1), rootkit (2), spy (1), warezmaster (4). The training class perl has zero validation rows. In addition, many KDDTest+ fine attack labels are absent from source training; these are natural unseen attack types and must not be represented as supervised training classes just because they occur in test.

The NSL-KDD table above is a raw-label count table. It intentionally does not collapse labels into DoS/Probe/R2L/U2R.

## 2. Candidate shared classes and mapping evidence

### Normal → Normal (candidate merge)

Both datasets explicitly contain a benign/normal category. UNSW-NB15 documents normal activity alongside its nine attack categories. NSL-KDD uses `normal` as its benign label. This is the strongest semantic candidate, but the datasets' traffic generation and feature distributions differ. A shared label does not imply identical distributions.

### UNSW DoS + NSL-KDD DoS family → DoS (candidate merge)

The UNSW-NB15 dataset description explicitly lists DoS as an attack category. The local NSL-KDD `Attack Types.csv` maps back, land, neptune, pod, smurf and teardrop to `dos`. NSL-KDD family references also describe DoS as attacks against availability/resource exhaustion. This is a plausible semantic merge. Caveat: attack implementations, time periods, traffic-generation tools, and feature distributions are not equivalent; validation must measure transfer rather than assume it.

### UNSW Reconnaissance + NSL-KDD Probe family → Probe (candidate merge)

UNSW-NB15 describes Reconnaissance as information gathering about a target. The NSL-KDD local `Attack Types.csv` maps ipsweep, nmap, portsweep and satan to `probe`; the NSL-KDD family literature describes probing/scanning to gather information. This is a defensible functional relationship, not an identity. Caveat: Reconnaissance can include discovery behaviors that do not correspond one-to-one to every NSL-KDD Probe subtype. Keep this mapping proposed and evaluate per-dataset/per-class performance.

### Dataset-specific classes (candidate preservation)

- UNSW-NB15: `UNSW_NB15::Generic`, `UNSW_NB15::Exploits`, `UNSW_NB15::Fuzzers`, `UNSW_NB15::Analysis`, `UNSW_NB15::Backdoor`, `UNSW_NB15::Shellcode`, `UNSW_NB15::Worms`. Their published descriptions identify distinct attack behaviors and there is no sufficiently justified one-to-one counterpart in the proposed shared Normal/DoS/Probe set.
- NSL-KDD: `NSL_KDD::R2L` and `NSL_KDD::U2R` are candidates to remain dataset-specific because the current UNSW-NB15 nine-category taxonomy has no clearly equivalent R2L or U2R family. Preserve their fine-grained source labels in provenance.
- NSL-KDD test-only/new fine labels need careful family assignment. The local `Attack Types.csv` covers the legacy label set, not every newer KDDTest+ label. Published mappings for some newer labels vary; for example, `worm` and `httptunnel` are not consistently assigned across secondary sources. Until a source-specific mapping is approved, retain such labels as `NSL_KDD::<original_label>` in a fine-grained proposal or mark family mapping as unresolved. Never infer a family from its test count or model output.

## 3. NSL-KDD label granularity decision: alternatives

### Fine-grained NSL-KDD labels

**Pros:** retains the original attack identity; exposes rare and test-only attack behavior; avoids potentially incorrect family mappings; makes open-set/novel-attack analysis more transparent.  
**Cons:** many classes have extremely small support; several classes have zero training examples and therefore cannot be learned as supervised classes; severe class imbalance makes stable per-class calibration and macro metrics difficult; a shared taxonomy with UNSW's broader categories becomes much larger and less balanced.

### NSL-KDD's five labels (Normal + DoS + Probe + R2L + U2R)

**Pros:** aligns with the commonly used NSL-KDD family view; reduces label sparsity; family-level mapping makes class-wise evaluation more statistically stable for R2L/U2R than individual fine labels; the local Attack Types.csv provides an explicit mapping for its legacy labels.  
**Cons:** hides differences among attack subtypes; test-only fine attacks need defensible family assignments; R2L and U2R have no clear UNSW-NB15 counterparts and remain dataset-specific; family-level labels can conceal failure to detect a novel subtype within a known family.

This is an unresolved design choice. No granularity is selected or approved by this proposal.

## 4. Candidate taxonomy outline (all mappings PROPOSED)

| Candidate unified class | Candidate source labels | Status and caveat |
|---|---|---|
| Normal | UNSW-NB15 Normal; NSL-KDD normal | PROPOSED; benign semantics align, distributions may not |
| DoS | UNSW-NB15 DoS; NSL-KDD labels explicitly mapped to DoS family | PROPOSED; verify all extended labels before including them |
| Probe | UNSW-NB15 Reconnaissance; NSL-KDD Probe family labels | PROPOSED; functional similarity, not exact equivalence |
| NSL_KDD::R2L | NSL-KDD R2L family | PROPOSED dataset-specific class |
| NSL_KDD::U2R | NSL-KDD U2R family | PROPOSED dataset-specific class |
| UNSW_NB15::Generic | UNSW-NB15 Generic | PROPOSED dataset-specific class |
| UNSW_NB15::Exploits | UNSW-NB15 Exploits | PROPOSED dataset-specific class |
| UNSW_NB15::Fuzzers | UNSW-NB15 Fuzzers | PROPOSED dataset-specific class |
| UNSW_NB15::Analysis | UNSW-NB15 Analysis | PROPOSED dataset-specific class |
| UNSW_NB15::Backdoor | UNSW-NB15 Backdoor | PROPOSED dataset-specific class |
| UNSW_NB15::Shellcode | UNSW-NB15 Shellcode | PROPOSED dataset-specific class |
| UNSW_NB15::Worms | UNSW-NB15 Worms | PROPOSED dataset-specific class |
| NSL_KDD::<fine_label> | NSL-KDD labels without accepted family evidence | PROPOSED pending label-by-label review |

The outline is illustrative only. It is not a valid active taxonomy file, and none of its classes or mappings may be used for training/evaluation until separately approved and encoded in an external taxonomy document with all mapping statuses approved.

## 5. Evidence references

1. UNSW Canberra, **The UNSW-NB15 Dataset** (official dataset page): https://research.unsw.edu.au/projects/unsw-nb15-dataset — describes the nine attack categories and dataset generation context.
2. Local source file, **NSL-KDD/Attack Types.csv** (the user's existing dataset folder): maps legacy NSL-KDD attack labels to normal, DoS, Probe, R2L and U2R. It does not cover all newer KDDTest+ labels.
3. NSL-KDD attack-family overview and original/fine attack examples: https://journals.sagepub.com/doi/10.1089/big.2020.0263 — documents four families and highlights attacks appearing only in the test set.
4. NSL-KDD family definitions and label grouping overview: https://pmc.ncbi.nlm.nih.gov/articles/PMC11177870/ — useful secondary reference; some secondary mappings differ, so ambiguous labels remain unresolved here.
5. UNSW-NB15 category definitions and descriptions: https://www.unb.ca/cic/datasets/cic-unsw-nb15.html — supplementary category descriptions; the UNSW Canberra dataset page remains the primary source for category names.

## 6. Approval required before activation

1. Select NSL-KDD fine-grained labels or its five-family granularity.
2. Approve or reject Normal, DoS, and Probe/Reconnaissance candidate merges individually.
3. Review every dataset-specific class and every extended NSL-KDD label mapping, especially conflicting secondary-source assignments.
4. Approve the complete class list and every original-label mapping. The taxonomy loader will reject training/evaluation if the overall status or any mapping remains proposed.
5. Only after approval, place a separately reviewed taxonomy file at an explicitly selected path. No active taxonomy file is created in config by this Stage 11 implementation.

**Final status: PROPOSED. No class mapping is approved.**
