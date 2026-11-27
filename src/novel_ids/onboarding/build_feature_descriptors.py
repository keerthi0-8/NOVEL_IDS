
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INVENTORY_PATH = (
    PROJECT_ROOT / "results" / "feature_inventory.csv"
)

REGISTRY_PATH = (
    PROJECT_ROOT
    / "src" / "novel_ids" / "onboarding"
    / "semantic_feature_registry.yaml"
)

NSL_METADATA_PATH = Path(
    r"G:\My Drive\NSL-KDD\NSL_KDD-master\Field Names.csv"
)

UNSW_METADATA_PATH = Path(
    r"G:\My Drive\UNSW-NB15\NUSW-NB15_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "src" / "novel_ids" / "onboarding"
    / "feature_descriptors.yaml"
)


# Level 2 semantic groups.
# These represent broad relationships, NOT equivalent features.
# Every feature retains its original identity and value.

LEVEL2_GROUPS = {

    "connection_state": {
        "nsl_kdd": ["flag"],
        "unsw_nb15": ["state"],
        "description": (
            "Categorical connection or protocol state measurements."
        ),
        "evidence": (
            "NSL-KDD flag describes connection status; UNSW-NB15 "
            "state describes protocol-dependent state. "
            "Related, not equivalent."
        ),
    },

    "authentication_activity": {
        "nsl_kdd": [
            "num_failed_logins",
            "logged_in",
            "is_host_login",
            "is_guest_login",
            "root_shell",
            "su_attempted",
        ],
        "unsw_nb15": ["is_ftp_login"],
        "description": (
            "Login, authentication, and privilege-related indicators."
        ),
        "evidence": (
            "Feature definitions identify login status, failed logins, "
            "host/guest access, shell privileges, and FTP login."
        ),
    },

    "host_and_privilege_activity": {
        "nsl_kdd": [
            "hot",
            "num_compromised",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
        ],
        "unsw_nb15": [],
        "description": (
            "Host activity, compromise, and privilege-related measurements."
        ),
        "evidence": (
            "NSL-KDD feature definitions describe hot indicators, "
            "compromised conditions, root activity, file creation, "
            "shells, and access files."
        ),
    },

    "application_protocol_activity": {
        "nsl_kdd": [],
        "unsw_nb15": [
            "ct_ftp_cmd",
            "ct_flw_http_mthd",
            "trans_depth",
        ],
        "description": (
            "Application-protocol commands, methods, and transaction depth."
        ),
        "evidence": (
            "UNSW-NB15 metadata describes FTP commands, HTTP methods, "
            "and HTTP transaction depth."
        ),
    },

    "application_content_size": {
        "nsl_kdd": [],
        "unsw_nb15": ["response_body_len"],
        "description": (
            "Application response content-size measurement."
        ),
        "evidence": (
            "UNSW-NB15 metadata describes the uncompressed "
            "response-body content size."
        ),
    },

    "connection_history_count": {
        "nsl_kdd": [
            "count",
            "srv_count",
            "dst_host_count",
            "dst_host_srv_count",
        ],
        "unsw_nb15": [
            "ct_srv_src",
            "ct_dst_ltm",
            "ct_src_dport_ltm",
            "ct_dst_sport_ltm",
            "ct_dst_src_ltm",
            "ct_src_ltm",
            "ct_srv_dst",
        ],
        "description": (
            "Connection and connection-history count measurements."
        ),
        "evidence": (
            "Both datasets contain connection-count features, but "
            "their observation windows and counting conditions differ. "
            "Group-only relationship."
        ),
    },

    "connection_error_rate": {
        "nsl_kdd": [
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
        ],
        "unsw_nb15": [],
        "description": (
            "Connection error-rate measurements."
        ),
        "evidence": (
            "NSL-KDD definitions identify SYN-error and REJ-error "
            "rates. These remain distinct measurements."
        ),
    },

    "connection_context_ratio": {
        "nsl_kdd": [
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
        ],
        "unsw_nb15": [],
        "description": (
            "Connection-context ratios involving services, hosts, "
            "and source ports."
        ),
        "evidence": (
            "NSL-KDD definitions describe ratios involving services, "
            "hosts, and source ports. Their individual denominators "
            "and meanings must remain distinct."
        ),
    },

    "bandwidth_load": {
        "nsl_kdd": [],
        "unsw_nb15": [
            "sload",
            "dload",
        ],
        "description": (
            "Directional network-load measurements."
        ),
        "evidence": (
            "UNSW-NB15 metadata identifies source and destination "
            "load in bits per second."
        ),
    },

    "packet_volume": {
        "nsl_kdd": [],
        "unsw_nb15": [
            "spkts",
            "dpkts",
        ],
        "description": (
            "Directional packet-count measurements."
        ),
        "evidence": (
            "UNSW-NB15 metadata identifies source-to-destination "
            "and destination-to-source packet counts."
        ),
    },

    "ttl_measurement": {
        "nsl_kdd": [],
        "unsw_nb15": [
            "sttl",
            "dttl",
        ],
        "description": (
            "Directional IP time-to-live measurements."
        ),
        "evidence": (
            "UNSW-NB15 metadata identifies source and destination "
            "TTL values."
        ),
    },

    "packet_loss_or_retransmission": {
        "nsl_kdd": [],
        "unsw_nb15": [
            "sloss",
            "dloss",
        ],
        "description": (
            "Directional packet-loss or retransmission measurements."
        ),
        "evidence": (
            "UNSW-NB15 metadata describes source and destination "
            "packets retransmitted or dropped."
        ),
    },

    "packet_interarrival_time": {
        "nsl_kdd": [],
        "unsw_nb15": [
            "sinpkt",
            "dinpkt",
        ],
        "description": (
            "Directional packet inter-arrival-time measurements."
        ),
        "evidence": (
            "UNSW-NB15 metadata identifies source and destination "
            "inter-packet arrival times in milliseconds."
        ),
    },

    "packet_jitter": {
        "nsl_kdd": [],
        "unsw_nb15": [
            "sjit",
            "djit",
        ],
        "description": (
            "Directional packet-jitter measurements."
        ),
        "evidence": (
            "UNSW-NB15 metadata identifies source and destination "
            "jitter in milliseconds."
        ),
    },

    "tcp_handshake_timing": {
        "nsl_kdd": [],
        "unsw_nb15": [
            "tcprtt",
            "synack",
            "ackdat",
        ],
        "description": (
            "TCP connection setup and handshake timing measurements."
        ),
        "evidence": (
            "UNSW-NB15 metadata describes TCP round-trip time and "
            "separate handshake intervals. These are related but "
            "distinct measurements."
        ),
    },

    "tcp_window_measurement": {
        "nsl_kdd": [],
        "unsw_nb15": [
            "swin",
            "dwin",
        ],
        "description": (
            "Directional TCP window advertisement measurements."
        ),
        "evidence": (
            "UNSW-NB15 metadata identifies source and destination "
            "TCP window advertisement values."
        ),
    },

    "packet_size_measurement": {
        "nsl_kdd": [],
        "unsw_nb15": [
            "smean",
            "dmean",
        ],
        "description": (
            "Directional mean packet-size measurements."
        ),
        "evidence": (
            "UNSW-NB15 metadata identifies mean packet sizes "
            "transmitted by source and destination."
        ),
    },

    "state_ttl_condition_count": {
        "nsl_kdd": [],
        "unsw_nb15": [
            "ct_state_ttl",
        ],
        "description": (
            "Connection-state counts conditioned on TTL ranges."
        ),
        "evidence": (
            "UNSW-NB15 metadata describes a count associated with "
            "each state according to source and destination TTL ranges."
        ),
    },
}


# Metadata aliases.
# Original dataset feature names are never changed.

UNSW_ALIASES = {
    "smean": "smeansz",
    "dmean": "dmeansz",
    "response_body_len": "res_bdy_len",
    "sinpkt": "Sintpkt",
    "dinpkt": "Dintpkt",
    "sjit": "Sjit",
    "djit": "Djit",
    "spkts": "Spkts",
    "dpkts": "Dpkts",
    "sload": "Sload",
    "dload": "Dload",
    "ct_src_ltm": "ct_src_ ltm",
}


def read_csv_with_encoding(path, **kwargs):
    """Try UTF-8 first, then fall back to Windows encoding."""

    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            **kwargs,
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            path,
            encoding="cp1252",
            **kwargs,
        )


def load_metadata():
    """Read original NSL-KDD and UNSW-NB15 feature metadata."""

    nsl = pd.read_csv(
        NSL_METADATA_PATH,
        header=None,
        names=["feature", "type"],
    )

    unsw = read_csv_with_encoding(
        UNSW_METADATA_PATH,
    )

    nsl_records = {}

    for _, row in nsl.iterrows():
        name = str(row["feature"]).strip()

        nsl_records[name] = {
    "metadata_name": name,
    "data_type": NSL_TYPE_OVERRIDES.get(
        name,
        str(row["type"]).strip(),
    ),
    "description": None,
    "unit": None,
    "description_source": "not_provided_by_metadata",
}

    unsw_records = {}

    for _, row in unsw.iterrows():
        name = str(row["Name"]).strip()

        description = str(row["Description"]).strip()
        data_type = str(row["Type "]).strip()

        if description.lower() == "nan":
            description = None

        if data_type.lower() == "nan":
            data_type = "unknown"

        unsw_records[
            name.lower().replace(" ", "")
        ] = {
            "metadata_name": name,
            "data_type": data_type,
            "description": description,
            "unit": None,
            "description_source": (
                "UNSW_NB15_feature_metadata"
                if description
                else "not_provided_by_metadata"
            ),
        }

    return nsl_records, unsw_records


def find_level2_group(dataset_key, feature):
    """Find a candidate semantic group for a feature."""

    matches = []

    for group_id, group in LEVEL2_GROUPS.items():
        if feature in group[dataset_key]:
            matches.append((group_id, group))

    if len(matches) > 1:
        raise ValueError(
            f"Feature {dataset_key}:{feature} appears in "
            f"multiple Level 2 groups: "
            f"{[item[0] for item in matches]}"
        )

    if not matches:
        return None, None

    return matches[0]


def find_level1_mapping(dataset, feature, registry):
    """Find an existing approved equivalent mapping."""

    dataset_key = (
        "nsl_kdd" if dataset == "NSL-KDD" else "unsw_nb15"
    )

    for item in registry.get("common_features", []):
        if item.get(dataset_key) == feature:
            return {
                "semantic_id": item["semantic_id"],
                "relationship": "equivalent",
                "confidence": item["confidence"],
                "evidence": (
                    "Existing approved semantic feature registry mapping."
                ),
                "review_status": "approved",
            }

    return None


def infer_description(feature):
    """
    Provide a cautious name-derived description.

    Do not invent exact units or measurement formulas.
    """

    return (
        f"Feature named '{feature}'. "
        "A detailed semantic description requires documentation review."
    )


def infer_unit(feature, description):
    """Extract only units explicitly stated in metadata."""

    if not description:
        return None

    text = description.lower()

    if "msec" in text or "millisecond" in text:
        return "milliseconds"

    if "bits per second" in text:
        return "bits_per_second"

    return None


def build_descriptor(
    dataset,
    feature,
    classification,
    nsl_metadata,
    unsw_metadata,
    registry,
):
    """Build one feature descriptor."""

    dataset_key = (
        "nsl_kdd" if dataset == "NSL-KDD" else "unsw_nb15"
    )

    if dataset_key == "nsl_kdd":
        metadata = nsl_metadata.get(feature)

    else:
        metadata_name = UNSW_ALIASES.get(feature, feature)

        lookup_key = metadata_name.lower().replace(" ", "")

        metadata = unsw_metadata.get(lookup_key)

    metadata = metadata or {}

    description = metadata.get("description")

    if description:
        description_source = metadata["description_source"]
        description_confidence = "high"

    else:
        description = infer_description(feature)
        description_source = "feature_name_inference"
        description_confidence = "low"

    level1 = find_level1_mapping(
        dataset,
        feature,
        registry,
    )

    group_id, group = find_level2_group(
        dataset_key,
        feature,
    )

    if level1:
        level = 1
        relationship = "equivalent"
        confidence = level1["confidence"]
        review_status = "approved"
        evidence = level1["evidence"]
        semantic_id = level1["semantic_id"]
        semantic_group = None

    elif group_id:
        level = 2
        relationship = "group_only"
        confidence = "medium"
        review_status = "candidate_review"
        evidence = group["evidence"]
        semantic_id = None
        semantic_group = group_id

    else:
        level = 3
        relationship = "dataset_specific"
        confidence = "unassessed"
        review_status = "dataset_specific"
        evidence = (
            "No approved equivalent mapping or reviewed semantic "
            "group assignment exists."
        )
        semantic_id = None
        semantic_group = None

    unit = infer_unit(
        feature,
        metadata.get("description"),
    )

    return {
        "dataset": dataset_key,
        "original_feature_name": feature,
        "metadata_feature_name": metadata.get("metadata_name"),
        "level": level,
        "classification": (
            "common" if level == 1
            else "semantic_group_candidate" if level == 2
            else "dataset_specific"
        ),
        "semantic_id": semantic_id,
        "semantic_group": semantic_group,
        "relationship": relationship,
        "confidence": confidence,
        "review_status": review_status,
        "data_type": metadata.get("data_type", "unknown"),
        "description": description,
        "description_source": description_source,
        "description_confidence": description_confidence,
        "unit": unit,
        "unit_status": (
            "documented"
            if unit
            else "unknown_or_not_provided"
        ),
        "evidence": evidence,
    }
# Data types verified using actual NSL-KDD training values.
# These features contain only 0 and 1.
NSL_TYPE_OVERRIDES = {
    "land": "binary",
    "logged_in": "binary",
    "is_host_login": "binary",
    "is_guest_login": "binary",
}


def validate_descriptors(descriptors, inventory, registry):
    """Validate coverage, uniqueness, and mapping status."""

    expected = {
        (
            "nsl_kdd" if row["dataset"] == "NSL-KDD" else "unsw_nb15",
            row["feature"],
        )
        for _, row in inventory.iterrows()
    }

    actual = {
        (
            item["dataset"],
            item["original_feature_name"],
        )
        for item in descriptors
    }

    if len(descriptors) != len(actual):
        raise AssertionError(
            "Duplicate dataset-feature descriptor entries detected."
        )

    if actual != expected:
        missing = expected - actual
        extra = actual - expected

        raise AssertionError(
            f"Feature coverage mismatch. "
            f"Missing: {missing}; Extra: {extra}"
        )

    level_counts = {
        level: sum(item["level"] == level for item in descriptors)
        for level in (1, 2, 3)
    }

    if level_counts[1] != 10:
        raise AssertionError(
            f"Expected 10 Level 1 entries, got {level_counts[1]}"
        )

    if level_counts[2] != 65:
        raise AssertionError(
            f"Expected 65 Level 2 candidates, got {level_counts[2]}"
        )

    if level_counts[3] != 8:
        raise AssertionError(
            f"Expected 8 Level 3 entries, got {level_counts[3]}"
        )

    approved_ids = {
        item["semantic_id"]
        for item in registry.get("common_features", [])
    }

    for item in descriptors:

        if item["level"] == 1:
            assert item["relationship"] == "equivalent"
            assert item["review_status"] == "approved"
            assert item["semantic_id"] in approved_ids

        elif item["level"] == 2:
            assert item["relationship"] == "group_only"
            assert item["review_status"] == "candidate_review"
            assert item["semantic_id"] is None
            assert item["semantic_group"] is not None

        elif item["level"] == 3:
            assert item["relationship"] == "dataset_specific"
            assert item["review_status"] == "dataset_specific"
            assert item["semantic_group"] is None

    return level_counts


def main():

    inventory = pd.read_csv(INVENTORY_PATH)

    with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
        registry = yaml.safe_load(file)

    nsl_metadata, unsw_metadata = load_metadata()

    descriptors = []

    for _, row in inventory.iterrows():

        descriptor = build_descriptor(
            dataset=row["dataset"],
            feature=row["feature"],
            classification=row["classification"],
            nsl_metadata=nsl_metadata,
            unsw_metadata=unsw_metadata,
            registry=registry,
        )

        descriptors.append(descriptor)

    level_counts = validate_descriptors(
        descriptors,
        inventory,
        registry,
    )

    output = {
        "schema_version": 1,
        "description": (
            "Dataset feature descriptors for NOVEL-IDS. "
            "Level 2 assignments are semantic-group candidates, "
            "not approved equivalent mappings."
        ),
        "level_definitions": {
            1: "Approved equivalent semantic identity",
            2: "Related semantic group; not equivalent",
            3: "Dataset-specific feature",
        },
        "feature_count": len(descriptors),
        "features": descriptors,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            output,
            file,
            sort_keys=False,
            allow_unicode=True,
            width=110,
        )

    print("=" * 65)
    print("NOVEL-IDS FEATURE DESCRIPTOR GENERATION")
    print("=" * 65)

    print("Total feature entries:", len(descriptors))
    print("Level 1 entries:", level_counts[1])
    print("Level 2 candidates:", level_counts[2])
    print("Level 3 entries:", level_counts[3])

    print(
        "Descriptors with metadata descriptions:",
        sum(
            item["description_source"]
            == "UNSW_NB15_feature_metadata"
            for item in descriptors
        ),
    )

    print(
        "Descriptors requiring description review:",
        sum(
            item["description_confidence"] == "low"
            for item in descriptors
        ),
    )

    print("\nSemantic groups:")

    group_counts = {}

    for item in descriptors:
        if item["level"] == 2:
            group = item["semantic_group"]
            group_counts[group] = group_counts.get(group, 0) + 1

    for group, count in group_counts.items():
        print(f"  {group}: {count} feature entries")

    print("\nDescriptor file saved to:")
    print(OUTPUT_PATH)

    print("\nDESCRIPTOR VALIDATION PASSED.")


if __name__ == "__main__":
    main()
