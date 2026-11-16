
import numpy as np
import pandas as pd
import torch


def encode_shared(value, feature, dataset, vocab):
    value = str(value).strip().lower()

    if value in ("nan", "none", ""):
        value = "unknown"

    shared_token = "SHARED::" + value
    specific_token = dataset + "::" + value

    if shared_token in vocab[feature]:
        return vocab[feature][shared_token]

    return vocab[feature].get(specific_token, 0)


def preprocess_uploaded_data(df, dataset, preprocessing):
    df = df.copy()

    if dataset == "UNSW":
        required = (
            ["dur", "proto", "service", "state", "sbytes", "dbytes"]
            + preprocessing["unsw_specific_numeric_columns"]
        )

        numeric_columns = preprocessing["unsw_specific_numeric_columns"]
        log_features = preprocessing["unsw_log_features"]
        scaler = preprocessing["unsw_scaler"]
        state_vocab = preprocessing["unsw_state_vocab"]

        shared_columns = ["dur", "sbytes", "dbytes"]

    elif dataset == "NSL":
        required = (
            ["duration", "protocol_type", "service", "flag",
             "src_bytes", "dst_bytes"]
            + preprocessing["nsl_specific_numeric_columns"]
        )

        numeric_columns = preprocessing["nsl_specific_numeric_columns"]
        log_features = preprocessing["nsl_log_features"]
        scaler = preprocessing["nsl_scaler"]
        state_vocab = preprocessing["nsl_flag_vocab"]

        shared_columns = ["duration", "src_bytes", "dst_bytes"]

    else:
        raise ValueError("Dataset must be UNSW or NSL.")

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    # Numeric conversion
    for col in shared_columns + numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_all = df[shared_columns + numeric_columns]

    if numeric_all.isna().any().any():
        raise ValueError(
            "Numeric features contain missing or non-numeric values. "
            "Please correct the uploaded file."
        )

    # Shared numeric preprocessing
    shared_raw = df[shared_columns].to_numpy(dtype=np.float64)

    shared_raw = np.log1p(
        np.maximum(shared_raw, 0)
    )

    shared_scaled = preprocessing["shared_scaler"].transform(
        shared_raw
    )

    # Dataset-specific numeric preprocessing
    specific_raw = df[numeric_columns].copy()

    for col in log_features:
        specific_raw[col] = np.log1p(
            np.maximum(specific_raw[col].to_numpy(), 0)
        )

    specific_scaled = scaler.transform(
        specific_raw[numeric_columns]
    )

    # Shared categorical encoding
    if dataset == "UNSW":
        protocol_col = "proto"
        category_col = "state"
    else:
        protocol_col = "protocol_type"
        category_col = "flag"

    vocab = preprocessing["categorical_vocab"]

    shared_categorical = np.column_stack([
        df[protocol_col].apply(
            lambda x: encode_shared(x, "protocol", dataset, vocab)
        ).to_numpy(),

        df["service"].apply(
            lambda x: encode_shared(x, "service", dataset, vocab)
        ).to_numpy()
    ]).astype(np.int64)

    # Dataset-specific categorical encoding
    category_values = (
        df[category_col]
        .fillna("UNK")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    specific_categorical = (
        category_values
        .map(state_vocab)
        .fillna(0)
        .to_numpy(dtype=np.int64)
        .reshape(-1, 1)
    )

    return (
        shared_scaled.astype(np.float32),
        shared_categorical,
        specific_scaled.astype(np.float32),
        specific_categorical
    )


def predict_uploaded_data(
    df,
    dataset,
    binary_model,
    unsw_category_model,
    nsl_category_model,
    preprocessing,
    device
):
    shared_num, shared_cat, specific_num, specific_cat = (
        preprocess_uploaded_data(df, dataset, preprocessing)
    )

    tensors = [
        torch.tensor(shared_num, dtype=torch.float32),
        torch.tensor(shared_cat, dtype=torch.long),
        torch.tensor(specific_num, dtype=torch.float32),
        torch.tensor(specific_cat, dtype=torch.long)
    ]

    category_model = (
        unsw_category_model if dataset == "UNSW"
        else nsl_category_model
    )

    binary_model.eval()
    category_model.eval()

    binary_predictions = []
    category_predictions = []
    attack_probabilities = []

    with torch.no_grad():
        for start in range(0, len(df), 512):
            end = min(start + 512, len(df))

            batch = [x[start:end].to(device) for x in tensors]

            logits, _, _ = binary_model(
                *batch,
                dataset=dataset
            )

            probabilities = torch.softmax(logits, dim=1)
            binary_pred = probabilities.argmax(dim=1)

            binary_predictions.extend(
                binary_pred.cpu().numpy()
            )

            attack_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

            category_pred = category_model(
                binary_model.final_classifier.layer_norm(
                    binary_model.final_classifier.transformer(
                        torch.cat([
                            binary_model.shared_encoder(batch[0], batch[1]),
                            (
                                binary_model.unsw_fusion(
                                    binary_model.shared_encoder(batch[0], batch[1]),
                                    binary_model.unsw_adapter(batch[2], batch[3])
                                )[0]
                                if dataset == "UNSW"
                                else binary_model.nsl_fusion(
                                    binary_model.shared_encoder(batch[0], batch[1]),
                                    binary_model.nsl_adapter(batch[2], batch[3])
                                )[0]
                            )
                        ], dim=1)
                    )
                ).mean(dim=1)
            )

            category_predictions.extend(
                category_pred.argmax(dim=1).cpu().numpy()
            )

    encoder = (
        preprocessing["unsw_category_encoder"]
        if dataset == "UNSW"
        else preprocessing["nsl_category_encoder"]
    )

    binary_predictions = np.array(binary_predictions)
    category_predictions = np.array(category_predictions)

    category_names = encoder.inverse_transform(category_predictions)

    results = pd.DataFrame({
        "Binary_Prediction": np.where(
            binary_predictions == 0, "Normal", "Attack"
        ),
        "Predicted_Category": np.where(
            binary_predictions == 0, "Normal", category_names
        ),
        "Attack_Probability": attack_probabilities
    })

    return results
