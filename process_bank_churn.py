# process_bank_churn.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

TARGET_COL: str = "Exited"

# Рекомендовано прибрати Surname; також часто є ідентифікатори
DEFAULT_DROP_COLS: List[str] = ["Surname", "CustomerId", "id"]

# Типові фічі (без Surname/CustomerId/id)
DEFAULT_INPUT_COLS: List[str] = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


def validate_columns_exist(df: pd.DataFrame, cols: List[str]) -> None:
    """
    Validate that all required columns exist in the dataframe.

    Args:
        df: Input dataframe.
        cols: Required column names.

    Raises:
        ValueError: if any required columns are missing.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def drop_unwanted_columns(df: pd.DataFrame, drop_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Drop columns if present.

    Args:
        df: Input dataframe.
        drop_cols: Columns to drop.

    Returns:
        Dataframe without dropped columns.
    """
    drop_cols = drop_cols or []
    return df.drop(columns=drop_cols, errors="ignore")


def split_train_val_test(
    df: pd.DataFrame,
    target_col: str,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/val/test.

    Args:
        df: Dataframe containing features + target.
        target_col: Target column.
        train_size: Fraction for train.
        val_size: Fraction for validation.
        test_size: Fraction for test.
        random_state: Random seed.
        stratify: Stratify by target.

    Returns:
        train_df, val_df, test_df
    """
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"train_size+val_size+test_size must be 1.0, got {total}")

    stratify_y = df[target_col] if stratify else None

    # split train vs temp
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        random_state=random_state,
        stratify=stratify_y,
    )

    # split temp into val/test
    stratify_temp = temp_df[target_col] if stratify else None
    # proportion of test within temp = test_size / (val_size + test_size)
    test_ratio_in_temp = test_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio_in_temp,
        random_state=random_state,
        stratify=stratify_temp,
    )

    return train_df, val_df, test_df


def get_numeric_and_categorical_cols(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Detect numeric and categorical columns.

    Args:
        X: Features dataframe.

    Returns:
        numeric_cols, categorical_cols
    """
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def fit_scaler(X_train: pd.DataFrame, numeric_cols: List[str]) -> MinMaxScaler:
    """
    Fit MinMaxScaler on numeric columns of training data.

    Args:
        X_train: Training features.
        numeric_cols: Numeric column names.

    Returns:
        Fitted MinMaxScaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(X_train[numeric_cols])
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: MinMaxScaler, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Transform numeric columns using fitted scaler.

    Args:
        X: Features dataframe.
        scaler: Fitted scaler.
        numeric_cols: Numeric column names to scale.

    Returns:
        Scaled dataframe.
    """
    out = X.copy()
    if numeric_cols:
        out[numeric_cols] = scaler.transform(out[numeric_cols])
    return out


def fit_encoder(X_train: pd.DataFrame, categorical_cols: List[str]) -> OneHotEncoder:
    """
    Fit OneHotEncoder on categorical columns of training data.

    Args:
        X_train: Training features.
        categorical_cols: Categorical column names.

    Returns:
        Fitted OneHotEncoder.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train[categorical_cols])
    return encoder


def apply_encoder(
    X: pd.DataFrame, encoder: OneHotEncoder, categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Transform categorical columns using fitted encoder and append one-hot columns.

    Args:
        X: Features dataframe.
        encoder: Fitted OneHotEncoder.
        categorical_cols: Columns to encode.

    Returns:
        Encoded dataframe (categorical cols replaced with one-hot features).
    """
    out = X.copy()
    if not categorical_cols:
        return out

    encoded_arr = encoder.transform(out[categorical_cols])
    encoded_names = list(encoder.get_feature_names_out(categorical_cols))
    encoded_df = pd.DataFrame(encoded_arr, columns=encoded_names, index=out.index)

    out = pd.concat([out.drop(columns=categorical_cols), encoded_df], axis=1)
    return out


def reorder_columns(X: pd.DataFrame, feature_order: List[str]) -> pd.DataFrame:
    """
    Ensure columns exactly match feature_order and reorder accordingly.

    Args:
        X: Features dataframe.
        feature_order: Desired feature order.

    Returns:
        Reordered dataframe with only feature_order columns.

    Raises:
        ValueError: if expected columns are missing.
    """
    missing = [c for c in feature_order if c not in X.columns]
    if missing:
        raise ValueError(f"Missing expected columns after preprocessing: {missing}")
    return X[feature_order].copy()


def preprocess_data(
    raw_df: pd.DataFrame,
    input_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COL,
    drop_cols: Optional[List[str]] = None,
    scaler_numeric: bool = True,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Dict[str, Any]:
    """
    Preprocess raw labeled data with train/val/test split.
    Fits encoder/scaler ONLY on train, transforms train+val, and returns RAW test to be
    transformed later via preprocess_new_data().

    Returns dict keys:
      - train_X, train_y
      - val_X, val_y
      - test_raw_X, test_y
      - input_cols (final feature names after encoding)
      - scaler, encoder
      - meta (raw_input_cols, numeric_cols, categorical_cols) for convenience
    """
    if input_cols is None:
        input_cols = DEFAULT_INPUT_COLS
    if drop_cols is None:
        drop_cols = DEFAULT_DROP_COLS

    df = drop_unwanted_columns(raw_df, drop_cols=drop_cols)

    # ensure required columns exist after dropping
    validate_columns_exist(df, input_cols + [target_col])

    df = df[input_cols + [target_col]].copy()

    train_df, val_df, test_df = split_train_val_test(
        df,
        target_col=target_col,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    X_train_raw = train_df[input_cols].copy()
    y_train = train_df[target_col].copy()

    X_val_raw = val_df[input_cols].copy()
    y_val = val_df[target_col].copy()

    X_test_raw = test_df[input_cols].copy()
    y_test = test_df[target_col].copy()

    numeric_cols, categorical_cols = get_numeric_and_categorical_cols(X_train_raw)

    # Fit on train only
    scaler: Optional[MinMaxScaler] = None
    if scaler_numeric and numeric_cols:
        scaler = fit_scaler(X_train_raw, numeric_cols)

    encoder: Optional[OneHotEncoder] = None
    if categorical_cols:
        encoder = fit_encoder(X_train_raw, categorical_cols)

    # Transform train + val (test stays raw here!)
    X_train = X_train_raw.copy()
    X_val = X_val_raw.copy()

    if scaler_numeric and scaler is not None and numeric_cols:
        X_train = apply_scaler(X_train, scaler, numeric_cols)
        X_val = apply_scaler(X_val, scaler, numeric_cols)

    if encoder is not None and categorical_cols:
        X_train = apply_encoder(X_train, encoder, categorical_cols)
        X_val = apply_encoder(X_val, encoder, categorical_cols)

    # Stable feature order based on train
    input_cols_final = X_train.columns.tolist()
    X_val = reorder_columns(X_val, input_cols_final)

    return {
        "train_X": X_train,
        "train_y": y_train,
        "val_X": X_val,
        "val_y": y_val,
        "test_raw_X": X_test_raw,
        "test_y": y_test,
        "input_cols": input_cols_final,
        "scaler": scaler,
        "encoder": encoder,
        "meta": {
            "raw_input_cols": input_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "drop_cols": drop_cols,
            "target_col": target_col,
        },
    }


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[MinMaxScaler],
    encoder: Optional[OneHotEncoder],
    scaler_numeric: bool = True,
    raw_input_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Preprocess new/unseen data (or held-out raw test) using already fitted scaler/encoder.
    Does NOT fit anything.

    Args:
        new_df: Raw dataframe WITHOUT target column (or with, but you should pass only features).
        input_cols: Final feature order from training preprocessing (data["input_cols"]).
        scaler: Fitted scaler from training (or None).
        encoder: Fitted encoder from training (or None).
        scaler_numeric: Must match training-time setting.
        raw_input_cols: Raw feature columns before encoding/scaling. If None -> DEFAULT_INPUT_COLS.
        drop_cols: Columns to drop if present (e.g., Surname/CustomerId/id).

    Returns:
        Processed dataframe with columns exactly == input_cols.
    """
    if raw_input_cols is None:
        raw_input_cols = DEFAULT_INPUT_COLS
    if drop_cols is None:
        drop_cols = DEFAULT_DROP_COLS

    X = drop_unwanted_columns(new_df, drop_cols=drop_cols)

    validate_columns_exist(X, raw_input_cols)
    X = X[raw_input_cols].copy()

    numeric_cols, categorical_cols = get_numeric_and_categorical_cols(X)

    if scaler_numeric and scaler is not None and numeric_cols:
        X = apply_scaler(X, scaler, numeric_cols)

    if encoder is not None and categorical_cols:
        X = apply_encoder(X, encoder, categorical_cols)

    return reorder_columns(X, input_cols)