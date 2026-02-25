# process_bank_churn.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


TARGET_COL: str = "Exited"

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


def select_columns(raw_df: pd.DataFrame, input_cols: List[str], target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Select only columns needed for modeling: input features + target.

    Args:
        raw_df: Raw dataframe with all columns.
        input_cols: List of feature columns to keep.
        target_col: Target column name.

    Returns:
        DataFrame containing only input_cols + target_col.
    """
    missing = [c for c in input_cols + [target_col] if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw_df: {missing}")

    return raw_df[input_cols + [target_col]].copy()


def split_train_val(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and validation parts.

    Args:
        df: DataFrame containing features + target.
        target_col: Target column name.
        test_size: Validation size fraction.
        random_state: Random seed.
        stratify: Whether to stratify by target_col.

    Returns:
        (train_df, val_df)
    """
    stratify_y = df[target_col] if stratify else None
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )
    return train_df, val_df


def get_numeric_and_categorical_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Detect numeric and categorical columns in a dataframe.

    Args:
        df: Input features dataframe (without target).

    Returns:
        (numeric_cols, categorical_cols)
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def fit_scaler(train_inputs: pd.DataFrame, numeric_cols: List[str]) -> MinMaxScaler:
    """
    Fit MinMaxScaler on train numeric columns.

    Args:
        train_inputs: Train features dataframe.
        numeric_cols: Numeric feature column names.

    Returns:
        Fitted MinMaxScaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Transform numeric columns using a fitted scaler.

    Args:
        df: Features dataframe.
        scaler: Fitted MinMaxScaler.
        numeric_cols: Numeric columns to scale.

    Returns:
        DataFrame with scaled numeric columns.
    """
    out = df.copy()
    out[numeric_cols] = scaler.transform(out[numeric_cols])
    return out


def fit_encoder(train_inputs: pd.DataFrame, categorical_cols: List[str]) -> OneHotEncoder:
    """
    Fit OneHotEncoder on train categorical columns.

    Args:
        train_inputs: Train features dataframe.
        categorical_cols: Categorical columns to encode.

    Returns:
        Fitted OneHotEncoder.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(train_inputs[categorical_cols])
    return encoder


def apply_encoder(df: pd.DataFrame, encoder: OneHotEncoder, categorical_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode categorical columns using a fitted encoder and drop original categorical columns.

    Args:
        df: Features dataframe.
        encoder: Fitted OneHotEncoder.
        categorical_cols: Categorical columns to encode.

    Returns:
        (encoded_df, encoded_feature_names)
    """
    out = df.copy()
    if not categorical_cols:
        return out, []

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    encoded_arr = encoder.transform(out[categorical_cols])

    encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cols, index=out.index)
    out = pd.concat([out.drop(columns=categorical_cols), encoded_df], axis=1)
    return out, encoded_cols


def reorder_columns(df: pd.DataFrame, final_feature_order: List[str]) -> pd.DataFrame:
    """
    Reorder columns to a fixed list (important for consistent training/prediction).

    Args:
        df: Features dataframe.
        final_feature_order: Desired column order.

    Returns:
        Reordered dataframe.
    """
    missing = [c for c in final_feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"After preprocessing, some expected columns are missing: {missing}")
    return df[final_feature_order].copy()


def preprocess_data(
    raw_df: pd.DataFrame,
    input_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COL,
    scaler_numeric: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Optional[MinMaxScaler], OneHotEncoder]:
    """
    Full preprocessing for training:
      - select columns
      - split to train/val
      - one-hot encode categorical
      - (optional) scale numeric
      - return artifacts to reuse later

    Args:
        raw_df: Raw dataframe (train.csv loaded).
        input_cols: Feature columns. If None, uses DEFAULT_INPUT_COLS.
        target_col: Target column name.
        scaler_numeric: If True, scale numeric columns with MinMaxScaler.
        test_size: Validation fraction.
        random_state: Random seed.
        stratify: Whether to stratify split by target.

    Returns:
        X_train: processed train features
        train_targets: train target series
        X_val: processed val features
        val_targets: val target series
        input_cols_final: final feature names used in X (after encoding)
        scaler: fitted MinMaxScaler or None
        encoder: fitted OneHotEncoder
    """
    if input_cols is None:
        input_cols = DEFAULT_INPUT_COLS

    df = select_columns(raw_df, input_cols=input_cols, target_col=target_col)
    train_df, val_df = split_train_val(
        df, target_col=target_col, test_size=test_size, random_state=random_state, stratify=stratify
    )

    train_inputs = train_df[input_cols].copy()
    val_inputs = val_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    val_targets = val_df[target_col].copy()

    numeric_cols, categorical_cols = get_numeric_and_categorical_cols(train_inputs)

    # (Optional) scale numeric
    scaler: Optional[MinMaxScaler] = None
    if scaler_numeric and numeric_cols:
        scaler = fit_scaler(train_inputs, numeric_cols)
        train_inputs = apply_scaler(train_inputs, scaler, numeric_cols)
        val_inputs = apply_scaler(val_inputs, scaler, numeric_cols)

    # Encode categorical
    encoder = fit_encoder(train_inputs, categorical_cols) if categorical_cols else OneHotEncoder(
        sparse_output=False, handle_unknown="ignore"
    )
    if categorical_cols:
        train_inputs, encoded_cols = apply_encoder(train_inputs, encoder, categorical_cols)
        val_inputs, _ = apply_encoder(val_inputs, encoder, categorical_cols)
    else:
        encoded_cols = []

    # Final feature order (stable)
    # numeric + all non-categorical leftovers (already there) + encoded
    input_cols_final = train_inputs.columns.tolist()

    X_train = train_inputs.copy()
    X_val = reorder_columns(val_inputs, input_cols_final)

    return X_train, train_targets, X_val, val_targets, input_cols_final, scaler, encoder


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols_final: List[str],
    scaler: Optional[MinMaxScaler],
    encoder: OneHotEncoder,
    scaler_numeric: bool = True,
    raw_input_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Preprocess new/unseen data (e.g., test.csv) using already fitted scaler & encoder.

    Important:
      - You must pass input_cols_final from preprocess_data(...) to keep the same feature set/order.

    Args:
        new_df: New raw data (without target).
        input_cols_final: Final feature names from training preprocessing (after encoding).
        scaler: Fitted MinMaxScaler from training (or None if scaling disabled).
        encoder: Fitted OneHotEncoder from training.
        scaler_numeric: Must match training-time setting.
        raw_input_cols: Original raw feature columns before encoding.
                       If None, will try to infer from encoder/scaler where possible,
                       otherwise falls back to DEFAULT_INPUT_COLS.

    Returns:
        Processed features dataframe with columns exactly == input_cols_final.
    """
    if raw_input_cols is None:
        # Best-effort inference:
        # - encoder.feature_names_in_ gives categorical columns used for fit (sklearn >= 1.0)
        # - scaler.feature_names_in_ gives numeric columns used for fit (if scaler exists and sklearn >= 1.0)
        inferred_cats = list(getattr(encoder, "feature_names_in_", []))
        inferred_nums = list(getattr(scaler, "feature_names_in_", [])) if scaler is not None else []
        if inferred_cats or inferred_nums:
            raw_input_cols = inferred_nums + inferred_cats
        else:
            raw_input_cols = DEFAULT_INPUT_COLS

    missing = [c for c in raw_input_cols if c not in new_df.columns]
    if missing:
        raise ValueError(f"New data is missing required raw feature columns: {missing}")

    inputs = new_df[raw_input_cols].copy()
    numeric_cols, categorical_cols = get_numeric_and_categorical_cols(inputs)

    if scaler_numeric and scaler is not None and numeric_cols:
        inputs = apply_scaler(inputs, scaler, numeric_cols)

    if categorical_cols:
        inputs, _ = apply_encoder(inputs, encoder, categorical_cols)

    # Ensure same columns and order as training
    # If some OHE columns didn't appear in new data, they still exist because encoder outputs them.
    # If there are extra columns (shouldn't happen), reorder_columns will drop them.
    return reorder_columns(inputs, input_cols_final)