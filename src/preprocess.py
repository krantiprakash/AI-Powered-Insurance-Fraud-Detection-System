"""
preprocess.py
-------------
All data cleaning and feature engineering logic.
Used by both train.py (fit + transform) and predict.py (transform only).
"""

import pandas as pd
import numpy as np


# ── Column definitions ────────────────────────────────────────────────────────

NUMERIC_FEATURES = ["Age", "DriverRating", "WeekOfMonth", "WeekOfMonthClaimed", "Year"]

ORDINAL_COLS = [
    "Deductible", "VehiclePrice", "Days_Policy_Accident", "Days_Policy_Claim",
    "PastNumberOfClaims", "AgeOfVehicle", "AgeOfPolicyHolder",
    "NumberOfSuppliments", "AddressChange_Claim", "NumberOfCars"
]

NOMINAL_COLS = [
    "Month", "DayOfWeek", "Make", "AccidentArea", "DayOfWeekClaimed",
    "MonthClaimed", "Sex", "MaritalStatus", "Fault", "PolicyType",
    "VehicleCategory", "PoliceReportFiled", "WitnessPresent",
    "AgentType", "BasePolicy"
]

DROP_COLS = ["PolicyNumber", "RepNumber"]

TARGET = "FraudFound_P"

# Ordinal category orders (low → high)
ORDINAL_ORDERS = {
    "Deductible":           [300, 400, 500, 700],
    "VehiclePrice":         ["less than 20000", "20000 to 29000", "30000 to 39000",
                             "40000 to 59000", "60000 to 69000", "more than 69000"],
    "Days_Policy_Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days_Policy_Claim":    ["none", "8 to 15", "15 to 30", "more than 30"],
    "PastNumberOfClaims":   ["none", "1", "2 to 4", "more than 4"],
    "AgeOfVehicle":         ["new", "2 years", "3 years", "4 years", "5 years",
                             "6 years", "7 years", "more than 7"],
    "AgeOfPolicyHolder":    ["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35",
                             "36 to 40", "41 to 50", "51 to 65", "over 65"],
    "NumberOfSuppliments":  ["none", "1 to 2", "3 to 5", "more than 5"],
    "AddressChange_Claim":  ["no change", "under 6 months", "1 year",
                             "2 to 3 years", "4 to 8 years"],
    "NumberOfCars":         ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"]
}


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to a raw dataframe.
    Safe to call on both training data and single inference rows.
    """
    df = df.copy()

    # Drop ID columns if present
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Fix Age = 0 placeholder → median of valid ages
    if (df["Age"] == 0).any():
        median_age = df[df["Age"] > 0]["Age"].median()
        df.loc[df["Age"] == 0, "Age"] = median_age

    # Fix '0' placeholder in date-derived categorical columns
    df["DayOfWeekClaimed"] = df["DayOfWeekClaimed"].replace("0", "Unknown")
    df["MonthClaimed"]     = df["MonthClaimed"].replace("0", "Unknown")

    return df


# ── Ordinal encoding ─────────────────────────────────────────────────────────

def apply_ordinal_encoding(df: pd.DataFrame, encoder) -> pd.DataFrame:
    """Apply a fitted OrdinalEncoder to the ordinal columns."""
    df = df.copy()
    df[ORDINAL_COLS] = encoder.transform(df[ORDINAL_COLS])
    return df


# ── One-Hot encoding ──────────────────────────────────────────────────────────

def apply_ohe(df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    """
    Apply OHE to nominal columns and align with training feature columns.
    Missing columns are filled with 0, extra columns are dropped.

    Note: drop_first=False here intentionally — for single-row inputs,
    drop_first=True would drop the only value present, zeroing all OHE
    columns. The expected_columns list (generated with drop_first=True
    during training) already excludes the reference categories, so
    alignment achieves the same result safely for any batch size.
    """
    df = df.copy()
    df_ohe = pd.get_dummies(df[NOMINAL_COLS], dtype=int)

    # Align columns to match training set exactly (adds missing as 0, drops extras)
    for col in expected_columns:
        if col not in df_ohe.columns:
            df_ohe[col] = 0
    df_ohe = df_ohe[expected_columns]

    return df_ohe


# ── Build feature matrix ──────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, encoder, scaler,
                         ohe_columns: list) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    clean → ordinal encode → OHE → scale numerics → return feature matrix.
    Does NOT add anomaly_score (that is added in predict.py / train.py).
    """
    df = clean(df)
    df = apply_ordinal_encoding(df, encoder)
    df_ohe = apply_ohe(df, ohe_columns)

    numeric_df   = df[NUMERIC_FEATURES].copy().reset_index(drop=True)
    ordinal_df   = df[ORDINAL_COLS].copy().reset_index(drop=True)
    ohe_df       = df_ohe.reset_index(drop=True)

    # Combine numeric + ordinal + OHE
    X = pd.concat([numeric_df, ordinal_df, ohe_df], axis=1)

    # Scale numeric columns
    X[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])

    return X


# ── OHE column list (used during training to record expected columns) ─────────

def get_ohe_columns(df: pd.DataFrame) -> list:
    """Fit OHE on a dataframe and return the resulting column names."""
    return pd.get_dummies(df[NOMINAL_COLS], drop_first=True, dtype=int).columns.tolist()
