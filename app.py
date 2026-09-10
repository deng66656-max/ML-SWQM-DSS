# -*- coding: utf-8 -*-
# ============================================================
# ML-SWQM-DSS
# Machine Learning-based Surface Water Quality Monitoring
# and Decision Support System
#
# Upper Athi River Catchment - Petroleum-Focused Case Study
# ============================================================

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
st.logo("static/icons/icon-512.png", size="large")
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ML-SWQM-DSS | Upper Athi River",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
)

DATA_FILE = os.path.join(
    DATA_DIR,
    "upper_athi_water_quality.csv",
)

PREDICTIONS_FILE = os.path.join(
    DATA_DIR,
    "athi_model_predictions.csv",
)

FEATURE_IMPORTANCE_FILE = os.path.join(
    DATA_DIR,
    "athi_feature_importance.csv",
)

REGRESSOR_FILE = os.path.join(
    MODEL_DIR,
    "athi_regressor.pkl",
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "athi_model_features.pkl",
)

METADATA_FILE = os.path.join(
    MODEL_DIR,
    "athi_model_metadata.pkl",
)


# ============================================================
# PROJECT CONSTANTS
# ============================================================

APP_NAME = "ML-SWQM-DSS"

APP_DESCRIPTION = (
    "Machine Learning-based Surface Water Quality Monitoring "
    "and Decision Support System"
)

CASE_STUDY = "Upper Athi River Catchment, Kenya"

# Features used by the Athi Random Forest model.
EXPECTED_MODEL_FEATURES = [
    "turbidity",
    "dissolved_oxygen",
    "total_phosphorus",
    "tss",
    "tds",
    "temperature",
    "conductivity",
    "total_nitrogen",
    "chla",
    "ph",
]

# WQI weights used in the project.
WQI_WEIGHTS = {
    "ph": 0.15,
    "dissolved_oxygen": 0.15,
    "turbidity": 0.15,
    "temperature": 0.05,
    "conductivity": 0.10,
    "tds": 0.10,
    "tss": 0.10,
    "total_nitrogen": 0.05,
    "total_phosphorus": 0.05,
    "chla": 0.10,
}

# Project WQI categories.
WQI_CATEGORIES = {
    "Excellent": (90, 100),
    "Good": (70, 89.999),
    "Medium": (50, 69.999),
    "Poor": (25, 49.999),
    "Very Poor": (0, 24.999),
}

# Project screening thresholds.
# These are screening values and should not be presented
# as legal/regulatory limits without verification.
SCREENING_THRESHOLDS = {
    "ph": {
        "label": "pH",
        "minimum": 6.5,
        "maximum": 8.5,
        "unit": "",
    },
    "dissolved_oxygen": {
        "label": "Dissolved Oxygen",
        "minimum": 5.0,
        "maximum": None,
        "unit": "mg/L",
    },
    "turbidity": {
        "label": "Turbidity",
        "minimum": None,
        "maximum": 5.0,
        "unit": "NTU",
    },
    "temperature": {
        "label": "Temperature",
        "minimum": 15.0,
        "maximum": 30.0,
        "unit": "°C",
    },
}


# ============================================================
# CUSTOM CSS

st.markdown(
    """
    <style>

        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1rem;
            color: #555;
            margin-bottom: 1.2rem;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            padding-bottom: 3rem;
        }

        .warning-box,
        .info-box,
        .success-box {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        .warning-box {
            border: 1px solid #e0b84c;
            background-color: #fff8df;
        }

        .info-box {
            border: 1px solid #9ecae1;
            background-color: #eef8fc;
        }

        .success-box {
            border: 1px solid #9ccc9c;
            background-color: #eef8ee;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #666;
        }

        .small-text {
            font-size: 0.82rem;
            color: #666;
        }

        div[data-testid="stMetric"] {
            border-radius: 8px;
            padding: 0.75rem;
        }

        div[data-testid="stDataFrame"] {
            width: 100%;
            overflow-x: auto;
        }

        div[data-testid="stPlotlyChart"] {
            width: 100%;
        }

        div.stButton > button {
            min-height: 42px;
            border-radius: 8px;
            font-weight: 600;
        }

        div[data-baseweb="select"] {
            width: 100%;
        }

        @media (max-width: 768px) {

            .block-container {
                padding-top: 1rem;
                padding-left: 0.75rem;
                padding-right: 0.75rem;
                padding-bottom: 2rem;
            }

            .main-title {
                font-size: 1.65rem;
                line-height: 1.2;
            }

            .subtitle {
                font-size: 0.9rem;
                line-height: 1.4;
            }

            .warning-box,
            .info-box,
            .success-box {
                padding: 12px;
                margin-bottom: 10px;
            }

            .metric-label {
                font-size: 0.8rem;
            }

            .small-text {
                font-size: 0.78rem;
            }

            div[data-testid="stMetric"] {
                padding: 0.5rem;
            }

            div.stButton > button {
                width: 100%;
                min-height: 44px;
                font-size: 0.95rem;
            }

            div[data-testid="stDataFrame"] {
                max-width: 100%;
                overflow-x: auto;
            }

            div[data-testid="stPlotlyChart"] {
                max-width: 100%;
                overflow-x: hidden;
            }

            button[data-baseweb="tab"] {
                font-size: 0.85rem;
                padding-left: 0.6rem;
                padding-right: 0.6rem;
            }
        }

        @media (max-width: 480px) {

            .block-container {
                padding-left: 0.5rem;
                padding-right: 0.5rem;
            }

            .main-title {
                font-size: 1.4rem;
            }

            .subtitle {
                font-size: 0.82rem;
            }

            div[data-testid="stMetric"] {
                padding: 0.4rem;
            }

            button[data-baseweb="tab"] {
                font-size: 0.75rem;
            }
        }

        @media (min-width: 1200px) {

            .block-container {
                padding-left: 3rem;
                padding-right: 3rem;
            }

            .main-title {
                font-size: 2.3rem;
            }
        }

    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def file_exists(path):
    """Return True if a file exists."""
    return os.path.isfile(path)


def safe_numeric(series):
    """Convert a pandas Series to numeric without deprecated options."""
    return pd.to_numeric(series, errors="coerce")


def clean_chla(series):
    """
    Chlorophyll-a values below zero are non-physical.
    They are retained in the original dataset but converted
    to zero for model/WQI calculations.
    """
    numeric = safe_numeric(series)
    return numeric.clip(lower=0)


def classify_wqi(wqi):
    """Convert WQI value into project WQI category."""
    if pd.isna(wqi):
        return "Unknown"

    value = float(wqi)

    if value >= 90:
        return "Excellent"
    elif value >= 70:
        return "Good"
    elif value >= 50:
        return "Medium"
    elif value >= 25:
        return "Poor"

    return "Very Poor"


def classify_risk(wqi):
    """
    Convert WQI into project screening risk.

    WQI < 50  = High
    50-69.99  = Medium
    >=70      = Low
    """
    if pd.isna(wqi):
        return "Unknown"

    value = float(wqi)

    if value < 50:
        return "High"
    elif value < 70:
        return "Medium"

    return "Low"


def format_number(value, decimals=2):
    """Safely format numerical values."""
    if pd.isna(value):
        return "N/A"

    return f"{float(value):,.{decimals}f}"


def get_wqi_column(df):
    """Find a WQI column in a dataframe."""
    candidates = [
        "wqi",
        "WQI",
        "measured_wqi",
        "actual_wqi",
        "calculated_wqi",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def get_prediction_column(df):
    """Find a predicted WQI column."""
    candidates = [
        "predicted_wqi",
        "Predicted_WQI",
        "ml_predicted_wqi",
        "prediction",
        "predicted",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def get_category_column(df):
    """Find a WQI category column."""
    candidates = [
        "wqi_category",
        "WQI_Category",
        "category",
        "measured_category",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


# ============================================================
# WQI FALLBACK CALCULATION
# ============================================================

def calculate_screening_wqi(df):
    """
    Transparent fallback WQI calculation.

    The application first attempts to use the WQI generated
    during model training from athi_model_predictions.csv.

    This function is only used when that reference WQI is
    unavailable.

    It creates parameter scores using project screening
    reference values and combines them using the established
    project weights.

    It should therefore be regarded as a screening index,
    not a regulatory WQI calculation.
    """

    result = pd.Series(
        0.0,
        index=df.index,
        dtype=float,
    )

    total_weight = 0.0

    # --------------------------------------------------------
    # pH
    # --------------------------------------------------------

    if "ph" in df.columns:
        ph = safe_numeric(df["ph"])

        score = np.where(
            (ph >= 6.5) & (ph <= 8.5),
            100,
            np.maximum(
                0,
                100 - (np.abs(ph - 7.5) / 3.0) * 100,
            ),
        )

        result += pd.Series(
            score,
            index=df.index,
        ) * WQI_WEIGHTS["ph"]

        total_weight += WQI_WEIGHTS["ph"]

    # --------------------------------------------------------
    # Dissolved Oxygen
    # --------------------------------------------------------

    if "dissolved_oxygen" in df.columns:
        do = safe_numeric(df["dissolved_oxygen"])

        score = np.minimum(
            100,
            np.maximum(
                0,
                do / 8.0 * 100,
            ),
        )

        result += score * WQI_WEIGHTS["dissolved_oxygen"]
        total_weight += WQI_WEIGHTS["dissolved_oxygen"]

    # --------------------------------------------------------
    # Turbidity
    # --------------------------------------------------------

    if "turbidity" in df.columns:
        turbidity = safe_numeric(df["turbidity"])

        score = (
            100
            * np.exp(-turbidity / 100.0)
        )

        result += score * WQI_WEIGHTS["turbidity"]
        total_weight += WQI_WEIGHTS["turbidity"]

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    if "temperature" in df.columns:
        temperature = safe_numeric(df["temperature"])

        score = np.where(
            (temperature >= 15) & (temperature <= 30),
            100,
            np.maximum(
                0,
                100 - np.abs(temperature - 22.5) * 5,
            ),
        )

        result += pd.Series(
            score,
            index=df.index,
        ) * WQI_WEIGHTS["temperature"]

        total_weight += WQI_WEIGHTS["temperature"]

    # --------------------------------------------------------
    # Conductivity
    # --------------------------------------------------------

    if "conductivity" in df.columns:
        conductivity = safe_numeric(df["conductivity"])

        score = (
            100
            * np.exp(-conductivity / 2000.0)
        )

        result += score * WQI_WEIGHTS["conductivity"]
        total_weight += WQI_WEIGHTS["conductivity"]

    # --------------------------------------------------------
    # TDS
    # --------------------------------------------------------

    if "tds" in df.columns:
        tds = safe_numeric(df["tds"])

        score = (
            100
            * np.exp(-tds / 1500.0)
        )

        result += score * WQI_WEIGHTS["tds"]
        total_weight += WQI_WEIGHTS["tds"]

    # --------------------------------------------------------
    # TSS
    # --------------------------------------------------------

    if "tss" in df.columns:
        tss = safe_numeric(df["tss"])

        score = (
            100
            * np.exp(-tss / 1000.0)
        )

        result += score * WQI_WEIGHTS["tss"]
        total_weight += WQI_WEIGHTS["tss"]

    # --------------------------------------------------------
    # Total Nitrogen
    # --------------------------------------------------------

    if "total_nitrogen" in df.columns:
        nitrogen = safe_numeric(df["total_nitrogen"])

        score = (
            100
            * np.exp(-nitrogen / 20.0)
        )

        result += score * WQI_WEIGHTS["total_nitrogen"]
        total_weight += WQI_WEIGHTS["total_nitrogen"]

    # --------------------------------------------------------
    # Total Phosphorus
    # --------------------------------------------------------

    if "total_phosphorus" in df.columns:
        phosphorus = safe_numeric(df["total_phosphorus"])

        score = (
            100
            * np.exp(-phosphorus / 5.0)
        )

        result += score * WQI_WEIGHTS["total_phosphorus"]
        total_weight += WQI_WEIGHTS["total_phosphorus"]

    # --------------------------------------------------------
    # Chlorophyll-a
    # --------------------------------------------------------

    if "chla" in df.columns:
        chla = clean_chla(df["chla"])

        score = (
            100
            * np.exp(-chla / 20.0)
        )

        result += score * WQI_WEIGHTS["chla"]
        total_weight += WQI_WEIGHTS["chla"]

    if total_weight == 0:
        return pd.Series(
            np.nan,
            index=df.index,
        )

    return (
        result / total_weight
    ).clip(
        lower=0,
        upper=100,
    )


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_athi_model():
    """Load the Athi Random Forest model and supporting files."""

    if not file_exists(REGRESSOR_FILE):
        raise FileNotFoundError(
            f"Athi regressor not found:\n{REGRESSOR_FILE}"
        )

    if not file_exists(FEATURE_FILE):
        raise FileNotFoundError(
            f"Athi model feature file not found:\n{FEATURE_FILE}"
        )

    model = joblib.load(REGRESSOR_FILE)
    features = joblib.load(FEATURE_FILE)

    metadata = {}

    if file_exists(METADATA_FILE):
        try:
            metadata = joblib.load(METADATA_FILE)
        except Exception:
            metadata = {}

    if isinstance(features, np.ndarray):
        features = features.tolist()

    if isinstance(features, tuple):
        features = list(features)

    return model, features, metadata


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_athi_data():
    """Load and prepare the Upper Athi water-quality dataset."""

    if not file_exists(DATA_FILE):
        raise FileNotFoundError(
            f"Upper Athi dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    # Clean column names.
    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    # Numeric conversion.
    numeric_columns = [
        column
        for column in df.columns
        if column not in [
            "station",
            "station_name",
        ]
    ]

    for column in numeric_columns:
        df[column] = safe_numeric(df[column])

    # Create cleaned chlorophyll-a without overwriting raw data.
    if "chla" in df.columns:
        df["chla_raw"] = df["chla"]
        df["chla_clean"] = clean_chla(df["chla"])

    return df


# ============================================================
# LOAD REFERENCE MODEL PREDICTIONS
# ============================================================

@st.cache_data
def load_reference_predictions():
    """Load predictions produced during model training."""

    if not file_exists(PREDICTIONS_FILE):
        return None

    try:
        predictions = pd.read_csv(
            PREDICTIONS_FILE
        )

        predictions.columns = [
            str(column).strip().lower()
            for column in predictions.columns
        ]

        return predictions

    except Exception:
        return None


# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

@st.cache_data
def load_feature_importance():
    """Load saved Random Forest feature importance."""

    if not file_exists(FEATURE_IMPORTANCE_FILE):
        return None

    try:
        importance = pd.read_csv(
            FEATURE_IMPORTANCE_FILE
        )

        importance.columns = [
            str(column).strip().lower()
            for column in importance.columns
        ]

        return importance

    except Exception:
        return None


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(df, model, model_features):
    """
    Prepare the final application dataframe.

    Preference is given to the WQI/predictions generated
    by the training script.
    """

    data = df.copy()

    # --------------------------------------------------------
    # Measured WQI
    # --------------------------------------------------------

    measured_wqi = None

    reference = load_reference_predictions()

    if reference is not None:

        reference_wqi_column = get_wqi_column(reference)

        if (
            reference_wqi_column is not None
            and "station" in reference.columns
            and "year" in reference.columns
            and "station" in data.columns
            and "year" in data.columns
        ):

            reference_small = reference[
                [
                    "station",
                    "year",
                    reference_wqi_column,
                ]
            ].copy()

            reference_small = reference_small.rename(
                columns={
                    reference_wqi_column: "reference_wqi"
                }
            )

            reference_small["year"] = safe_numeric(
                reference_small["year"]
            )

            data["year"] = safe_numeric(
                data["year"]
            )

            data = data.merge(
                reference_small,
                on=[
                    "station",
                    "year",
                ],
                how="left",
            )

            measured_wqi = data[
                "reference_wqi"
            ]

    # If the original dataset itself contains WQI.
    if measured_wqi is None:

        dataset_wqi_column = get_wqi_column(data)

        if dataset_wqi_column is not None:
            measured_wqi = safe_numeric(
                data[dataset_wqi_column]
            )

    # If no training WQI exists, use screening fallback.
    if measured_wqi is None:
        measured_wqi = calculate_screening_wqi(
            data
        )

    data["measured_wqi"] = pd.to_numeric(
        measured_wqi,
        errors="coerce",
    ).clip(
        lower=0,
        upper=100,
    )

    # --------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------

    # Build model input from the application dataframe.
    X = data.copy()

    # The application normalizes dataset columns to lowercase,
    # while the trained model was fitted using "pH".
    if "pH" in model_features and "pH" not in X.columns and "ph" in X.columns:
        X["pH"] = X["ph"]

    # Use cleaned chlorophyll-a for modelling when available.
    if "chla" in model_features and "chla_clean" in X.columns:
        X["chla"] = X["chla_clean"]

    # Confirm that every model feature is available.
    missing_features = [
        feature
        for feature in model_features
        if feature not in X.columns
    ]

    if missing_features:
        data["predicted_wqi"] = np.nan

    else:
        X = X[model_features].copy()

        for column in model_features:
            X[column] = safe_numeric(X[column])

        valid_rows = X.notna().all(axis=1)

        predictions = pd.Series(
            np.nan,
            index=data.index,
            dtype=float,
        )

        if valid_rows.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                predictions.loc[valid_rows] = model.predict(
                    X.loc[
                        valid_rows,
                        model_features,
                    ]
                )

        data["predicted_wqi"] = (
            predictions
            .clip(
                lower=0,
                upper=100,
            )
        )

    # --------------------------------------------------------
    # Categories and risk
    # --------------------------------------------------------

    data["measured_category"] = (
        data["measured_wqi"]
        .apply(classify_wqi)
    )

    data["predicted_category"] = (
        data["predicted_wqi"]
        .apply(classify_wqi)
    )

    data["screening_risk"] = (
        data["measured_wqi"]
        .apply(classify_risk)
    )

    data["predicted_risk"] = (
        data["predicted_wqi"]
        .apply(classify_risk)
    )

    # Prediction error.
    data["prediction_error"] = (
        data["predicted_wqi"]
        - data["measured_wqi"]
    )

    data["absolute_error"] = (
        data["prediction_error"]
        .abs()
    )

    return data


# ============================================================
# SCREENING FLAGS
# ============================================================

def add_screening_flags(df):
    """Add parameter screening flags."""

    data = df.copy()

    for parameter, threshold in SCREENING_THRESHOLDS.items():

        if parameter not in data.columns:
            continue

        values = safe_numeric(
            data[parameter]
        )

        flag = pd.Series(
            False,
            index=data.index,
        )

        if threshold["minimum"] is not None:
            flag |= (
                values
                < threshold["minimum"]
            )

        if threshold["maximum"] is not None:
            flag |= (
                values
                > threshold["maximum"]
            )

        data[
            f"{parameter}_exceeded"
        ] = flag

    return data


# ============================================================
# FUTURE FORECASTING IMPORT
# ============================================================
from core.future_dashboard import show_future_forecasting

# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render application navigation."""

    st.sidebar.markdown(
        "## ML-SWQM-DSS"
    )

    st.sidebar.caption(
        "Upper Athi River Catchment"
    )

    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Monitoring Data",
            "ML Predictions",
            "Risk Analysis",
            "Water Quality Analysis",
            "Model Information",
            "Data Quality",
            "Petroleum DSS",
            "Station Investigation",
            "Station Comparison",
            "Petroleum Event Screening",
            "Future Forecasting",
            "About",
        ],
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        "**Case Study**"
    )

    st.sidebar.caption(
        "Upper Athi River Catchment, Kenya"
    )

    st.sidebar.markdown(
        "**Model**"
    )

    st.sidebar.caption(
        "Random Forest Regression"
    )

    st.sidebar.markdown(
        "**Dataset**"
    )

    st.sidebar.caption(
        "18 stations Ã— 2 years = 36 observations"
    )

    return page


# ============================================================
# HEADER
# ============================================================

def render_header():
    """Render application header."""

    st.markdown(
        '<div class="main-title">'
        'ML-SWQM-DSS'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning-based Surface Water Quality '
        'Monitoring and Decision Support System'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-box">'
        '<b>Case Study:</b> Upper Athi River Catchment, Kenya'
        '<br>'
        '<b>Application focus:</b> Surface-water quality '
        'screening in environments potentially influenced '
        'by urban, industrial and petroleum-related activities.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "Academic/exploratory baseline: the current Athi model "
        "uses 36 station-year observations. It should not be "
        "interpreted as a fully operational forecasting system "
        "or as proof that observed contamination is caused by "
        "petroleum activities."
    )


# ============================================================
# DASHBOARD
# ============================================================

def show_dashboard(data):
    """Main dashboard."""

    st.header(
        "Water Quality Dashboard"
    )

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    total_records = len(data)

    station_count = (
        data["station"].nunique()
        if "station" in data.columns
        else 0
    )

    year_count = (
        data["year"].nunique()
        if "year" in data.columns
        else 0
    )

    mean_wqi = data[
        "measured_wqi"
    ].mean()

    high_risk = (
        data["screening_risk"]
        == "High"
    ).sum()

    medium_risk = (
        data["screening_risk"]
        == "Medium"
    ).sum()

    low_risk = (
        data["screening_risk"]
        == "Low"
    ).sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Records",
        total_records,
    )

    col2.metric(
        "Stations",
        station_count,
    )

    col3.metric(
        "Years",
        year_count,
    )

    col4.metric(
        "Mean WQI",
        format_number(mean_wqi),
    )

    col5.metric(
        "High-Risk Records",
        high_risk,
    )

    st.divider()

    # --------------------------------------------------------
    # WQI distribution
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader(
            "WQI Category Distribution"
        )

        category_counts = (
            data["measured_category"]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = [
            "Category",
            "Count",
        ]

        fig = px.bar(
            category_counts,
            x="Category",
            y="Count",
            text="Count",
            title="Measured WQI Categories",
        )

        fig.update_layout(
            xaxis_title="WQI Category",
            yaxis_title="Number of Records",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    with right:

        st.subheader(
            "Screening Risk Distribution"
        )

        risk_counts = (
            data["screening_risk"]
            .value_counts()
            .reindex(
                [
                    "Low",
                    "Medium",
                    "High",
                ],
                fill_value=0,
            )
            .reset_index()
        )

        risk_counts.columns = [
            "Risk",
            "Count",
        ]

        fig = px.bar(
            risk_counts,
            x="Risk",
            y="Count",
            text="Count",
            title="WQI-Based Screening Risk",
        )

        fig.update_layout(
            xaxis_title="Risk Level",
            yaxis_title="Number of Records",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    # --------------------------------------------------------
    # Station WQI
    # --------------------------------------------------------

    st.subheader(
        "Measured WQI by Monitoring Station"
    )

    station_summary = (
        data.groupby(
            [
                "station",
                "station_name",
            ],
            as_index=False,
        )[
            "measured_wqi"
        ]
        .mean()
        .sort_values(
            "measured_wqi"
        )
    )

    fig = px.bar(
        station_summary,
        x="station",
        y="measured_wqi",
        hover_name="station_name",
        title="Average Measured WQI by Station",
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="Good WQI threshold",
    )

    fig.add_hline(
        y=50,
        line_dash="dash",
        annotation_text="Medium WQI threshold",
    )

    fig.update_layout(
        xaxis_title="Monitoring Station",
        yaxis_title="WQI",
        yaxis_range=[0, 100],
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    # --------------------------------------------------------
    # Risk summary
    # --------------------------------------------------------

    st.subheader(
        "Current Screening Summary"
    )

    c1, c2, c3 = st.columns(3)

    c1.success(
        f"Low Risk: **{low_risk}** records"
    )

    c2.warning(
        f"Medium Risk: **{medium_risk}** records"
    )

    c3.error(
        f"High Risk: **{high_risk}** records"
    )


# ============================================================
# MONITORING DATA
# ============================================================

def show_monitoring_data(data):
    """Monitoring data page."""

    st.header(
        "Upper Athi Monitoring Data"
    )

    st.write(
        "Explore the station-level water-quality observations "
        "used by the DSS."
    )

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    stations = sorted(
        data["station"].dropna().unique()
    )

    years = sorted(
        data["year"].dropna().unique()
    )

    categories = [
        "All",
        "Excellent",
        "Good",
        "Medium",
        "Poor",
        "Very Poor",
    ]

    with col1:
        selected_stations = st.multiselect(
            "Monitoring Stations",
            stations,
            default=stations,
        )

    with col2:
        selected_years = st.multiselect(
            "Year",
            years,
            default=years,
        )

    with col3:
        selected_category = st.selectbox(
            "WQI Category",
            categories,
        )

    filtered = data[
        data["station"].isin(
            selected_stations
        )
        &
        data["year"].isin(
            selected_years
        )
    ].copy()

    if selected_category != "All":
        filtered = filtered[
            filtered["measured_category"]
            == selected_category
        ]

    st.metric(
        "Filtered Records",
        len(filtered),
    )

    # --------------------------------------------------------
    # Display table
    # --------------------------------------------------------

    preferred_columns = [
        "station",
        "station_name",
        "year",
        "ph",
        "dissolved_oxygen",
        "turbidity",
        "temperature",
        "conductivity",
        "tds",
        "tss",
        "total_nitrogen",
        "total_phosphorus",
        "chla",
        "chloride",
        "chromium",
        "copper",
        "iron",
        "manganese",
        "lead",
        "zinc",
        "measured_wqi",
        "measured_category",
        "screening_risk",
    ]

    display_columns = [
        column
        for column in preferred_columns
        if column in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns],
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "??Â Download Filtered Data",
        data=csv_data,
        file_name="upper_athi_filtered_data.csv",
        mime="text/csv",
    )


# ============================================================
# ML PREDICTIONS
# ============================================================

def show_ml_predictions(data, model_features):
    """Machine-learning prediction page."""

    st.header(
        "Machine Learning Predictions"
    )

    st.write(
        "The Random Forest regression model estimates Water "
        "Quality Index (WQI) from the selected water-quality "
        "parameters."
    )

    st.markdown(
        '<div class="info-box">'
        '<b>Model architecture:</b> Random Forest Regressor'
        '<br>'
        '<b>Input:</b> 10 water-quality features'
        '<br>'
        '<b>Output:</b> Predicted WQI (0â€“100)'
        '<br>'
        '<b>Risk:</b> Derived from predicted WQI rather than '
        'from a separate classifier.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Prediction comparison
    # --------------------------------------------------------

    comparison = data[
        [
            "station",
            "station_name",
            "year",
            "measured_wqi",
            "predicted_wqi",
            "measured_category",
            "predicted_category",
            "prediction_error",
            "absolute_error",
        ]
    ].copy()

    st.subheader(
        "Measured vs ML-Predicted WQI"
    )

    fig = px.scatter(
        comparison,
        x="measured_wqi",
        y="predicted_wqi",
        hover_name="station_name",
        hover_data=[
            "station",
            "year",
            "measured_category",
            "predicted_category",
        ],
        title="Measured WQI vs Random Forest Prediction",
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Perfect prediction",
        )
    )

    fig.update_layout(
        xaxis_title="Measured WQI",
        yaxis_title="Predicted WQI",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    # --------------------------------------------------------
    # Error metrics
    # --------------------------------------------------------

    valid = comparison.dropna(
        subset=[
            "measured_wqi",
            "predicted_wqi",
        ]
    )

    if len(valid) > 0:

        mae = (
            valid["absolute_error"]
            .mean()
        )

        rmse = np.sqrt(
            np.mean(
                valid["prediction_error"] ** 2
            )
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Prediction MAE",
            format_number(mae),
        )

        c2.metric(
            "Prediction RMSE",
            format_number(rmse),
        )

    st.subheader(
        "Prediction Results"
    )

    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # Individual prediction
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "?? Inspect an Individual Observation"
    )

    options = data.index.tolist()

    if options:

        selected_index = st.selectbox(
            "Select observation",
            options,
            format_func=lambda x: (
                f"{data.loc[x, 'station']} - "
                f"{data.loc[x, 'station_name']} - "
                f"{int(data.loc[x, 'year'])}"
            ),
        )

        row = data.loc[
            selected_index
        ]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Measured WQI",
            format_number(
                row["measured_wqi"]
            ),
        )

        c2.metric(
            "Predicted WQI",
            format_number(
                row["predicted_wqi"]
            ),
        )

        c3.metric(
            "Measured Category",
            row["measured_category"],
        )

        c4.metric(
            "Screening Risk",
            row["screening_risk"],
        )

        st.markdown(
            "### Model Input Values"
        )

        input_values = {}

        for feature in model_features:
            if feature in row.index:
                input_values[feature] = row[
                    feature
                ]

        input_df = pd.DataFrame(
            {
                "Feature": list(
                    input_values.keys()
                ),
                "Value": list(
                    input_values.values()
                ),
            }
        )

        st.dataframe(
            input_df,
            width="stretch",
            hide_index=True,
        )

    prediction_csv = data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "??Â Download ML Prediction Results",
        data=prediction_csv,
        file_name="athi_ml_predictions.csv",
        mime="text/csv",
    )


# ============================================================
# RISK ANALYSIS
# ============================================================

def show_risk_analysis(data):
    """Risk assessment page."""

    st.header(
        "??Â Water Quality Risk Analysis"
    )

    st.write(
        "Risk is classified from the WQI as a screening "
        "indicator. It is not a regulatory compliance "
        "classification."
    )

    # --------------------------------------------------------
    # Risk table
    # --------------------------------------------------------

    risk_counts = (
        data["screening_risk"]
        .value_counts()
        .reindex(
            [
                "Low",
                "Medium",
                "High",
            ],
            fill_value=0,
        )
    )

    total = len(data)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Low Risk",
        int(risk_counts["Low"]),
        f"{risk_counts['Low'] / total * 100:.1f}%",
    )

    c2.metric(
        "Medium Risk",
        int(risk_counts["Medium"]),
        f"{risk_counts['Medium'] / total * 100:.1f}%",
    )

    c3.metric(
        "High Risk",
        int(risk_counts["High"]),
        f"{risk_counts['High'] / total * 100:.1f}%",
    )

    st.divider()

    # --------------------------------------------------------
    # Station risk
    # --------------------------------------------------------

    st.subheader(
        "Risk by Monitoring Station"
    )

    risk_station = (
        data.groupby(
            [
                "station",
                "station_name",
            ]
        )[
            "screening_risk"
        ]
        .value_counts()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    for risk in [
        "Low",
        "Medium",
        "High",
    ]:
        if risk not in risk_station.columns:
            risk_station[risk] = 0

    fig = px.bar(
        risk_station,
        x="station",
        y=[
            "Low",
            "Medium",
            "High",
        ],
        hover_name="station_name",
        title="Screening Risk Distribution by Station",
    )

    fig.update_layout(
        xaxis_title="Station",
        yaxis_title="Number of Records",
        barmode="stack",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    # --------------------------------------------------------
    # Parameter screening
    # --------------------------------------------------------

    st.subheader(
        "Parameter Screening"
    )

    screened = add_screening_flags(
        data
    )

    results = []

    for parameter, threshold in SCREENING_THRESHOLDS.items():

        flag_column = (
            f"{parameter}_exceeded"
        )

        if flag_column not in screened.columns:
            continue

        exceeded = int(
            screened[
                flag_column
            ].sum()
        )

        results.append(
            {
                "Parameter": threshold["label"],
                "Unit": threshold["unit"],
                "Records Exceeding Screening Threshold": exceeded,
                "Percentage": (
                    exceeded / len(screened) * 100
                    if len(screened) > 0
                    else 0
                ),
            }
        )

    screening_table = pd.DataFrame(
        results
    )

    st.dataframe(
        screening_table,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Screening thresholds shown above are project-level "
        "screening criteria. Confirm applicable Kenyan "
        "regulatory standards before operational use."
    )

    # --------------------------------------------------------
    # High-risk records
    # --------------------------------------------------------

    st.subheader(
        "High-Risk Observations"
    )

    high_risk = data[
        data["screening_risk"]
        == "High"
    ].copy()

    if high_risk.empty:

        st.success(
            "No high-risk observations were identified "
            "using the current WQI screening classification."
        )

    else:

        columns = [
            "station",
            "station_name",
            "year",
            "measured_wqi",
            "measured_category",
            "screening_risk",
            "dissolved_oxygen",
            "turbidity",
            "tss",
        ]

        columns = [
            column
            for column in columns
            if column in high_risk.columns
        ]

        st.dataframe(
            high_risk[columns],
            width="stretch",
            hide_index=True,
        )


# ============================================================
# WATER QUALITY ANALYSIS
# ============================================================

def show_water_quality_analysis(data):
    """Detailed parameter analysis."""

    st.header(
        "ðŸ“ˆ Water Quality Analysis"
    )

    parameters = [
        column
        for column in [
            "ph",
            "dissolved_oxygen",
            "turbidity",
            "temperature",
            "conductivity",
            "tds",
            "tss",
            "total_nitrogen",
            "total_phosphorus",
            "chla",
            "chloride",
            "chromium",
            "copper",
            "iron",
            "manganese",
            "lead",
            "zinc",
        ]
        if column in data.columns
    ]

    if not parameters:
        st.error(
            "No water-quality parameters are available."
        )
        return

    selected_parameter = st.selectbox(
        "Select water-quality parameter",
        parameters,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    values = safe_numeric(
        data[selected_parameter]
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Minimum",
        format_number(values.min()),
    )

    c2.metric(
        "Maximum",
        format_number(values.max()),
    )

    c3.metric(
        "Mean",
        format_number(values.mean()),
    )

    c4.metric(
        "Median",
        format_number(values.median()),
    )

    st.divider()

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        fig = px.histogram(
            data,
            x=selected_parameter,
            nbins=15,
            title=(
                f"Distribution of "
                f"{selected_parameter}"
            ),
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    with right:

        fig = px.box(
            data,
            y=selected_parameter,
            title=(
                f"Box Plot of "
                f"{selected_parameter}"
            ),
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    # --------------------------------------------------------
    # Station comparison
    # --------------------------------------------------------

    if "station" in data.columns:

        station_parameter = (
            data.groupby(
                [
                    "station",
                    "station_name",
                ],
                as_index=False,
            )[
                selected_parameter
            ]
            .mean()
            .sort_values(
                selected_parameter
            )
        )

        fig = px.bar(
            station_parameter,
            x="station",
            y=selected_parameter,
            hover_name="station_name",
            title=(
                f"Average {selected_parameter} "
                "by Monitoring Station"
            ),
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    # --------------------------------------------------------
    # Correlation with WQI
    # --------------------------------------------------------

    if (
        selected_parameter
        != "measured_wqi"
        and
        data["measured_wqi"].notna().sum()
        > 2
    ):

        correlation = data[
            [
                selected_parameter,
                "measured_wqi",
            ]
        ].corr().iloc[0, 1]

        st.metric(
            "Correlation with Measured WQI",
            format_number(
                correlation,
                3,
            ),
        )

        st.caption(
            "Correlation indicates statistical association "
            "within this dataset; it does not establish causation."
        )


# ============================================================
# CORRELATION MATRIX
# ============================================================

def show_correlation_matrix(data):
    """Display correlation matrix."""

    numeric_columns = [
        column
        for column in data.columns
        if pd.api.types.is_numeric_dtype(
            data[column]
        )
    ]

    # Remove derived columns.
    excluded = [
        "year",
        "measured_wqi",
        "predicted_wqi",
        "prediction_error",
        "absolute_error",
    ]

    numeric_columns = [
        column
        for column in numeric_columns
        if column not in excluded
    ]

    if len(numeric_columns) < 2:
        st.info(
            "Not enough numerical variables "
            "for a correlation matrix."
        )
        return

    correlation = data[
        numeric_columns
    ].corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Water Quality Parameter Correlation Matrix",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

def show_model_information(
    model,
    model_features,
    metadata,
):
    """Display model information."""

    st.header(
        "Model Information"
    )

    st.markdown(
        '<div class="info-box">'
        '<b>Model:</b> Random Forest Regressor'
        '<br>'
        '<b>Purpose:</b> Exploratory WQI prediction'
        '<br>'
        '<b>Case Study:</b> Upper Athi River Catchment'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Model properties
    # --------------------------------------------------------

    st.subheader(
        "Model Configuration"
    )

    properties = {}

    if hasattr(
        model,
        "n_estimators",
    ):
        properties[
            "Number of Trees"
        ] = model.n_estimators

    if hasattr(
        model,
        "max_depth",
    ):
        properties[
            "Maximum Depth"
        ] = model.max_depth

    if hasattr(
        model,
        "random_state",
    ):
        properties[
            "Random State"
        ] = model.random_state

    if hasattr(
        model,
        "n_features_in_",
    ):
        properties[
            "Number of Model Features"
        ] = model.n_features_in_

    properties[
        "Saved Feature Count"
    ] = len(model_features)

    properties[
        "Model File"
    ] = os.path.basename(
        REGRESSOR_FILE
    )

    properties_df = pd.DataFrame(
    [
        {
            "Property": key,
            "Value": str(value),
        }
        for key, value in properties.items()
    ]
)

    st.dataframe(
        properties_df,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    st.subheader(
        "Model Input Features"
    )

    feature_df = pd.DataFrame(
        {
            "Feature": model_features,
            "WQI Weight": [
                WQI_WEIGHTS.get(
                    feature,
                    np.nan,
                )
                for feature in model_features
            ],
        }
    )

    st.dataframe(
        feature_df,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    st.subheader(
        "Random Forest Feature Importance"
    )

    importance = load_feature_importance()

    if importance is not None:

        # Identify columns dynamically.
        feature_column = None
        importance_column = None

        for column in [
            "feature",
            "features",
            "variable",
            "name",
        ]:
            if column in importance.columns:
                feature_column = column
                break

        for column in [
            "importance",
            "feature_importance",
            "importance_score",
        ]:
            if column in importance.columns:
                importance_column = column
                break

        if (
            feature_column is not None
            and importance_column is not None
        ):

            importance[
                importance_column
            ] = safe_numeric(
                importance[
                    importance_column
                ]
            )

            importance = importance.sort_values(
                importance_column,
                ascending=True,
            )

            fig = px.bar(
                importance,
                x=importance_column,
                y=feature_column,
                orientation="h",
                title="Feature Importance",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            st.dataframe(
                importance.sort_values(
                    importance_column,
                    ascending=False,
                ),
                width="stretch",
                hide_index=True,
            )

        else:
            st.dataframe(
                importance,
                width="stretch",
                hide_index=True,
            )

    else:

        # Try the model directly.
        if hasattr(
            model,
            "feature_importances_",
        ):

            importance_df = pd.DataFrame(
                {
                    "Feature": model_features,
                    "Importance": (
                        model.feature_importances_
                    ),
                }
            ).sort_values(
                "Importance",
                ascending=False,
            )

            fig = px.bar(
                importance_df.sort_values(
                    "Importance"
                ),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    st.subheader(
        "Training Metadata"
    )

    if metadata:

        try:
            st.json(
                json.loads(
                    json.dumps(
                        metadata,
                        default=str,
                    )
                )
            )
        except Exception:

            st.write(
                metadata
            )

    else:

        st.info(
            "No model metadata file was available."
        )

    # --------------------------------------------------------
    # Important methodological note
    # --------------------------------------------------------

    st.warning(
        "Methodological limitation: the current model was "
        "trained on 36 station-year observations from 18 "
        "monitoring stations over two years. Its R²/MAE/RMSE "
        "results should therefore be treated as exploratory "
        "baseline performance rather than proof of "
        "generalizable future forecasting capability."
    )


# ============================================================
# DATA QUALITY
# ============================================================

def show_data_quality(data):
    """Data-quality diagnostics."""

    st.header(
        "âœ… Data Quality"
    )

    # --------------------------------------------------------
    # Dataset overview
    # --------------------------------------------------------

    missing_total = int(
        data.isna().sum().sum()
    )

    duplicate_rows = int(
        data.duplicated().sum()
    )

    if {
        "station",
        "year",
    }.issubset(
        data.columns
    ):

        duplicate_station_year = int(
            data.duplicated(
                subset=[
                    "station",
                    "year",
                ]
            ).sum()
        )

    else:

        duplicate_station_year = 0

    negative_chla = 0

    if "chla_raw" in data.columns:
        negative_chla = int(
            (
                safe_numeric(
                    data["chla_raw"]
                )
                < 0
            ).sum()
        )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        len(data),
    )

    c2.metric(
        "Columns",
        len(data.columns),
    )

    c3.metric(
        "Missing Values",
        missing_total,
    )

    c4.metric(
        "Duplicate Rows",
        duplicate_rows,
    )

    st.divider()

    # --------------------------------------------------------
    # Validation status
    # --------------------------------------------------------

    if missing_total == 0:
        st.success(
            "No missing values detected in the loaded dataset."
        )
    else:
        st.warning(
            f"{missing_total} missing values were detected."
        )

    if duplicate_rows == 0:
        st.success(
            "No duplicate rows detected."
        )
    else:
        st.warning(
            f"{duplicate_rows} duplicate rows detected."
        )

    if duplicate_station_year == 0:
        st.success(
            "No duplicate station-year observations detected."
        )
    else:
        st.warning(
            f"{duplicate_station_year} duplicate "
            "station-year observations detected."
        )

    if negative_chla > 0:

        st.warning(
            f"{negative_chla} negative chlorophyll-a values "
            "were found in the raw data. These are retained "
            "for traceability but converted to zero for "
            "model/WQI calculations."
        )

    # --------------------------------------------------------
    # Missing values by column
    # --------------------------------------------------------

    st.subheader(
        "Missing Values by Column"
    )

    missing_table = (
        data.isna()
        .sum()
        .reset_index()
    )

    missing_table.columns = [
        "Column",
        "Missing Values",
    ]

    missing_table[
        "Percentage"
    ] = (
        missing_table[
            "Missing Values"
        ]
        / len(data)
        * 100
    )

    st.dataframe(
        missing_table,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    st.subheader(
        "Column Data Types"
    )

    dtype_table = pd.DataFrame(
        {
            "Column": data.columns,
            "Data Type": [
                str(dtype)
                for dtype in data.dtypes
            ],
        }
    )

    st.dataframe(
        dtype_table,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    st.subheader(
        "Numerical Summary"
    )

    numeric = data.select_dtypes(
        include=np.number
    )

    if not numeric.empty:

        st.dataframe(
            numeric.describe()
            .T
            .round(3),
            width="stretch",
        )


# ============================================================
# PETROLEUM DSS
# ============================================================




def show_petroleum_event_screening(data):
    """Petroleum Incident and Environmental Event Screening."""

    st.markdown(
        "## Petroleum Incident & Event Screening"
    )

    st.caption(
        "Screen potential petroleum-related environmental events "
        "and determine the priority for follow-up investigation."
    )

    working = data.copy()

    # --------------------------------------------------------
    # Event information
    # --------------------------------------------------------

    st.markdown("### 1. Event Information")

    event_type = st.selectbox(
        "Event / activity type",
        [
            "Pipeline leak",
            "Petroleum spill",
            "Tank or storage leak",
            "Loading / unloading incident",
            "Road tanker incident",
            "Petroleum facility stormwater runoff",
            "Maintenance / drainage release",
            "Suspected petroleum release",
            "Unknown environmental event",
        ],
        key="petroleum_event_type",
    )

    event_date = st.date_input(
        "Event date",
        key="petroleum_event_date",
    )

    event_duration = st.selectbox(
        "Estimated event duration",
        [
            "Unknown",
            "Less than 1 hour",
            "1â€“6 hours",
            "6â€“24 hours",
            "More than 24 hours",
        ],
        key="petroleum_event_duration",
    )

    estimated_volume = st.selectbox(
        "Estimated release magnitude",
        [
            "Unknown",
            "No visible release",
            "Small",
            "Moderate",
            "Large",
            "Major / uncontrolled",
        ],
        key="petroleum_event_volume",
    )

    # --------------------------------------------------------
    # Environmental pathway
    # --------------------------------------------------------

    st.markdown("### 2. Environmental Pathway")

    pathway = st.multiselect(
        "Potential pathway(s)",
        [
            "Direct discharge to river",
            "Stormwater drainage",
            "Pipeline corridor",
            "Tank farm drainage",
            "Road drainage",
            "Soil / groundwater pathway",
            "Wetland / floodplain",
            "Unknown pathway",
        ],
        default=["Unknown pathway"],
        key="petroleum_event_pathway",
    )

    rainfall = st.selectbox(
        "Rainfall / runoff conditions",
        [
            "Unknown",
            "Dry conditions",
            "Light rainfall",
            "Heavy rainfall / storm event",
        ],
        key="petroleum_event_rainfall",
    )

    visible_sheen = st.selectbox(
        "Visible oil sheen / hydrocarbon indication",
        [
            "Not observed",
            "Unknown",
            "Observed",
        ],
        key="petroleum_visible_sheen",
    )

    # --------------------------------------------------------
    # Monitoring station
    # --------------------------------------------------------

    st.markdown("### 3. Potentially Affected Monitoring Station")

    station_column = (
        "station"
        if "station" in working.columns
        else None
    )

    if station_column is None:
        st.error(
            "The dataset does not contain a station identifier."
        )
        return

    name_column = (
        "station_name"
        if "station_name" in working.columns
        else station_column
    )

    station_table = (
        working[
            [station_column, name_column]
        ]
        .drop_duplicates()
        .sort_values(station_column)
    )

    station_labels = {}

    for _, row in station_table.iterrows():

        station_id = str(row[station_column])
        station_name = str(row[name_column])

        station_labels[
            f"{station_id} â€” {station_name}"
        ] = station_id

    selected_label = st.selectbox(
        "Select potentially affected station",
        list(station_labels.keys()),
        key="petroleum_event_station",
    )

    selected_station = station_labels[selected_label]

    station_data = working[
        working[station_column].astype(str)
        == str(selected_station)
    ].copy()

    # --------------------------------------------------------
    # WQI and risk
    # --------------------------------------------------------

    wqi_column = get_wqi_column(station_data)

    if wqi_column is None:

        station_data["wqi"] = station_data.apply(
            calculate_screening_wqi,
            axis=1
        )

        wqi_column = "wqi"

    station_data[wqi_column] = pd.to_numeric(
        station_data[wqi_column],
        errors="coerce"
    )

    mean_wqi = station_data[wqi_column].mean()

    risk = classify_risk(mean_wqi)

    category = classify_wqi(mean_wqi)

    # --------------------------------------------------------
    # Current environmental condition
    # --------------------------------------------------------

    st.markdown("### 4. Current Environmental Condition")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Mean WQI",
            format_number(mean_wqi, 2),
        )

    with col2:
        st.metric(
            "WQI Category",
            category,
        )

    with col3:
        st.metric(
            "Screening Risk",
            risk,
        )

    # --------------------------------------------------------
    # Water quality screening
    # --------------------------------------------------------

    exceedances = 0

    if "ph" in station_data.columns:

        ph_values = pd.to_numeric(
            station_data["ph"],
            errors="coerce"
        )

        exceedances += int(
            (
                (ph_values < 6.5)
                | (ph_values > 8.5)
            )
            .fillna(False)
            .sum()
        )

    if "turbidity" in station_data.columns:

        turbidity_values = pd.to_numeric(
            station_data["turbidity"],
            errors="coerce"
        )

        exceedances += int(
            (turbidity_values > 5)
            .fillna(False)
            .sum()
        )

    if "dissolved_oxygen" in station_data.columns:

        do_values = pd.to_numeric(
            station_data["dissolved_oxygen"],
            errors="coerce"
        )

        exceedances += int(
            (do_values < 5)
            .fillna(False)
            .sum()
        )

    # --------------------------------------------------------
    # Event scoring
    # --------------------------------------------------------

    st.markdown("### 5. Event Severity Screening")

    event_score = 0

    event_scores = {
        "Pipeline leak": 20,
        "Petroleum spill": 25,
        "Tank or storage leak": 20,
        "Loading / unloading incident": 15,
        "Road tanker incident": 20,
        "Petroleum facility stormwater runoff": 10,
        "Maintenance / drainage release": 10,
        "Suspected petroleum release": 15,
        "Unknown environmental event": 5,
    }

    event_score += event_scores.get(
        event_type,
        5
    )

    duration_scores = {
        "Unknown": 0,
        "Less than 1 hour": 2,
        "1â€“6 hours": 5,
        "6â€“24 hours": 10,
        "More than 24 hours": 15,
    }

    event_score += duration_scores.get(
        event_duration,
        0
    )

    volume_scores = {
        "Unknown": 0,
        "No visible release": 0,
        "Small": 3,
        "Moderate": 7,
        "Large": 12,
        "Major / uncontrolled": 20,
    }

    event_score += volume_scores.get(
        estimated_volume,
        0
    )

    if "Direct discharge to river" in pathway:
        event_score += 15

    if "Stormwater drainage" in pathway:
        event_score += 8

    if "Tank farm drainage" in pathway:
        event_score += 8

    if "Pipeline corridor" in pathway:
        event_score += 5

    if rainfall == "Heavy rainfall / storm event":
        event_score += 8

    if visible_sheen == "Observed":
        event_score += 15

    # Existing water-quality condition
    if risk == "High":
        event_score += 20
    elif risk == "Medium":
        event_score += 10

    # Existing exceedances
    event_score += min(
        exceedances * 2,
        20
    )

    event_score = min(
        event_score,
        100
    )

    if event_score >= 70:
        event_priority = "Critical"
    elif event_score >= 45:
        event_priority = "High"
    elif event_score >= 20:
        event_priority = "Medium"
    else:
        event_priority = "Low"

    # --------------------------------------------------------
    # Display event score
    # --------------------------------------------------------

    score_col1, score_col2 = st.columns(2)

    with score_col1:
        st.metric(
            "Event Screening Score",
            f"{event_score}/100",
        )

    with score_col2:
        st.metric(
            "Investigation Priority",
            event_priority,
        )

    if event_priority == "Critical":

        st.error(
            "Critical investigation priority. "
            "Immediate environmental assessment and "
            "confirmatory sampling should be considered."
        )

    elif event_priority == "High":

        st.warning(
            "High investigation priority. "
            "Prioritise site inspection and confirmatory sampling."
        )

    elif event_priority == "Medium":

        st.info(
            "Medium investigation priority. "
            "Further assessment is recommended."
        )

    else:

        st.success(
            "Low screening priority based on the information entered. "
            "Continue routine monitoring."
        )

    # --------------------------------------------------------
    # Petroleum analytical requirements
    # --------------------------------------------------------

    st.markdown("### 6. Recommended Petroleum Analyses")

    petroleum_indicators = [
        "TPH",
        "BTEX",
        "PAHs",
        "Oil and grease",
    ]

    for indicator in petroleum_indicators:
        st.markdown(
            f"- {indicator}"
        )

    st.caption(
        "Additional parameters should be selected according to "
        "the petroleum product involved, site conditions and "
        "the applicable environmental monitoring requirements."
    )

    # --------------------------------------------------------
    # Decision support
    # --------------------------------------------------------

    st.markdown("### 7. Decision-Support Actions")

    actions = []

    if event_priority in ["Critical", "High"]:

        actions.extend(
            [
                "Initiate or prioritise environmental site inspection.",
                "Collect confirmatory upstream and downstream water samples.",
                "Analyse samples for petroleum hydrocarbons and supporting water-quality parameters.",
                "Document the event location, pathway and potentially affected receptors.",
                "Review available spill, maintenance, pipeline and facility records.",
            ]
        )

    elif event_priority == "Medium":

        actions.extend(
            [
                "Schedule targeted environmental inspection.",
                "Repeat water-quality measurements under comparable conditions.",
                "Consider petroleum hydrocarbon laboratory analysis.",
                "Review nearby petroleum infrastructure and drainage pathways.",
            ]
        )

    else:

        actions.extend(
            [
                "Continue routine monitoring.",
                "Maintain event documentation.",
                "Escalate investigation if water-quality conditions deteriorate.",
            ]
        )

    for action in actions:

        st.markdown(
            f"- {action}"
        )

    # --------------------------------------------------------
    # Evidence availability
    # --------------------------------------------------------

    st.markdown("### 8. Petroleum Evidence Availability")

    direct_columns = [
        "tph",
        "total_petroleum_hydrocarbons",
        "btex",
        "benzene",
        "toluene",
        "ethylbenzene",
        "xylene",
        "pah",
        "pahs",
        "oil_and_grease",
        "oil_grease",
    ]

    available_direct = [
        column
        for column in direct_columns
        if column in working.columns
    ]

    if available_direct:

        st.success(
            "Direct petroleum-related measurements are available "
            "in the current dataset."
        )

        st.write(
            "Available fields:",
            ", ".join(available_direct)
        )

    else:

        st.warning(
            "The current Upper Athi dataset does not contain direct "
            "petroleum hydrocarbon measurements."
        )

        st.info(
            "The event score is therefore an investigation-screening "
            "score and must not be interpreted as proof of petroleum "
            "contamination."
        )

    # --------------------------------------------------------
    # Final DSS interpretation
    # --------------------------------------------------------

    st.markdown("### 9. DSS Interpretation")

    st.info(
        f"The system assigns a {event_priority.lower()} investigation "
        f"priority based on the reported event characteristics and "
        f"the current environmental screening condition at "
        f"{selected_station}. This result identifies the need for "
        "investigation; it does not establish petroleum contamination "
        "or causality."
    )

    st.caption(
        "The event score is a project decision-support index. "
        "It should be calibrated against field experience, "
        "regulatory requirements and additional environmental data "
        "before operational deployment."
    )

def show_station_comparison(data):
    """Station Comparison and Petroleum Source Screening."""

    st.markdown(
        "## Station Comparison & Source Screening"
    )

    st.caption(
        "Spatial comparison of Upper Athi River monitoring stations "
        "for environmental deterioration and petroleum-related investigation screening."
    )

    working = data.copy()

    # --------------------------------------------------------
    # Prepare numeric fields
    # --------------------------------------------------------

    numeric_columns = [
        "ph",
        "dissolved_oxygen",
        "turbidity",
        "temperature",
        "conductivity",
        "tds",
        "tss",
        "total_nitrogen",
        "total_phosphorus",
        "chla",
        "iron",
        "manganese",
        "lead",
        "chromium",
        "copper",
        "zinc",
        "chloride",
    ]

    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(
                working[column],
                errors="coerce"
            )

    # Keep negative chlorophyll values out of derived analysis.
    if "chla" in working.columns:
        working["chla_clean"] = working["chla"].clip(lower=0)

    # --------------------------------------------------------
    # Identify station columns
    # --------------------------------------------------------

    station_column = (
        "station"
        if "station" in working.columns
        else None
    )

    if station_column is None:
        st.error(
            "The dataset does not contain a station identifier."
        )
        return

    name_column = (
        "station_name"
        if "station_name" in working.columns
        else station_column
    )

    station_table = (
        working[
            [station_column, name_column]
        ]
        .drop_duplicates()
        .sort_values(station_column)
    )

    station_labels = {}

    for _, row in station_table.iterrows():
        station_id = str(row[station_column])
        station_name = str(row[name_column])

        station_labels[
            f"{station_id} â€” {station_name}"
        ] = station_id

    if len(station_labels) < 2:
        st.warning(
            "At least two monitoring stations are required for comparison."
        )
        return

    # --------------------------------------------------------
    # Station selection
    # --------------------------------------------------------

    selected_labels = st.multiselect(
        "Select stations to compare",
        options=list(station_labels.keys()),
        default=list(station_labels.keys())[:2],
        max_selections=8,
    )

    if len(selected_labels) < 2:
        st.info(
            "Select at least two stations to perform a comparison."
        )
        return

    selected_stations = [
        station_labels[label]
        for label in selected_labels
    ]

    comparison = working[
        working[station_column].astype(str).isin(
            selected_stations
        )
    ].copy()

    # --------------------------------------------------------
    # Calculate WQI where necessary
    # --------------------------------------------------------

    wqi_column = get_wqi_column(comparison)

    if wqi_column is None:
        comparison["wqi"] = comparison.apply(
            calculate_screening_wqi,
            axis=1
        )
        wqi_column = "wqi"

    comparison[wqi_column] = pd.to_numeric(
        comparison[wqi_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Station summary
    # --------------------------------------------------------

    summary = (
        comparison
        .groupby(
            [station_column, name_column],
            dropna=False
        )
        .agg(
            mean_wqi=(wqi_column, "mean"),
            minimum_wqi=(wqi_column, "min"),
            maximum_wqi=(wqi_column, "max"),
            observations=(wqi_column, "count"),
        )
        .reset_index()
    )

    summary["WQI Category"] = summary["mean_wqi"].apply(
        classify_wqi
    )

    summary["Screening Risk"] = summary["mean_wqi"].apply(
        classify_risk
    )

    # --------------------------------------------------------
    # Exceedance screening
    # --------------------------------------------------------

    def count_exceedances(frame):
        count = 0

        if "ph" in frame.columns:
            count += int(
                ((frame["ph"] < 6.5) | (frame["ph"] > 8.5))
                .fillna(False)
                .sum()
            )

        if "turbidity" in frame.columns:
            count += int(
                (frame["turbidity"] > 5)
                .fillna(False)
                .sum()
            )

        if "dissolved_oxygen" in frame.columns:
            count += int(
                (frame["dissolved_oxygen"] < 5)
                .fillna(False)
                .sum()
            )

        if "temperature" in frame.columns:
            count += int(
                (
                    (frame["temperature"] < 15)
                    | (frame["temperature"] > 30)
                )
                .fillna(False)
                .sum()
            )

        return count

    exceedance_values = []

    for station_id in summary[station_column]:
        station_data = comparison[
            comparison[station_column] == station_id
        ]

        exceedance_values.append(
            count_exceedances(station_data)
        )

    summary["Screening Exceedances"] = exceedance_values

    summary["Investigation Priority"] = (
        summary["Screening Exceedances"] * 10
        + (70 - summary["mean_wqi"]).clip(lower=0)
    )

    summary["Priority"] = summary[
        "Investigation Priority"
    ].apply(
        lambda value:
            "High" if value >= 40
            else "Medium" if value >= 15
            else "Low"
    )

    # --------------------------------------------------------
    # Display station summary
    # --------------------------------------------------------

    st.markdown("### 1. Station Comparison")

    display_summary = summary.copy()

    display_summary["Mean WQI"] = (
        display_summary["mean_wqi"].round(2)
    )

    display_summary["Minimum WQI"] = (
        display_summary["minimum_wqi"].round(2)
    )

    display_summary["Maximum WQI"] = (
        display_summary["maximum_wqi"].round(2)
    )

    display_summary = display_summary[
        [
            station_column,
            name_column,
            "Mean WQI",
            "Minimum WQI",
            "Maximum WQI",
            "WQI Category",
            "Screening Risk",
            "Screening Exceedances",
            "Priority",
        ]
    ]

    st.dataframe(
        display_summary,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # Parameter comparison
    # --------------------------------------------------------

    st.markdown("### 2. Parameter-Level Comparison")

    comparison_parameters = [
        "ph",
        "dissolved_oxygen",
        "turbidity",
        "temperature",
        "conductivity",
        "tds",
        "tss",
        "total_nitrogen",
        "total_phosphorus",
        "iron",
        "manganese",
        "lead",
        "chromium",
        "copper",
        "zinc",
    ]

    available_parameters = [
        column
        for column in comparison_parameters
        if column in comparison.columns
    ]

    parameter_summary = (
        comparison
        .groupby(
            [station_column, name_column],
            dropna=False
        )[available_parameters]
        .mean()
        .reset_index()
    )

    st.dataframe(
        parameter_summary.round(3),
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # Select upstream / reference and comparison station
    # --------------------------------------------------------

    st.markdown("### 3. Upstream / Downstream Screening")

    st.caption(
        "This comparison identifies changes between two selected "
        "stations. It does not establish causality."
    )

    station_options = list(station_labels.keys())

    upstream_label = st.selectbox(
        "Reference / upstream station",
        station_options,
        index=0,
        key="comparison_upstream_station",
    )

    downstream_default = (
        1 if len(station_options) > 1 else 0
    )

    downstream_label = st.selectbox(
        "Comparison / downstream station",
        station_options,
        index=downstream_default,
        key="comparison_downstream_station",
    )

    upstream_station = station_labels[upstream_label]
    downstream_station = station_labels[downstream_label]

    if upstream_station == downstream_station:
        st.warning(
            "Select two different stations for upstream/downstream screening."
        )
        return

    upstream_data = working[
        working[station_column].astype(str)
        == str(upstream_station)
    ]

    downstream_data = working[
        working[station_column].astype(str)
        == str(downstream_station)
    ]

    # --------------------------------------------------------
    # Calculate mean differences
    # --------------------------------------------------------

    source_parameters = [
        "ph",
        "dissolved_oxygen",
        "turbidity",
        "temperature",
        "conductivity",
        "tds",
        "tss",
        "total_nitrogen",
        "total_phosphorus",
        "iron",
        "manganese",
        "lead",
        "chromium",
        "copper",
        "zinc",
    ]

    difference_rows = []

    for parameter in source_parameters:

        if parameter not in working.columns:
            continue

        upstream_value = upstream_data[
            parameter
        ].mean()

        downstream_value = downstream_data[
            parameter
        ].mean()

        if pd.isna(upstream_value) or pd.isna(downstream_value):
            continue

        difference = downstream_value - upstream_value

        if abs(upstream_value) > 1e-9:
            percentage_change = (
                difference / abs(upstream_value)
            ) * 100
        else:
            percentage_change = np.nan

        difference_rows.append(
            {
                "Parameter": parameter,
                "Reference Mean": upstream_value,
                "Comparison Mean": downstream_value,
                "Difference": difference,
                "Percentage Change": percentage_change,
            }
        )

    difference_df = pd.DataFrame(difference_rows)

    if not difference_df.empty:

        st.dataframe(
            difference_df.round(3),
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # Deterioration indicators
    # --------------------------------------------------------

    st.markdown("### 4. Environmental Deterioration Screening")

    deterioration_rules = {
        "turbidity": "increase",
        "tss": "increase",
        "conductivity": "increase",
        "tds": "increase",
        "iron": "increase",
        "manganese": "increase",
        "lead": "increase",
        "chromium": "increase",
        "copper": "increase",
        "zinc": "increase",
        "total_nitrogen": "increase",
        "total_phosphorus": "increase",
        "dissolved_oxygen": "decrease",
    }

    deterioration_rows = []

    for parameter, direction in deterioration_rules.items():

        if parameter not in working.columns:
            continue

        upstream_value = upstream_data[
            parameter
        ].mean()

        downstream_value = downstream_data[
            parameter
        ].mean()

        if pd.isna(upstream_value) or pd.isna(downstream_value):
            continue

        if direction == "increase":
            changed = downstream_value > upstream_value
        else:
            changed = downstream_value < upstream_value

        if changed:
            deterioration_rows.append(
                {
                    "Parameter": parameter,
                    "Reference": upstream_value,
                    "Comparison": downstream_value,
                    "Change": downstream_value - upstream_value,
                    "Pattern": (
                        "Potential deterioration"
                        if changed
                        else "No deterioration"
                    ),
                }
            )

    if deterioration_rows:

        deterioration_df = pd.DataFrame(
            deterioration_rows
        )

        st.dataframe(
            deterioration_df.round(3),
            width="stretch",
            hide_index=True,
        )

        deterioration_count = len(
            deterioration_rows
        )

        if deterioration_count >= 5:
            st.warning(
                f"{deterioration_count} parameters show a "
                "potential downstream deterioration pattern. "
                "This warrants further environmental investigation."
            )

        elif deterioration_count >= 2:
            st.info(
                f"{deterioration_count} parameters show a "
                "potential deterioration pattern."
            )

        else:
            st.info(
                "A limited deterioration pattern was identified."
            )

    else:
        st.success(
            "No deterioration pattern was identified using "
            "the selected screening parameters."
        )

    # --------------------------------------------------------
    # Petroleum-specific evidence
    # --------------------------------------------------------

    st.markdown("### 5. Petroleum Source-Screening")

    petroleum_indicators = [
        "tph",
        "tp_h",
        "total_petroleum_hydrocarbons",
        "btex",
        "benzene",
        "toluene",
        "ethylbenzene",
        "xylene",
        "pah",
        "pahs",
        "oil_and_grease",
        "oil_grease",
    ]

    available_petroleum = [
        column
        for column in petroleum_indicators
        if column in working.columns
    ]

    if available_petroleum:

        st.success(
            "Direct petroleum-related indicator fields are "
            "available in the dataset."
        )

        petroleum_summary = []

        for parameter in available_petroleum:

            upstream_value = upstream_data[
                parameter
            ].mean()

            downstream_value = downstream_data[
                parameter
            ].mean()

            petroleum_summary.append(
                {
                    "Indicator": parameter,
                    "Reference": upstream_value,
                    "Comparison": downstream_value,
                    "Difference": (
                        downstream_value
                        - upstream_value
                    )
                }
            )

        st.dataframe(
            pd.DataFrame(
                petroleum_summary
            ).round(4),
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No direct petroleum hydrocarbon indicators "
            "(such as TPH, BTEX or PAHs) are present in the "
            "current Upper Athi dataset."
        )

        st.warning(
            "Observed water-quality deterioration cannot by "
            "itself be attributed to petroleum contamination."
        )

    # --------------------------------------------------------
    # Investigation recommendation
    # --------------------------------------------------------

    st.markdown("### 6. DSS Investigation Recommendation")

    wqi_upstream = upstream_data[wqi_column].mean()
    wqi_downstream = downstream_data[wqi_column].mean()

    if (
        not pd.isna(wqi_upstream)
        and not pd.isna(wqi_downstream)
    ):

        wqi_change = wqi_downstream - wqi_upstream

        st.metric(
            "WQI Change",
            f"{wqi_change:+.2f}",
            help=(
                "Comparison station WQI minus reference station WQI."
            ),
        )

        if wqi_change < -10:

            st.error(
                "Significant WQI deterioration detected between "
                "the selected stations. Prioritise confirmatory "
                "sampling and source investigation."
            )

        elif wqi_change < 0:

            st.warning(
                "WQI deterioration is present. Further "
                "investigation is recommended."
            )

        else:

            st.success(
                "The comparison station does not show lower WQI "
                "than the selected reference station."
            )

    # --------------------------------------------------------
    # Recommended petroleum investigation
    # --------------------------------------------------------

    st.markdown("### 7. Recommended Petroleum Investigation")

    recommendations = [
        "Verify the upstream and downstream station locations and sampling dates.",
        "Repeat sampling during comparable hydrological conditions.",
        "Collect TPH, BTEX, PAH and oil-and-grease measurements where petroleum exposure is plausible.",
        "Review nearby petroleum infrastructure, transport routes, storage facilities and spill/event records.",
        "Use upstream/downstream and, where possible, upstream-to-source-to-downstream sampling designs.",
        "Combine laboratory results with site inspection before assigning a petroleum source.",
    ]

    for recommendation in recommendations:
        st.markdown(
            f"- {recommendation}"
        )

    # --------------------------------------------------------
    # Scientific interpretation
    # --------------------------------------------------------

    st.markdown("### 8. Scientific Interpretation")

    st.info(
        "This module is a screening and decision-support tool. "
        "A downstream increase in conventional water-quality "
        "parameters indicates a change in environmental condition, "
        "but does not identify petroleum as the source. Petroleum "
        "attribution requires direct hydrocarbon measurements, "
        "source investigation, appropriate spatial sampling and "
        "supporting environmental evidence."
    )

    st.caption(
        "Project screening thresholds are used for DSS screening "
        "and should not be interpreted as verified regulatory "
        "limits unless independently confirmed against the "
        "applicable Kenyan regulatory framework."
    )

def show_station_investigation(data):
    """Station-level water-quality and petroleum pathway screening."""

    st.title("Station Investigation & Source Screening")
    st.caption(
        "Upper Athi River Catchment â€¢ Station-level environmental investigation"
    )

    if data is None or data.empty:
        st.warning("No monitoring data are available.")
        return

    working = data.copy()

    # ------------------------------------------------------------
    # Standardise pH column
    # ------------------------------------------------------------
    if "ph" not in working.columns and "pH" in working.columns:
        working["ph"] = working["pH"]

    required = [
        "station",
        "station_name",
        "year",
        "ph",
        "dissolved_oxygen",
        "turbidity",
        "temperature",
        "conductivity",
        "tds",
        "tss",
        "total_nitrogen",
        "total_phosphorus",
        "chla",
    ]

    missing = [column for column in required if column not in working.columns]

    if missing:
        st.error(
            "Station investigation cannot run because these columns are missing: "
            + ", ".join(missing)
        )
        return

    numeric_columns = [
        "year",
        "ph",
        "dissolved_oxygen",
        "turbidity",
        "temperature",
        "conductivity",
        "tds",
        "tss",
        "total_nitrogen",
        "total_phosphorus",
        "chla",
    ]

    for column in numeric_columns:
        working[column] = pd.to_numeric(
            working[column], errors="coerce"
        )

    # ------------------------------------------------------------
    # WQI / prediction columns
    # ------------------------------------------------------------
    wqi_column = get_wqi_column(working)
    prediction_column = get_prediction_column(working)

    if wqi_column is None:
        working["investigation_wqi"] = working.apply(
            calculate_screening_wqi,
            axis=1
        )
        wqi_column = "investigation_wqi"

    working[wqi_column] = pd.to_numeric(
        working[wqi_column], errors="coerce"
    )

    if prediction_column is not None:
        working[prediction_column] = pd.to_numeric(
            working[prediction_column], errors="coerce"
        )

    # ------------------------------------------------------------
    # Station selector
    # ------------------------------------------------------------
    station_options = (
        working[["station", "station_name"]]
        .drop_duplicates()
        .sort_values(["station", "station_name"])
    )

    station_labels = {
        row["station"]:
            f'{row["station"]} â€” {row["station_name"]}'
        for _, row in station_options.iterrows()
    }

    selected_station = st.selectbox(
        "Select monitoring station",
        station_options["station"].tolist(),
        format_func=lambda x: station_labels.get(x, str(x)),
    )

    station_data = working[
        working["station"] == selected_station
    ].copy()

    if station_data.empty:
        st.warning("No observations are available for the selected station.")
        return

    station_data = station_data.sort_values("year")

    # ------------------------------------------------------------
    # Station summary
    # ------------------------------------------------------------
    mean_wqi = station_data[wqi_column].mean()

    if pd.isna(mean_wqi):
        category = "Unknown"
        risk = "Unknown"
    else:
        category = classify_wqi(mean_wqi)
        risk = classify_risk(mean_wqi)

    # Parameter flags
    station_data["flag_ph"] = (
        (station_data["ph"] < 6.5)
        | (station_data["ph"] > 8.5)
    )

    station_data["flag_turbidity"] = (
        station_data["turbidity"] > 5
    )

    station_data["flag_do"] = (
        station_data["dissolved_oxygen"] < 5
    )

    station_data["flag_temperature"] = (
        (station_data["temperature"] < 15)
        | (station_data["temperature"] > 30)
    )

    flag_columns = [
        "flag_ph",
        "flag_turbidity",
        "flag_do",
        "flag_temperature",
    ]

    station_data["parameter_exceedances"] = (
        station_data[flag_columns]
        .sum(axis=1)
    )

    total_exceedances = int(
        station_data["parameter_exceedances"].sum()
    )

    # ------------------------------------------------------------
    # Priority score
    # ------------------------------------------------------------
    priority_score = (
        total_exceedances * 10
        + max(0, 70 - mean_wqi)
        if pd.notna(mean_wqi)
        else total_exceedances * 10
    )

    if priority_score >= 40:
        priority = "High"
    elif priority_score >= 15:
        priority = "Medium"
    else:
        priority = "Low"

    # ------------------------------------------------------------
    # Header metrics
    # ------------------------------------------------------------
    st.subheader("Station Environmental Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Mean WQI",
            f"{mean_wqi:.1f}" if pd.notna(mean_wqi) else "N/A"
        )

    with c2:
        st.metric("WQI Category", category)

    with c3:
        st.metric("Screening Risk", risk)

    with c4:
        st.metric("Investigation Priority", priority)

    if priority == "High":
        st.error(
            "High-priority station: multiple indicators or degraded WQI "
            "conditions justify further environmental investigation."
        )
    elif priority == "Medium":
        st.warning(
            "Medium-priority station: continued monitoring and targeted "
            "source investigation are recommended."
        )
    else:
        st.success(
            "Low-priority station under the current screening framework."
        )

    # ------------------------------------------------------------
    # Observation history
    # ------------------------------------------------------------
    st.subheader("Station Observation History")

    history_columns = [
        "year",
        wqi_column,
        "ph",
        "dissolved_oxygen",
        "turbidity",
        "temperature",
        "conductivity",
        "tds",
        "tss",
        "total_nitrogen",
        "total_phosphorus",
        "chla",
    ]

    history_columns = [
        column for column in history_columns
        if column in station_data.columns
    ]

    history = station_data[history_columns].copy()

    rename_map = {
        wqi_column: "WQI",
        "ph": "pH",
        "dissolved_oxygen": "DO",
        "turbidity": "Turbidity",
        "temperature": "Temperature",
        "conductivity": "Conductivity",
        "tds": "TDS",
        "tss": "TSS",
        "total_nitrogen": "Total Nitrogen",
        "total_phosphorus": "Total Phosphorus",
        "chla": "Chlorophyll-a",
    }

    history = history.rename(columns=rename_map)

    st.dataframe(
        history.round(3),
        width="stretch",
        hide_index=True,
    )

    # ------------------------------------------------------------
    # ML prediction comparison
    # ------------------------------------------------------------
    if prediction_column is not None:

        st.subheader("Machine-Learning Prediction Assessment")

        prediction_data = station_data[
            [wqi_column, prediction_column]
        ].dropna()

        if not prediction_data.empty:

            prediction_data = prediction_data.copy()

            prediction_data["Prediction Error"] = (
                prediction_data[prediction_column]
                - prediction_data[wqi_column]
            )

            prediction_data["Absolute Error"] = (
                prediction_data["Prediction Error"]
                .abs()
            )

            mean_absolute_error = (
                prediction_data["Absolute Error"].mean()
            )

            pc1, pc2, pc3 = st.columns(3)

            with pc1:
                st.metric(
                    "Mean Absolute Error",
                    f"{mean_absolute_error:.2f}"
                )

            with pc2:
                st.metric(
                    "Measured WQI",
                    f"{prediction_data[wqi_column].mean():.2f}"
                )

            with pc3:
                st.metric(
                    "Predicted WQI",
                    f"{prediction_data[prediction_column].mean():.2f}"
                )

            comparison = prediction_data.rename(
                columns={
                    wqi_column: "Measured WQI",
                    prediction_column: "Predicted WQI",
                }
            )

            st.dataframe(
                comparison.round(3),
                width="stretch",
                hide_index=True,
            )

    # ------------------------------------------------------------
    # Parameter screening
    # ------------------------------------------------------------
    st.subheader("Parameter-Level Screening")

    latest = station_data.iloc[-1]

    parameter_rows = [
        {
            "Parameter": "pH",
            "Observed": latest["ph"],
            "Screening criterion": "6.5â€“8.5",
            "Status": (
                "Flagged"
                if not (6.5 <= latest["ph"] <= 8.5)
                else "Within range"
            ),
        },
        {
            "Parameter": "Turbidity",
            "Observed": latest["turbidity"],
            "Screening criterion": "= 5 NTU",
            "Status": (
                "Flagged"
                if latest["turbidity"] > 5
                else "Within range"
            ),
        },
        {
            "Parameter": "Dissolved Oxygen",
            "Observed": latest["dissolved_oxygen"],
            "Screening criterion": "= 5 mg/L",
            "Status": (
                "Flagged"
                if latest["dissolved_oxygen"] < 5
                else "Within range"
            ),
        },
        {
            "Parameter": "Temperature",
            "Observed": latest["temperature"],
            "Screening criterion": "15â€“30 °C",
            "Status": (
                "Flagged"
                if not (15 <= latest["temperature"] <= 30)
                else "Within range"
            ),
        },
    ]

    parameter_df = pd.DataFrame(parameter_rows)

    st.dataframe(
        parameter_df.round(3),
        width="stretch",
        hide_index=True,
    )

    # ------------------------------------------------------------
    # Supporting water-quality indicators
    # ------------------------------------------------------------
    st.subheader("Supporting Water-Quality Indicators")

    supporting = {
        "TSS": latest["tss"],
        "TDS": latest["tds"],
        "Conductivity": latest["conductivity"],
        "Total Nitrogen": latest["total_nitrogen"],
        "Total Phosphorus": latest["total_phosphorus"],
        "Chlorophyll-a": max(0, latest["chla"])
            if pd.notna(latest["chla"])
            else None,
    }

    support_cols = st.columns(3)

    items = list(supporting.items())

    for index, (name, value) in enumerate(items):
        with support_cols[index % 3]:
            if value is None or pd.isna(value):
                st.metric(name, "N/A")
            else:
                st.metric(name, f"{value:.3f}")

    # ------------------------------------------------------------
    # Petroleum pathway assessment
    # ------------------------------------------------------------
    st.subheader("Petroleum Pathway Screening")

    st.markdown(
        """
        The following assessment identifies petroleum-related pathways that
        should be considered during field investigation. It is a screening
        exercise and does **not** establish that petroleum caused the observed
        water-quality condition.
        """
    )

    if risk == "High":
        pathway_priority = "High"
    elif risk == "Medium":
        pathway_priority = "Moderate"
    else:
        pathway_priority = "Routine"

    pathway_table = pd.DataFrame({
        "Potential pathway": [
            "Spill / accidental release",
            "Storage and fuel handling",
            "Pipeline transportation",
            "Industrial discharge",
            "Wastewater influence",
            "Urban/catchment runoff",
        ],
        "Investigation priority": [
            pathway_priority,
            pathway_priority,
            pathway_priority,
            "Investigate",
            "Consider",
            "Consider",
        ],
    })

    st.dataframe(
        pathway_table,
        width="stretch",
        hide_index=True,
    )

    # ------------------------------------------------------------
    # Petroleum-specific evidence gap
    # ------------------------------------------------------------
    st.subheader("Petroleum-Specific Evidence Check")

    direct_petroleum = [
        column for column in [
            "TPH",
            "tph",
            "BTEX",
            "btex",
            "PAHs",
            "pahs",
            "oil_and_grease",
        ]
        if column in station_data.columns
    ]

    if direct_petroleum:
        st.success(
            "Direct petroleum-related indicators are available in the dataset: "
            + ", ".join(direct_petroleum)
        )
    else:
        st.warning(
            "No direct petroleum hydrocarbon measurements are available for "
            "this station in the current dataset."
        )

        st.markdown(
            """
            Recommended confirmatory indicators where petroleum exposure is
            suspected:

            - Total Petroleum Hydrocarbons (TPH)
            - BTEX compounds
            - Polycyclic Aromatic Hydrocarbons (PAHs)
            - Oil and Grease
            - Appropriate supporting physicochemical parameters
            """
        )

    # ------------------------------------------------------------
    # Recommended actions
    # ------------------------------------------------------------
    st.subheader("Recommended Investigation Actions")

    if priority == "High":
        recommendations = [
            "Prioritise the station for field verification.",
            "Review nearby petroleum infrastructure and drainage pathways.",
            "Check available spill, leakage, maintenance and operational records.",
            "Collect confirmatory samples using an appropriate sampling protocol.",
            "Include petroleum-specific laboratory analysis where justified.",
            "Compare upstream and downstream locations before assigning a source.",
        ]

    elif priority == "Medium":
        recommendations = [
            "Increase monitoring attention at the station.",
            "Investigate repeated parameter exceedances.",
            "Review nearby industrial, wastewater and petroleum-related activities.",
            "Consider targeted petroleum-specific laboratory analysis.",
            "Continue comparing observations across years and nearby stations.",
        ]

    else:
        recommendations = [
            "Continue routine monitoring.",
            "Track future WQI and parameter trends.",
            "Maintain awareness of nearby petroleum and industrial activities.",
            "Use event-triggered sampling after unusual releases or discharges.",
        ]

    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")

    # ------------------------------------------------------------
    # Final interpretation
    # ------------------------------------------------------------
    st.subheader("Environmental Interpretation")

    st.info(
        f"""
        **Station:** {selected_station} â€” {station_labels.get(selected_station, "")}

        **Current screening interpretation:** This station has a mean WQI of
        {mean_wqi:.2f} and is classified as **{category}**, with a screening
        risk of **{risk}**.

        The station should therefore be prioritised according to the observed
        water-quality condition and parameter exceedances. Where petroleum
        activities are present in the surrounding area, petroleum-related
        pathways should be investigated alongside other potential sources.

        **Important:** WQI deterioration alone is not evidence of petroleum
        contamination. Petroleum causation requires appropriate source
        investigation and petroleum-specific analytical evidence.
        """
    )


def show_petroleum_dss(data):
    """
    Petroleum-focused Surface Water Quality Decision Support System.

    This module provides screening-level environmental decision support
    for petroleum exploration, transportation, storage, processing and
    associated industrial activities.

    Important:
    The current Upper Athi dataset does not contain direct petroleum
    hydrocarbon indicators such as TPH, BTEX, PAHs or oil-and-grease.
    Therefore, the system identifies water-quality deterioration and
    petroleum-relevant screening priorities but does not independently
    confirm petroleum contamination or establish causality.
    """

    st.title("Petroleum Environmental Decision Support")
    st.caption(
        "Upper Athi River Catchment â€¢ Petroleum-focused surface-water "
        "quality screening and decision support"
    )

    if data is None or data.empty:
        st.warning("No monitoring data are available for petroleum DSS analysis.")
        return

    st.markdown(
        """
        This module integrates monitored water-quality parameters, Water
        Quality Index (WQI), machine-learning predictions and screening
        thresholds to support environmental decision-making around
        petroleum-related activities.

        **The DSS is a screening and prioritisation tool. It does not replace
        laboratory analysis, field inspection, engineering judgement or
        regulatory assessment.**
        """
    )

    # ------------------------------------------------------------
    # 1. Ensure required columns exist
    # ------------------------------------------------------------
    working = data.copy()

    required_columns = [
        "station",
        "station_name",
        "year",
        "ph",
        "dissolved_oxygen",
        "turbidity",
        "temperature",
        "conductivity",
        "tds",
        "tss",
        "total_nitrogen",
        "total_phosphorus",
        "chla",
    ]

    # Support datasets where pH is stored as pH rather than ph.
    if "ph" not in working.columns and "pH" in working.columns:
        working["ph"] = working["pH"]

    missing = [c for c in required_columns if c not in working.columns]

    if missing:
        st.error(
            "The petroleum DSS cannot run because required monitoring "
            f"columns are missing: {', '.join(missing)}"
        )
        return

    # Numeric conversion for analysis.
    numeric_columns = [
        "ph",
        "dissolved_oxygen",
        "turbidity",
        "temperature",
        "conductivity",
        "tds",
        "tss",
        "total_nitrogen",
        "total_phosphorus",
        "chla",
    ]

    for column in numeric_columns:
        working[column] = pd.to_numeric(
            working[column], errors="coerce"
        )

    # Clean chlorophyll-a only for derived DSS calculations.
    # Raw source data remain unchanged.
    working["chla_clean"] = working["chla"].clip(lower=0)

    # ------------------------------------------------------------
    # 2. Determine WQI and risk columns
    # ------------------------------------------------------------
    wqi_column = get_wqi_column(working)
    prediction_column = get_prediction_column(working)
    category_column = get_category_column(working)

    if wqi_column is None:
        working["dss_wqi"] = working.apply(
            lambda row: calculate_screening_wqi(row),
            axis=1
        )
        wqi_column = "dss_wqi"

    working[wqi_column] = pd.to_numeric(
        working[wqi_column], errors="coerce"
    )

    if prediction_column is not None:
        working[prediction_column] = pd.to_numeric(
            working[prediction_column], errors="coerce"
        )

    # Create a consistent screening risk directly from WQI.
    working["dss_risk"] = working[wqi_column].apply(classify_risk)

    # ------------------------------------------------------------
    # 3. Parameter-level screening
    # ------------------------------------------------------------
    thresholds = {
        "ph": ("pH", 6.5, 8.5, "inside"),
        "turbidity": ("Turbidity", 5.0, None, "maximum"),
        "dissolved_oxygen": ("Dissolved Oxygen", 5.0, None, "minimum"),
        "temperature": ("Temperature", 15.0, 30.0, "inside"),
    }

    def parameter_exceedance_count(row):
        count = 0

        ph_value = row.get("ph")
        turbidity_value = row.get("turbidity")
        do_value = row.get("dissolved_oxygen")
        temp_value = row.get("temperature")

        if pd.notna(ph_value) and not (6.5 <= ph_value <= 8.5):
            count += 1

        if pd.notna(turbidity_value) and turbidity_value > 5:
            count += 1

        if pd.notna(do_value) and do_value < 5:
            count += 1

        if pd.notna(temp_value) and not (15 <= temp_value <= 30):
            count += 1

        return count

    working["parameter_exceedances"] = working.apply(
        parameter_exceedance_count,
        axis=1
    )

    working["parameter_screening"] = working["parameter_exceedances"].apply(
        lambda x: (
            "High Priority" if x >= 3
            else "Moderate Priority" if x >= 1
            else "No Flag"
        )
    )

    # ------------------------------------------------------------
    # 4. Overall DSS status
    # ------------------------------------------------------------
    mean_wqi = working[wqi_column].mean()

    if pd.isna(mean_wqi):
        overall_category = "Unknown"
        overall_risk = "Unknown"
    else:
        overall_category = classify_wqi(mean_wqi)
        overall_risk = classify_risk(mean_wqi)

    high_risk_count = int((working["dss_risk"] == "High").sum())
    medium_risk_count = int((working["dss_risk"] == "Medium").sum())
    low_risk_count = int((working["dss_risk"] == "Low").sum())
    flagged_records = int((working["parameter_exceedances"] > 0).sum())

    # ------------------------------------------------------------
    # 5. Executive DSS summary
    # ------------------------------------------------------------
    st.subheader("Executive Environmental Screening")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Mean WQI",
            f"{mean_wqi:.1f}" if pd.notna(mean_wqi) else "N/A"
        )

    with c2:
        st.metric("Overall WQI Class", overall_category)

    with c3:
        st.metric("Screening Risk", overall_risk)

    with c4:
        st.metric("Flagged Records", flagged_records)

    if overall_risk == "High":
        st.error(
            "HIGH PRIORITY: The monitoring dataset indicates poor water-quality "
            "conditions requiring investigation and confirmatory sampling."
        )
    elif overall_risk == "Medium":
        st.warning(
            "MODERATE PRIORITY: Water-quality conditions require increased "
            "attention, parameter investigation and continued monitoring."
        )
    elif overall_risk == "Low":
        st.success(
            "LOW SCREENING PRIORITY: Current WQI conditions support routine "
            "monitoring, subject to site-specific environmental requirements."
        )
    else:
        st.info("Overall screening status could not be determined.")

    # ------------------------------------------------------------
    # 6. Risk distribution
    # ------------------------------------------------------------
    st.subheader("Screening Risk Distribution")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("High", high_risk_count)

    with r2:
        st.metric("Medium", medium_risk_count)

    with r3:
        st.metric("Low", low_risk_count)

    risk_distribution = (
        working["dss_risk"]
        .value_counts()
        .reindex(["High", "Medium", "Low"], fill_value=0)
    )

    st.bar_chart(risk_distribution)

    # ------------------------------------------------------------
    # 7. Station prioritisation
    # ------------------------------------------------------------
    st.subheader("Station Prioritisation")

    station_summary = (
        working.groupby(["station", "station_name"], dropna=False)
        .agg(
            mean_wqi=(wqi_column, "mean"),
            max_wqi=(wqi_column, "max"),
            parameter_exceedances=("parameter_exceedances", "sum"),
            observations=("station", "size"),
        )
        .reset_index()
    )

    station_summary["screening_risk"] = station_summary["mean_wqi"].apply(
        classify_risk
    )

    station_summary["priority_score"] = (
        station_summary["parameter_exceedances"] * 10
        + (70 - station_summary["mean_wqi"]).clip(lower=0)
    )

    station_summary = station_summary.sort_values(
        ["priority_score", "mean_wqi"],
        ascending=[False, True]
    )

    display_station_summary = station_summary.copy()

    display_station_summary["mean_wqi"] = display_station_summary[
        "mean_wqi"
    ].round(2)

    display_station_summary["max_wqi"] = display_station_summary[
        "max_wqi"
    ].round(2)

    st.dataframe(
        display_station_summary,
        width="stretch",
        hide_index=True,
    )

    # ------------------------------------------------------------
    # 8. Parameter exceedance assessment
    # ------------------------------------------------------------
    st.subheader("Water-Quality Screening Indicators")

    indicator_rows = []

    def add_indicator(label, column, condition, threshold_text):
        values = working[column].dropna()

        if len(values) == 0:
            indicator_rows.append({
                "Indicator": label,
                "Threshold": threshold_text,
                "Flagged observations": 0,
                "Percentage flagged": 0.0,
                "Interpretation": "No data",
            })
            return

        flags = condition(values)
        count = int(flags.sum())
        percentage = float(flags.mean() * 100)

        indicator_rows.append({
            "Indicator": label,
            "Threshold": threshold_text,
            "Flagged observations": count,
            "Percentage flagged": round(percentage, 1),
            "Interpretation": (
                "Priority indicator"
                if count > 0
                else "No screening exceedance"
            ),
        })

    add_indicator(
        "pH",
        "ph",
        lambda x: (x < 6.5) | (x > 8.5),
        "6.5â€“8.5",
    )

    add_indicator(
        "Turbidity",
        "turbidity",
        lambda x: x > 5,
        "= 5 NTU",
    )

    add_indicator(
        "Dissolved Oxygen",
        "dissolved_oxygen",
        lambda x: x < 5,
        "= 5 mg/L",
    )

    add_indicator(
        "Temperature",
        "temperature",
        lambda x: (x < 15) | (x > 30),
        "15â€“30 °C",
    )

    indicator_df = pd.DataFrame(indicator_rows)

    st.dataframe(
        indicator_df,
        width="stretch",
        hide_index=True,
    )

    # ------------------------------------------------------------
    # 9. Petroleum pathway screening
    # ------------------------------------------------------------
    st.subheader("Petroleum-Relevant Pollution Pathway Screening")

    st.markdown(
        """
        Observed water-quality deterioration can arise from several
        interacting sources in a catchment. The DSS therefore screens
        petroleum-relevant pathways without assuming that petroleum is
        automatically the cause.
        """
    )

    pathway_data = pd.DataFrame({
        "Potential pathway": [
            "Spill or accidental release",
            "Fuel storage and handling",
            "Pipeline transportation",
            "Petroleum-related industrial activity",
            "Wastewater / industrial discharge",
            "Urban stormwater and catchment runoff",
        ],
        "Screening relevance": [
            "Investigate after reported or suspected release events.",
            "Check drainage, containment and nearby receiving-water conditions.",
            "Compare upstream/downstream stations and inspect crossing points.",
            "Investigate industrial discharge and operational activities.",
            "Consider as a competing or interacting source.",
            "Consider rainfall, land use and catchment transport processes.",
        ],
    })

    st.dataframe(
        pathway_data,
        width="stretch",
        hide_index=True,
    )

    # ------------------------------------------------------------
    # 10. Petroleum monitoring data gap
    # ------------------------------------------------------------
    st.subheader("Petroleum-Specific Data Coverage")

    petroleum_parameters = {
        "TPH": "Total Petroleum Hydrocarbons",
        "BTEX": "Benzene, Toluene, Ethylbenzene and Xylenes",
        "PAHs": "Polycyclic Aromatic Hydrocarbons",
        "oil_and_grease": "Oil and Grease",
        "BOD": "Biochemical Oxygen Demand",
        "COD": "Chemical Oxygen Demand",
    }

    available_petroleum = [
        key for key in petroleum_parameters
        if key in working.columns
    ]

    missing_petroleum = [
        key for key in petroleum_parameters
        if key not in working.columns
    ]

    if available_petroleum:
        st.success(
            "Petroleum-related indicators currently available: "
            + ", ".join(available_petroleum)
        )
    else:
        st.warning(
            "No direct petroleum hydrocarbon indicators are present in the "
            "current Upper Athi dataset."
        )

    if missing_petroleum:
        st.info(
            "Recommended additions for petroleum-specific monitoring: "
            + ", ".join(missing_petroleum)
        )

    st.markdown(
        """
        **Scientific interpretation:** The current dataset can identify
        degraded water-quality conditions and locations requiring attention,
        but it cannot by itself confirm petroleum contamination. Confirmation
        requires petroleum-specific analytical parameters and appropriate
        upstream/downstream or event-based sampling.
        """
    )

    # ------------------------------------------------------------
    # 11. Decision-support actions
    # ------------------------------------------------------------
    st.subheader("Recommended Decision-Support Actions")

    if overall_risk == "High":
        actions = [
            "Prioritise the highest-risk monitoring stations for field investigation.",
            "Collect confirmatory water samples using an appropriate laboratory sampling protocol.",
            "Include petroleum-specific parameters such as TPH, BTEX, PAHs and oil-and-grease where petroleum exposure is suspected.",
            "Inspect nearby petroleum infrastructure, drainage pathways and potential discharge points.",
            "Review recent spill, leakage, maintenance and operational records.",
            "Increase monitoring frequency while the source investigation is underway.",
        ]
    elif overall_risk == "Medium":
        actions = [
            "Increase monitoring attention at stations with repeated parameter exceedances.",
            "Investigate unusually high turbidity, TSS, low dissolved oxygen and other flagged indicators.",
            "Review nearby industrial, wastewater, transport and petroleum-related activities.",
            "Add petroleum-specific laboratory parameters where a petroleum pathway is plausible.",
            "Compare upstream and downstream observations before assigning a potential source.",
        ]
    else:
        actions = [
            "Maintain routine monitoring at the current sampling locations.",
            "Continue trend analysis to identify deterioration over time.",
            "Maintain inspection of relevant drainage, storage and transportation infrastructure.",
            "Use event-triggered sampling following spills, unusual discharges or major rainfall events.",
        ]

    for action in actions:
        st.markdown(f"- {action}")

    # ------------------------------------------------------------
    # 12. DSS decision logic
    # ------------------------------------------------------------
    st.subheader("DSS Decision Logic")

    decision_table = pd.DataFrame({
        "Screening condition": [
            "Low risk",
            "Medium risk",
            "High risk",
            "Petroleum-specific evidence available",
            "Petroleum-specific evidence unavailable",
        ],
        "Recommended response": [
            "Routine monitoring and trend tracking.",
            "Increased monitoring and source investigation.",
            "Priority investigation and confirmatory sampling.",
            "Use hydrocarbon indicators to strengthen source assessment.",
            "Do not conclude petroleum causation from WQI alone.",
        ],
    })

    st.dataframe(
        decision_table,
        width="stretch",
        hide_index=True,
    )

    # ------------------------------------------------------------
    # 13. Current limitations
    # ------------------------------------------------------------
    with st.expander("Important methodological limitations"):
        st.markdown(
            """
            **1. Dataset size:** The current Upper Athi case study contains
            36 station-year observations from 18 monitoring stations over
            two years.

            **2. Model purpose:** The Random Forest model is an exploratory
            WQI prediction baseline. Its current performance should not be
            interpreted as proof of generalisable future forecasting.

            **3. WQI relationship:** The model predicts WQI using water-quality
            variables that also contribute to WQI calculation. Therefore,
            it approximates the relationship between measured parameters and
            the derived WQI rather than performing a true independent
            time-series forecast.

            **4. Petroleum attribution:** The present dataset does not
            contain direct petroleum hydrocarbon indicators such as TPH,
            BTEX or PAHs. Consequently, the DSS cannot independently prove
            that petroleum activities caused an observed water-quality
            change.

            **5. Regulatory use:** Screening thresholds shown by this
            application are project-level screening criteria unless a
            specific regulatory standard is explicitly identified and
            verified. They should not automatically be interpreted as
            Kenyan legal compliance limits.

            **6. Confirmatory investigation:** Field observations, laboratory
            analysis, infrastructure inspection, source tracing and
            regulatory requirements remain necessary for environmental
            decisions.
            """
        )

    # ------------------------------------------------------------`r`n    # 14. DSS workflow
    # ------------------------------------------------------------
    st.subheader("Petroleum Environmental DSS Workflow")

    st.code(
        "Monitoring Data\n"
        "      ?\n"
        "Data Quality Control & Pre-processing\n"
        "      ?\n"
        "WQI Calculation\n"
        "      ?\n"
        "Random Forest WQI Prediction\n"
        "      ?\n"
        "Risk Screening\n"
        "      ?\n"
        "Station Prioritisation\n"
        "      ?\n"
        "Petroleum Pathway Investigation\n"
        "      ?\n"
        "Confirmatory Sampling & Laboratory Analysis\n"
        "      ?\n"
        "Environmental / Petroleum Engineering Decision\n"
        "      ?\n"
        "Monitoring, Mitigation & Follow-up",
        language="text",
    )

    st.caption(
        "ML-SWQM-DSS: Machine Learning-based Surface Water Quality "
        "Monitoring and Decision Support System"
    )



def show_about():
    """Application information."""

    st.header(
        "??Â About ML-SWQM-DSS"
    )

    st.markdown(
        """
        ## ML-SWQM-DSS

        **Machine Learning-based Surface Water Quality Monitoring
        and Decision Support System**

        ### Case Study

        **Upper Athi River Catchment, Kenya**

        ### Project objective

        The system is designed to demonstrate how machine-learning
        techniques can be integrated with surface-water monitoring
        data to support environmental assessment and early
        decision-making.

        ### Current model

        The current case-study model is a:

        **Random Forest Regression model**

        It predicts:

        **Water Quality Index (WQI)**

        from ten water-quality variables:

        - pH
        - dissolved oxygen
        - turbidity
        - temperature
        - conductivity
        - TDS
        - TSS
        - total nitrogen
        - total phosphorus
        - chlorophyll-a

        ### Dataset

        The Upper Athi case study contains:

        **18 monitoring stations Ã— 2 years = 36 station-year observations**
        observations.**

        ### Important academic limitation

        The current dataset is appropriate for demonstrating the
        DSS architecture and exploratory modelling. However, the
        number of observations is too small to justify describing
        the model as a fully validated operational forecasting
        system.

        Additional temporal observations, direct petroleum
        hydrocarbon measurements, meteorological information,
        hydrological data and verified pollution-event records
        would strengthen future versions.

        ### Petroleum focus

        The application is structured to support future integration
        of petroleum-specific environmental indicators including
        TPH, BTEX, PAHs, oil and grease, spill records and proximity
        to petroleum facilities.

        ### Intended users

        The DSS can support:

        - environmental monitoring teams;
        - petroleum and environmental engineers;
        - water-quality researchers;
        - academic researchers;
        - environmental management teams; and
        - decision-makers requiring rapid screening information.
        """
    )

    st.divider()

    st.caption(
        "ML-SWQM-DSS | Upper Athi River Catchment Case Study"
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

def show_system_status(
    data,
    model,
    model_features,
):
    """Display a compact system status panel."""

    with st.sidebar.expander(
        "System Status",
        expanded=False,
    ):

        st.write(
            "Dataset:",
            "? Loaded" if not data.empty else "? Not Loaded"
        )

        st.write(
            "Random Forest:",
            "? Loaded" if model is not None else "? Not Loaded"
        )

        st.write(
            "Model Features:",
            f"{len(model_features)}",
        )

        st.write(
            "Dataset Rows:",
            len(data),
        )

        st.write(
            "Dataset File:",
            os.path.basename(
                DATA_FILE
            ),
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        (
            model,
            model_features,
            metadata,
        ) = load_athi_model()

    except Exception as error:

        st.error(
            "The Athi machine-learning model could not be loaded."
        )

        st.code(
            str(error)
        )

        st.info(
            "Confirm that these files exist inside the models "
            "folder: athi_regressor.pkl, athi_model_features.pkl "
            "and optionally athi_model_metadata.pkl."
        )

        st.stop()

    # --------------------------------------------------------
    # Validate model features
    # --------------------------------------------------------

    if not model_features:

        st.error(
            "The Athi model feature list is empty."
        )

        st.stop()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    try:

        raw_data = load_athi_data()

    except Exception as error:

        st.error(
            "The Upper Athi dataset could not be loaded."
        )

        st.code(
            str(error)
        )

        st.info(
            f"Expected dataset:\n{DATA_FILE}"
        )

        st.stop()

    # --------------------------------------------------------
    # Prepare dataset
    # --------------------------------------------------------

    try:

        data = prepare_dataset(
            raw_data,
            model,
            model_features,
        )

    except Exception as error:

        st.error(
            "An error occurred while preparing the dataset."
        )

        st.exception(
            error
        )

        st.stop()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    render_header()

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    page = render_sidebar()

    show_system_status(
        data,
        model,
        model_features,
    )

    # --------------------------------------------------------
    # Page routing
    # --------------------------------------------------------

    if page == "Dashboard":

        show_dashboard(
            data
        )

    elif page == "Monitoring Data":

        show_monitoring_data(
            data
        )

    elif page == "ML Predictions":

        show_ml_predictions(
            data,
            model_features,
        )

    elif page == "Risk Analysis":

        show_risk_analysis(
            data
        )

    elif page == "Water Quality Analysis":

        show_water_quality_analysis(
            data
        )

        st.divider()

        with st.expander(
            "Show Correlation Matrix"
        ):

            show_correlation_matrix(
                data
            )

    elif page == "Model Information":

        show_model_information(
            model,
            model_features,
            metadata,
        )

    elif page == "Data Quality":

        show_data_quality(
            data
        )

    elif page == "Petroleum DSS":

        show_petroleum_dss(
            data
        )

    elif page == "Station Investigation":

        show_station_investigation(
            data
        )

    elif page == "Station Comparison":

        show_station_comparison(
            data
        )

    elif page == "Petroleum Event Screening":

        show_petroleum_event_screening(
            data
        )

    elif page == "Future Forecasting":

        show_future_forecasting()



    elif page == "About":

        show_about()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


