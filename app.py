# ML-SWQM-DSS Final Application
"""
ML-SWQM-DSS
Machine Learning-Based Surface Water Quality Monitoring
and Decision Support System

Optimized Streamlit Application
--------------------------------
Core functions:
- Dashboard
- Surface-water monitoring
- ML prediction
- Risk analysis
- Decision support
- Alerts
- Reports
- Data explorer
- Data quality assessment
- Model information

Design principle:
The application separates:
1. Screening WQI
2. ML risk prediction
3. Parameter-limit screening
4. Decision-support recommendations

IMPORTANT:
The WQI implemented here is an APPLICATION SCREENING WQI.
It should not be described as a standardized national/international
WQI unless the methodology has been formally validated.
"""

from __future__ import annotations

import os
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# ATHI RIVER CASE STUDY CONFIGURATION
# ============================================================

ATHI_RIVER = "Athi River"

ATHI_STATIONS = {
    "AR01": {
        "name": "Upper Athi",
        "role": "Upstream/reference station",
    },
    "AR02": {
        "name": "Athi River Town",
        "role": "Urban/industrial influence station",
    },
    "AR03": {
        "name": "Kinanie",
        "role": "Industrial influence station",
    },
    "AR04": {
        "name": "Downstream Athi",
        "role": "Downstream assessment station",
    },
}

ATHI_STATION_IDS = list(ATHI_STATIONS.keys())

ATHI_STATION_NAMES = [
    station["name"]
    for station in ATHI_STATIONS.values()
]
def enforce_athi_river_dataset(df):
    """
    Restrict the ML-SWQM-DSS dataset to the Athi River
    four-station case study.
    """

    if df is None or df.empty:
        return df

    df = df.copy()

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Force the project river
    df["river"] = ATHI_RIVER

    # Keep only the four Athi stations if station_id exists
    if "station_id" in df.columns:
        df = df[df["station_id"].isin(ATHI_STATION_IDS)]

        df["station_name"] = df["station_id"].map(
            {
                station_id: station["name"]
                for station_id, station in ATHI_STATIONS.items()
            }
        )

    return df.reset_index(drop=True)
# ============================================================
# OPTIONAL REPORTLAB
# ============================================================

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB = True

except Exception:
    REPORTLAB = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ML-SWQM-DSS",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_data.csv",
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

REGRESSOR_FILE = os.path.join(
    MODEL_DIR,
    "saved_regressor.pkl",
)

CLASSIFIER_FILE = os.path.join(
    MODEL_DIR,
    "saved_classifier.pkl",
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "scaler.pkl",
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "model_features.pkl",
)

LOGO_FILE = os.path.join(
    BASE_DIR,
    "static",
    "images",
    "logo.png",
)


# ============================================================
# MODEL FEATURES
# ============================================================

DEFAULT_FEATURES = [
    "temperature",
    "dissolved_oxygen",
    "turbidity",
    "chla",
    "ph",
    "conductivity",
    "tds",
    "total_hardness",
    "sodium",
    "potassium",
    "nitrite_nitrogen",
    "ammonium_nitrogen",
    "chloride",
    "sulphates",
    "srp",
    "total_phosphorus",
    "nitrate_nitrogen",
    "total_nitrogen",
    "tss",
]


# ============================================================
# DISPLAY LABELS
# ============================================================

LABELS = {
    "temperature": "Temperature",
    "dissolved_oxygen": "Dissolved Oxygen",
    "turbidity": "Turbidity",
    "chla": "Chlorophyll-a",
    "ph": "pH",
    "conductivity": "Conductivity",
    "tds": "Total Dissolved Solids",
    "total_hardness": "Total Hardness",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "nitrite_nitrogen": "Nitrite-N",
    "ammonium_nitrogen": "Ammonium-N",
    "chloride": "Chloride",
    "sulphates": "Sulphates",
    "srp": "Soluble Reactive Phosphorus",
    "total_phosphorus": "Total Phosphorus",
    "nitrate_nitrogen": "Nitrate-N",
    "total_nitrogen": "Total Nitrogen",
    "tss": "Total Suspended Solids",
}


UNITS = {
    "temperature": "°C",
    "dissolved_oxygen": "mg/L",
    "turbidity": "NTU",
    "chla": "µg/L",
    "ph": "",
    "conductivity": "µS/cm",
    "tds": "mg/L",
    "total_hardness": "mg/L",
    "sodium": "mg/L",
    "potassium": "mg/L",
    "nitrite_nitrogen": "mg/L",
    "ammonium_nitrogen": "mg/L",
    "chloride": "mg/L",
    "sulphates": "mg/L",
    "srp": "mg/L",
    "total_phosphorus": "mg/L",
    "nitrate_nitrogen": "mg/L",
    "total_nitrogen": "mg/L",
    "tss": "mg/L",
}


# ============================================================
# PARAMETER SCREENING LIMITS
# ============================================================

SCREENING_LIMITS = {
    "ph": {
        "minimum": 6.5,
        "maximum": 8.5,
    },
    "dissolved_oxygen": {
        "minimum": 5.0,
        "maximum": None,
    },
    "turbidity": {
        "minimum": None,
        "maximum": 5.0,
    },
    "temperature": {
        "minimum": 15.0,
        "maximum": 30.0,
    },
    "tss": {
        "minimum": None,
        "maximum": 500.0,
    },
    "total_nitrogen": {
        "minimum": None,
        "maximum": 10.0,
    },
    "total_phosphorus": {
        "minimum": None,
        "maximum": 1.0,
    },
}


# ============================================================
# PETROLEUM-RELATED OPTIONAL PARAMETERS
# ============================================================
#
# These are intentionally optional.
# The application must NOT pretend these parameters exist
# unless they are actually present in the dataset.
#
# If petroleum-specific columns are later added to the dataset,
# this dictionary provides a place to define their interpretation.

PETROLEUM_PARAMETERS = {
    "tph": "Total Petroleum Hydrocarbons",
    "tpH": "Total Petroleum Hydrocarbons",
    "oil_and_grease": "Oil and Grease",
    "btEX": "BTEX",
    "benzene": "Benzene",
    "toluene": "Toluene",
    "ethylbenzene": "Ethylbenzene",
    "xylene": "Xylene",
    "pah": "Polycyclic Aromatic Hydrocarbons",
}


# ============================================================
# APPLICATION CSS
# ============================================================

st.markdown(
    """
    <style>

    .header {
        padding: 22px;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #0d6efd,
            #084298
        );
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }

    .header h1 {
        margin-bottom: 5px;
    }

    .header p {
        margin-bottom: 0;
        font-size: 1rem;
    }

    .section-note {
        color: #666;
        font-size: 0.90rem;
    }

    .risk-high {
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(value) -> Optional[float]:
    """Convert a value safely to float."""

    try:
        if pd.isna(value):
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def display_name(feature: str) -> str:
    """Return human-readable parameter name."""

    return LABELS.get(feature, feature.replace("_", " ").title())


def format_value(
    feature: str,
    value,
    decimals: int = 2,
) -> str:
    """Format parameter value with units."""

    numeric = safe_float(value)

    if numeric is None:
        return "N/A"

    if feature == "ph":
        return f"{numeric:.{decimals}f}"

    unit = UNITS.get(feature, "")

    return f"{numeric:.{decimals}f} {unit}".strip()


# ============================================================
# WQI
# ============================================================

def wqi_category(value) -> str:
    """
    Classify the application screening WQI.

    Higher values indicate better screening quality.
    """

    numeric = safe_float(value)

    if numeric is None:
        return "Unavailable"

    if numeric >= 90:
        return "Excellent"

    if numeric >= 70:
        return "Good"

    if numeric >= 50:
        return "Medium"

    if numeric >= 25:
        return "Poor"

    return "Very Poor"


def calculate_screening_wqi(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the application's screening WQI.

    This is a weighted screening index rather than a claim of
    compliance with a particular regulatory WQI methodology.
    """

    required = [
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

    if any(column not in df.columns for column in required):
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    ph = np.clip(
        100 - (abs(df["ph"] - 7) / 2) * 100,
        0,
        100,
    )

    do = np.clip(
        df["dissolved_oxygen"] / 9 * 100,
        0,
        100,
    )

    turbidity = np.clip(
        100 - df["turbidity"],
        0,
        100,
    )

    temperature = np.clip(
        100 - (abs(df["temperature"] - 20) / 15) * 100,
        0,
        100,
    )

    conductivity = np.clip(
        100 - df["conductivity"] / 500 * 100,
        0,
        100,
    )

    tds = np.clip(
        100 - df["tds"] / 500 * 100,
        0,
        100,
    )

    tss = np.clip(
        100 - df["tss"] / 500 * 100,
        0,
        100,
    )

    nitrogen = np.clip(
        100 - df["total_nitrogen"] / 10 * 100,
        0,
        100,
    )

    phosphorus = np.clip(
        100 - df["total_phosphorus"] * 100,
        0,
        100,
    )

    chlorophyll = np.clip(
        100 - df["chla"] / 20 * 100,
        0,
        100,
    )

    wqi = (
        ph * 0.15
        + do * 0.15
        + turbidity * 0.15
        + temperature * 0.05
        + conductivity * 0.10
        + tds * 0.10
        + tss * 0.10
        + nitrogen * 0.05
        + phosphorus * 0.05
        + chlorophyll * 0.10
    )

    return np.clip(
        wqi,
        0,
        100,
    )


def add_wqi_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add screening WQI and screening risk."""

    result = df.copy()

    result["wqi"] = calculate_screening_wqi(result)

    result["screening_risk"] = pd.cut(
        result["wqi"],
        bins=[
            -np.inf,
            50,
            70,
            np.inf,
        ],
        labels=[
            "High",
            "Medium",
            "Low",
        ],
    ).astype(object)

    result.loc[
        result["wqi"].isna(),
        "screening_risk",
    ] = "Unavailable"

    return result


# ============================================================
# PARAMETER STATUS
# ============================================================

def parameter_status(
    feature: str,
    value,
) -> str:
    """Evaluate a parameter against configured screening limits."""

    numeric = safe_float(value)

    if numeric is None:
        return "Unavailable"

    limits = SCREENING_LIMITS.get(feature)

    if limits is None:
        return "Monitoring"

    minimum = limits["minimum"]
    maximum = limits["maximum"]

    if minimum is not None and numeric < minimum:
        return "Below Limit"

    if maximum is not None and numeric > maximum:
        return "Above Limit"

    return "Normal"


def abnormal_parameters(
    row: pd.Series,
    features: Optional[List[str]] = None,
) -> List[Dict]:
    """Return parameters outside configured screening limits."""

    if features is None:
        features = list(SCREENING_LIMITS.keys())

    results = []

    for feature in features:

        if feature not in row.index:
            continue

        status = parameter_status(
            feature,
            row[feature],
        )

        if status not in [
            "Normal",
            "Monitoring",
            "Unavailable",
        ]:

            results.append(
                {
                    "Feature": feature,
                    "Parameter": display_name(feature),
                    "Value": format_value(
                        feature,
                        row[feature],
                    ),
                    "Status": status,
                }
            )

    return results


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    """
    Load saved ML components.

    Returns:
        regressor
        classifier
        scaler
        model_features
    """

    required_files = {
        "Regressor": REGRESSOR_FILE,
        "Classifier": CLASSIFIER_FILE,
        "Scaler": SCALER_FILE,
        "Feature list": FEATURE_FILE,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required_files.items()
        if not os.path.exists(path)
    ]

    if missing:
        raise FileNotFoundError(
            "Missing ML model component(s):\n"
            + "\n".join(missing)
        )

    regressor = joblib.load(
        REGRESSOR_FILE
    )

    classifier = joblib.load(
        CLASSIFIER_FILE
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    model_features = joblib.load(
        FEATURE_FILE
    )

    if not isinstance(model_features, (list, tuple)):
        raise ValueError(
            "model_features.pkl must contain a list or tuple."
        )

    model_features = list(model_features)

    return (
        regressor,
        classifier,
        scaler,
        model_features,
    )


def validate_model_components(
    regressor,
    classifier,
    scaler,
    features,
) -> List[str]:
    """Check compatibility of saved model components."""

    problems = []

    feature_count = len(features)

    reg_features = getattr(
        regressor,
        "n_features_in_",
        None,
    )

    clf_features = getattr(
        classifier,
        "n_features_in_",
        None,
    )

    scaler_features = getattr(
        scaler,
        "n_features_in_",
        None,
    )

    if reg_features is not None and reg_features != feature_count:
        problems.append(
            f"Regressor expects {reg_features} features "
            f"but feature list contains {feature_count}."
        )

    if clf_features is not None and clf_features != feature_count:
        problems.append(
            f"Classifier expects {clf_features} features "
            f"but feature list contains {feature_count}."
        )

    if scaler_features is not None and scaler_features != feature_count:
        problems.append(
            f"Scaler expects {scaler_features} features "
            f"but feature list contains {feature_count}."
        )

    return problems


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and prepare the processed monitoring dataset."""

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Processed dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    # Convert known numerical fields.
    for feature in DEFAULT_FEATURES:

        if feature in df.columns:

            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

    # Normalize timestamps if available.
    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

    # Add screening WQI.
    df = add_wqi_risk(
        df
    )

    # Add ML risk.
    df["ml_risk"] = "Unavailable"

    try:

        (
            regressor,
            classifier,
            scaler,
            features,
        ) = load_models()

        missing_features = [
            feature
            for feature in features
            if feature not in df.columns
        ]

        if not missing_features:

            complete_rows = (
                df[features]
                .notna()
                .all(axis=1)
            )

            if complete_rows.any():

                X = df.loc[
                    complete_rows,
                    features,
                ]

                X_scaled = scaler.transform(
                    X
                )

                predictions = classifier.predict(
                    X_scaled
                )

                df.loc[
                    complete_rows,
                    "ml_risk",
                ] = predictions.astype(str)

    except Exception:
        # Keep application usable even if ML files are unavailable.
        pass

    return df


# ============================================================
# ATHI RIVER STATION HELPERS
# ============================================================

ATHI_RIVER = "Athi River"

ATHI_STATIONS = {
    "AR01": {
        "name": "Upper Athi",
        "role": "Upstream/reference station",
    },
    "AR02": {
        "name": "Athi River Town",
        "role": "Urban/industrial influence station",
    },
    "AR03": {
        "name": "Kinanie",
        "role": "Industrial influence station",
    },
    "AR04": {
        "name": "Downstream Athi",
        "role": "Downstream assessment station",
    },
}


def get_station_column(
    df: pd.DataFrame,
) -> Optional[str]:

    if "station_id" in df.columns:
        return "station_id"

    return None


def filter_by_station(
    df: pd.DataFrame,
    station: str,
) -> pd.DataFrame:

    if station == "All Stations":
        return df.copy()

    if "station_id" not in df.columns:
        return df.iloc[0:0].copy()

    return df[
        df["station_id"]
        .astype(str)
        .eq(station)
    ].copy()


def station_options(
    df: pd.DataFrame,
) -> List[str]:

    options = ["All Stations"]

    for station_id, station in ATHI_STATIONS.items():

        if "station_id" in df.columns:

            if station_id in df["station_id"].astype(str).values:
                options.append(
                    f"{station_id} — {station['name']}"
                )

    return options


# ============================================================
# DECISION SUPPORT
# ============================================================

def generate_recommendations(
    row: pd.Series,
    ml_risk: str,
) -> List[str]:

    actions = []

    wqi = safe_float(
        row.get("wqi")
    )

    category = wqi_category(
        wqi
    )

    # WQI recommendations.
    if category == "Excellent":

        actions.append(
            "Maintain routine surface-water monitoring."
        )

    elif category == "Good":

        actions.append(
            "Continue routine monitoring and periodic sampling."
        )

    elif category == "Medium":

        actions.append(
            "Increase monitoring attention and investigate "
            "parameters contributing to reduced water quality."
        )

    elif category == "Poor":

        actions.append(
            "Conduct follow-up sampling and investigate possible "
            "sources of water-quality deterioration."
        )

    elif category == "Very Poor":

        actions.append(
            "Conduct an immediate follow-up assessment and "
            "investigate possible contamination sources."
        )

    # ML recommendations.
    if ml_risk == "High":

        actions.extend(
            [
                "Confirm the high-risk prediction using additional sampling.",
                "Investigate possible upstream and local pollution sources.",
                "Increase monitoring frequency.",
            ]
        )

    elif ml_risk == "Medium":

        actions.extend(
            [
                "Conduct follow-up sampling.",
                "Increase monitoring attention.",
            ]
        )

    elif ml_risk == "Low":

        actions.append(
            "The ML model indicates low risk; continue routine "
            "monitoring unless abnormal measurements are observed."
        )

    # Parameter recommendations.
    abnormalities = abnormal_parameters(
        row
    )

    for item in abnormalities:

        actions.append(
            f"Investigate {item['Parameter']} because the "
            f"measured value is {item['Status'].lower()}."
        )

    # Remove duplicates while preserving order.
    return list(
        dict.fromkeys(
            actions
        )
    )


# ============================================================
# DASHBOARD
# ============================================================

def dashboard(
    df: pd.DataFrame,
):

    st.markdown(
        """
        <div class="header">
            <h1>💧 ML-SWQM-DSS</h1>
            <p>
            Machine Learning-Based Surface Water Quality Monitoring
            and Decision Support System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        "📊 Monitoring Overview"
    )

    total = len(df)

    avg_wqi = (
        df["wqi"].mean()
        if "wqi" in df.columns
        else np.nan
    )

    low = int(
        (df["screening_risk"] == "Low").sum()
    )

    medium = int(
        (df["screening_risk"] == "Medium").sum()
    )

    high = int(
        (df["screening_risk"] == "High").sum()
    )

    ml_high = int(
        (df["ml_risk"] == "High").sum()
    )

    a, b, c, d, e = st.columns(5)

    a.metric(
        "Monitoring Samples",
        f"{total:,}",
    )

    b.metric(
        "Average WQI",
        "N/A"
        if pd.isna(avg_wqi)
        else f"{avg_wqi:.2f}",
    )

    c.metric(
        "Low Screening Risk",
        f"{low:,}",
    )

    d.metric(
        "Medium Screening Risk",
        f"{medium:,}",
    )

    e.metric(
        "High Screening Risk",
        f"{high:,}",
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        risk_counts = pd.DataFrame(
            {
                "Risk": [
                    "Low",
                    "Medium",
                    "High",
                ],
                "Count": [
                    low,
                    medium,
                    high,
                ],
            }
        )

        fig = px.bar(
            risk_counts,
            x="Risk",
            y="Count",
            title="Screening Risk Classification",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    with right:

        if "wqi" in df.columns:

            wqi_data = df[
                df["wqi"].notna()
            ]

            fig = px.histogram(
                wqi_data,
                x="wqi",
                nbins=25,
                title="Screening WQI Distribution",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    st.subheader(
        "🧠 Machine-Learning Status"
    )

    ml_available = int(
        (df["ml_risk"] != "Unavailable").sum()
    )

    a, b, c = st.columns(3)

    a.metric(
        "Samples with ML Assessment",
        f"{ml_available:,}",
    )

    b.metric(
        "ML High Risk",
        f"{ml_high:,}",
    )

    c.metric(
        "ML Coverage",
        f"{(ml_available / total * 100):.1f}%"
        if total
        else "0%",
    )

    st.subheader(
        "📌 System Summary"
    )

    st.write(
        f"The processed monitoring dataset contains "
        f"**{total:,} records** and **{len(df.columns)} columns**."
    )

    if pd.notna(avg_wqi):

        st.write(
            f"The mean application screening WQI is "
            f"**{avg_wqi:.2f} ({wqi_category(avg_wqi)})**."
        )

    st.caption(
        "The screening WQI is an application-level index used "
        "for comparative monitoring and decision support."
    )

    if high:

        st.warning(
            f"{high:,} observation(s) have High screening risk "
            "and should receive follow-up investigation."
        )

    else:

        st.success(
            "No High screening-risk observations are currently detected."
        )


# ============================================================
# MONITORING
# ============================================================

def monitoring(
    df: pd.DataFrame,
):

    st.header(
        "📊 Surface Water Monitoring"
    )

    if df.empty:

        st.warning(
            "No monitoring data is available."
        )

        return

    left, right = st.columns(2)

    with left:

        location = st.selectbox(
            "Select Location / Station",
            location_options(df),
        )

    with right:

        available_parameters = [
            feature
            for feature in DEFAULT_FEATURES
            if feature in df.columns
        ]

        if not available_parameters:

            st.warning(
                "No configured monitoring parameters are available."
            )

            return

        parameter = st.selectbox(
            "Select Parameter",
            available_parameters,
            format_func=display_name,
        )

    z = filter_by_location(
        df,
        location,
    )

    if z.empty:

        st.warning(
            "No records match the selected location."
        )

        return

    a, b, c = st.columns(3)

    a.metric(
        "Records",
        f"{len(z):,}",
    )

    b.metric(
        "Average",
        format_value(
            parameter,
            z[parameter].mean(),
        ),
    )

    c.metric(
        "Missing",
        f"{int(z[parameter].isna().sum()):,}",
    )

    # Time series.
    if "timestamp" in z.columns:

        plot_df = z[
            ["timestamp", parameter]
        ].dropna()

        if not plot_df.empty:

            plot_df = plot_df.sort_values(
                "timestamp"
            )

            fig = px.line(
                plot_df,
                x="timestamp",
                y=parameter,
                markers=True,
                title=f"{display_name(parameter)} Over Time",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    st.subheader(
        "Latest Monitoring Records"
    )

    display_columns = [
        column
        for column in [
            "timestamp",
            "date",
            "country",
            "location",
            "station_code",
            "water_body",
            parameter,
            "wqi",
            "screening_risk",
            "ml_risk",
        ]
        if column in z.columns
    ]

    latest = z[
        display_columns
    ].tail(100).iloc[::-1]

    st.dataframe(
        latest,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# PREDICTIONS
# ============================================================

def predictions(
    df: pd.DataFrame,
):

    st.header(
        "🤖 Machine-Learning Predictions"
    )

    try:

        (
            regressor,
            classifier,
            scaler,
            features,
        ) = load_models()

    except Exception as exc:

        st.error(
            "The saved ML model components could not be loaded."
        )

        st.code(
            str(exc)
        )

        return

    problems = validate_model_components(
        regressor,
        classifier,
        scaler,
        features,
    )

    if problems:

        st.error(
            "Model compatibility problems were detected."
        )

        for problem in problems:
            st.write(
                f"• {problem}"
            )

        return

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:

        st.error(
            "The dataset is missing required ML input features:"
        )

        st.write(
            ", ".join(
                missing_features
            )
        )

        return

    complete = (
        df[features]
        .notna()
        .all(axis=1)
    )

    if not complete.any():

        st.warning(
            "No records contain all required ML input features."
        )

        return

    valid = df.loc[
        complete
    ].copy()

    st.caption(
        f"{len(valid):,} records contain all "
        f"{len(features)} required ML input features."
    )

    sample = st.number_input(
        "Select monitoring sample",
        min_value=1,
        max_value=len(valid),
        value=1,
        step=1,
    )

    row = valid.iloc[
        int(sample) - 1
    ]

    X = pd.DataFrame(
        [
            [
                row[feature]
                for feature in features
            ]
        ],
        columns=features,
    )

    try:

        X_scaled = scaler.transform(
            X
        )

        predicted_wqi = float(
            regressor.predict(
                X_scaled
            )[0]
        )

        predicted_wqi = float(
            np.clip(
                predicted_wqi,
                0,
                100,
            )
        )

        predicted_risk = str(
            classifier.predict(
                X_scaled
            )[0]
        )

    except Exception as exc:

        st.error(
            "Prediction failed."
        )

        st.code(
            str(exc)
        )

        return

    a, b, c = st.columns(3)

    a.metric(
        "Predicted WQI",
        f"{predicted_wqi:.2f}",
    )

    b.metric(
        "Predicted Risk",
        predicted_risk,
    )

    c.metric(
        "WQI Category",
        wqi_category(
            predicted_wqi
        ),
    )

    st.subheader(
        "Prediction Interpretation"
    )

    if predicted_risk == "High":

        st.error(
            "The classifier predicts HIGH risk for this sample."
        )

    elif predicted_risk == "Medium":

        st.warning(
            "The classifier predicts MEDIUM risk for this sample."
        )

    elif predicted_risk == "Low":

        st.success(
            "The classifier predicts LOW risk for this sample."
        )

    else:

        st.info(
            f"The classifier returned: {predicted_risk}"
        )

    st.subheader(
        "Model Input Values"
    )

    input_table = pd.DataFrame(
        {
            "Feature": features,
            "Parameter": [
                display_name(feature)
                for feature in features
            ],
            "Value": [
                format_value(
                    feature,
                    row[feature],
                )
                for feature in features
            ],
        }
    )

    st.dataframe(
        input_table,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# RISK ANALYSIS
# ============================================================

def risk_analysis(
    df: pd.DataFrame,
):

    st.header(
        "⚠️ Risk Analysis"
    )

    if "screening_risk" not in df.columns:

        st.error(
            "Screening risk classification is unavailable."
        )

        return

    low = int(
        (df["screening_risk"] == "Low").sum()
    )

    medium = int(
        (df["screening_risk"] == "Medium").sum()
    )

    high = int(
        (df["screening_risk"] == "High").sum()
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Total",
        f"{len(df):,}",
    )

    b.metric(
        "Low",
        f"{low:,}",
    )

    c.metric(
        "Medium",
        f"{medium:,}",
    )

    d.metric(
        "High",
        f"{high:,}",
    )

    risk_df = pd.DataFrame(
        {
            "Risk": [
                "Low",
                "Medium",
                "High",
            ],
            "Count": [
                low,
                medium,
                high,
            ],
        }
    )

    fig = px.bar(
        risk_df,
        x="Risk",
        y="Count",
        title="Screening Risk Level Distribution",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    if high:

        st.error(
            f"{high:,} High Risk observation(s) require "
            "investigation and follow-up sampling."
        )

    elif medium:

        st.warning(
            f"{medium:,} Medium Risk observation(s) require "
            "closer monitoring."
        )

    else:

        st.success(
            "No Medium or High screening-risk observations are present."
        )

    st.subheader(
        "Risk Records"
    )

    columns = [
        column
        for column in [
            "timestamp",
            "date",
            "country",
            "location",
            "station_code",
            "water_body",
            "mission",
            "wqi",
            "screening_risk",
            "ml_risk",
        ]
        if column in df.columns
    ]

    risk_records = (
        df[
            columns
        ]
        .sort_values(
            "wqi",
            ascending=True,
            na_position="last",
        )
        .head(300)
    )

    st.dataframe(
        risk_records,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# DECISION SUPPORT
# ============================================================

def decision_support(
    df: pd.DataFrame,
):

    st.header(
        "🧠 Decision Support System"
    )

    st.caption(
        "The DSS combines the application screening WQI, "
        "machine-learning risk classification and configured "
        "parameter screening limits to support monitoring decisions."
    )

    if df is None or df.empty:

        st.warning(
            "No monitoring data is available."
        )

        return

    required_columns = [
        "wqi",
        "ml_risk",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        st.error(
            "Decision-support data is incomplete: "
            + ", ".join(missing)
        )

        return

    valid_df = df[
        df["wqi"].notna()
        & df["ml_risk"].notna()
        & df["ml_risk"].ne("Unavailable")
    ].copy()

    if valid_df.empty:

        st.warning(
            "No monitoring samples currently have valid "
            "WQI and ML risk results."
        )

        return

    valid_df = valid_df.reset_index(
        drop=True
    )

    sample_number = st.number_input(
        "Select monitoring sample",
        min_value=1,
        max_value=len(valid_df),
        value=1,
        step=1,
    )

    row = valid_df.iloc[
        int(sample_number) - 1
    ]

    wqi_value = safe_float(
        row["wqi"]
    )

    ml_risk = str(
        row["ml_risk"]
    )

    category = wqi_category(
        wqi_value
    )

    abnormalities = abnormal_parameters(
        row
    )

    st.subheader(
        "📊 Water Quality Assessment"
    )

    a, b, c = st.columns(3)

    a.metric(
        "Screening WQI",
        f"{wqi_value:.2f}"
        if wqi_value is not None
        else "N/A",
    )

    b.metric(
        "ML Risk",
        ml_risk,
    )

    c.metric(
        "WQI Category",
        category,
    )

    st.subheader(
        "🔎 Assessment Interpretation"
    )

    st.info(
        f"**Screening WQI:** "
        f"{wqi_value:.2f} — {category}"
    )

    if ml_risk == "High":

        st.error(
            "The machine-learning model classifies this sample "
            "as HIGH RISK."
        )

    elif ml_risk == "Medium":

        st.warning(
            "The machine-learning model classifies this sample "
            "as MEDIUM RISK."
        )

    elif ml_risk == "Low":

        st.success(
            "The machine-learning model classifies this sample "
            "as LOW RISK."
        )

    else:

        st.info(
            "Machine-learning risk classification is unavailable."
        )

    # --------------------------------------------------------
    # PARAMETER ASSESSMENT
    # --------------------------------------------------------

    st.subheader(
        "📋 Parameter Assessment"
    )

    assessment = []

    for feature in DEFAULT_FEATURES:

        if feature not in row.index:
            continue

        if pd.isna(row[feature]):
            continue

        assessment.append(
            {
                "Parameter": display_name(
                    feature
                ),
                "Value": format_value(
                    feature,
                    row[feature],
                ),
                "Status": parameter_status(
                    feature,
                    row[feature],
                ),
            }
        )

    if assessment:

        assessment_df = pd.DataFrame(
            assessment
        )

        st.dataframe(
            assessment_df,
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No parameter-level assessment is available."
        )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    st.subheader(
        "📌 Recommended Actions"
    )

    actions = generate_recommendations(
        row,
        ml_risk,
    )

    for action in actions:

        st.write(
            f"• {action}"
        )

    # --------------------------------------------------------
    # PARAMETERS REQUIRING ATTENTION
    # --------------------------------------------------------

    if abnormalities:

        st.subheader(
            "⚠️ Parameters Requiring Attention"
        )

        for item in abnormalities:

            st.warning(
                f"**{item['Parameter']}** — "
                f"{item['Status']} — "
                f"{item['Value']}"
            )

    else:

        st.success(
            "No configured parameter screening limits "
            "were exceeded."
        )

    # --------------------------------------------------------
    # PETROLEUM CONTEXT
    # --------------------------------------------------------

    petroleum_columns = [
        column
        for column in PETROLEUM_PARAMETERS
        if column in row.index
    ]

    if petroleum_columns:

        st.subheader(
            "🛢️ Petroleum-Related Indicators"
        )

        petroleum_rows = []

        for column in petroleum_columns:

            petroleum_rows.append(
                {
                    "Parameter": PETROLEUM_PARAMETERS[column],
                    "Value": row[column],
                }
            )

        st.dataframe(
            pd.DataFrame(
                petroleum_rows
            ),
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # DECISION SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "📝 Decision Summary"
    )

    location = row.get(
        "location",
        row.get(
            "station_code",
            "selected monitoring location",
        ),
    )

    if abnormalities:

        st.write(
            f"The selected sample from **{location}** has a "
            f"screening WQI of **{wqi_value:.2f} ({category})** "
            f"and an ML risk classification of **{ml_risk}**. "
            f"{len(abnormalities)} configured parameter(s) require "
            "attention and should be verified through follow-up sampling."
        )

    else:

        st.write(
            f"The selected sample from **{location}** has a "
            f"screening WQI of **{wqi_value:.2f} ({category})** "
            f"and an ML risk classification of **{ml_risk}**. "
            "No configured parameter screening limits were exceeded."
        )


# ============================================================
# ALERTS
# ============================================================

def alerts(
    df: pd.DataFrame,
):

    st.header(
        "🔔 Water Quality Alerts"
    )

    monitored = [
        feature
        for feature in SCREENING_LIMITS
        if feature in df.columns
    ]

    if not monitored:

        st.info(
            "No configured screening parameters are available."
        )

        return

    alert_frames = []

    for feature in monitored:

        limits = SCREENING_LIMITS[
            feature
        ]

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        mask = pd.Series(
            False,
            index=df.index,
        )

        if limits["minimum"] is not None:

            mask |= (
                values
                < limits["minimum"]
            )

        if limits["maximum"] is not None:

            mask |= (
                values
                > limits["maximum"]
            )

        if not mask.any():
            continue

        subset = df.loc[
            mask
        ].copy()

        alert = pd.DataFrame(
            {
                "Sample": subset.index + 1,
                "Timestamp": (
                    subset["timestamp"]
                    if "timestamp" in subset.columns
                    else "N/A"
                ),
                "Location": (
                    subset["location"]
                    if "location" in subset.columns
                    else subset.get(
                        "station_code",
                        "N/A",
                    )
                ),
                "Parameter": display_name(
                    feature
                ),
                "Value": [
                    format_value(
                        feature,
                        value,
                    )
                    for value in subset[feature]
                ],
                "Severity": "High",
                "Status": [
                    parameter_status(
                        feature,
                        value,
                    )
                    for value in subset[feature]
                ],
            },
            index=subset.index,
        )

        alert_frames.append(
            alert.reset_index(
                drop=True
            )
        )

    if not alert_frames:

        st.success(
            "No configured parameter-limit alerts were detected."
        )

        return

    alert_df = pd.concat(
        alert_frames,
        ignore_index=True,
    )

    a, b = st.columns(2)

    a.metric(
        "Total Alerts",
        f"{len(alert_df):,}",
    )

    b.metric(
        "High Severity",
        f"{len(alert_df):,}",
    )

    st.dataframe(
        alert_df.head(500),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# REPORT DATA
# ============================================================

def create_report_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    mean_wqi = (
        df["wqi"].mean()
        if "wqi" in df.columns
        else np.nan
    )

    summary = {
        "Samples": f"{len(df):,}",
        "Average pH": (
            f"{df['ph'].mean():.2f}"
            if "ph" in df.columns
            else "N/A"
        ),
        "Average Turbidity": (
            f"{df['turbidity'].mean():.2f} NTU"
            if "turbidity" in df.columns
            else "N/A"
        ),
        "Average Dissolved Oxygen": (
            f"{df['dissolved_oxygen'].mean():.2f} mg/L"
            if "dissolved_oxygen" in df.columns
            else "N/A"
        ),
        "Average Temperature": (
            f"{df['temperature'].mean():.2f} °C"
            if "temperature" in df.columns
            else "N/A"
        ),
        "Average Screening WQI": (
            "N/A"
            if pd.isna(mean_wqi)
            else f"{mean_wqi:.2f}"
        ),
        "Low Screening Risk": (
            f"{int((df['screening_risk'] == 'Low').sum()):,}"
            if "screening_risk" in df.columns
            else "N/A"
        ),
        "Medium Screening Risk": (
            f"{int((df['screening_risk'] == 'Medium').sum()):,}"
            if "screening_risk" in df.columns
            else "N/A"
        ),
        "High Screening Risk": (
            f"{int((df['screening_risk'] == 'High').sum()):,}"
            if "screening_risk" in df.columns
            else "N/A"
        ),
        "ML High Risk": (
            f"{int((df['ml_risk'] == 'High').sum()):,}"
            if "ml_risk" in df.columns
            else "N/A"
        ),
    }

    return pd.DataFrame(
        {
            "Metric": list(
                summary.keys()
            ),
            "Value": list(
                summary.values()
            ),
        }
    )


# ============================================================
# REPORTS
# ============================================================

def reports(
    df: pd.DataFrame,
):

    st.header(
        "📄 Reports"
    )

    report_type = st.selectbox(
        "Report Type",
        [
            "Daily Summary",
            "Weekly Report",
            "Monthly Analysis",
            "Annual Review",
        ],
    )

    preview = create_report_summary(
        df
    )

    st.dataframe(
        preview,
        width="stretch",
        hide_index=True,
    )

    if not st.button(
        "📄 Generate PDF",
        type="primary",
    ):
        return

    if not REPORTLAB:

        st.error(
            "ReportLab is not installed. "
            "Run: pip install reportlab"
        )

        return

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            "ML-SWQM-DSS",
            styles["Title"],
        ),
        Paragraph(
            "Machine Learning-Based Surface Water Quality "
            "Monitoring and Decision Support System",
            styles["Normal"],
        ),
        Spacer(
            1,
            5,
        ),
        Paragraph(
            f"{report_type} — "
            f"{datetime.now():%Y-%m-%d %H:%M}",
            styles["Normal"],
        ),
        Spacer(
            1,
            12,
        ),
        Paragraph(
            "Monitoring Summary",
            styles["Heading2"],
        ),
    ]

    table_data = [
        [
            "Metric",
            "Value",
        ]
    ]

    table_data.extend(
        [
            [
                row.Metric,
                row.Value,
            ]
            for row in preview.itertuples()
        ]
    )

    table = Table(
        table_data,
        colWidths=[
            100 * mm,
            55 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#1976D2"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor(
                            "#F5F9FC"
                        ),
                    ],
                ),
            ]
        )
    )

    story.extend(
        [
            table,
            Spacer(
                1,
                12,
            ),
            Paragraph(
                "Decision-Support Guidance",
                styles["Heading2"],
            ),
            Paragraph(
                "Continue routine monitoring, track the screening "
                "WQI and individual parameters, investigate abnormal "
                "measurements, and review medium- and high-risk "
                "observations through follow-up sampling.",
                styles["Normal"],
            ),
            Spacer(
                1,
                8,
            ),
            Paragraph(
                "Note: The WQI presented by this application is a "
                "screening index intended for monitoring and decision "
                "support. It should not be interpreted as a regulatory "
                "compliance determination without reference to the "
                "applicable water-quality standards.",
                styles["Normal"],
            ),
        ]
    )

    document.build(
        story
    )

    buffer.seek(0)

    filename = (
        "ML-SWQM-DSS_"
        + report_type.replace(
            " ",
            "_",
        )
        + ".pdf"
    )

    st.download_button(
        "📥 Download PDF",
        buffer.getvalue(),
        filename,
        "application/pdf",
    )


# ============================================================
# DATA EXPLORER
# ============================================================

def explorer(
    df: pd.DataFrame,
):

    st.header(
        "🗃️ Data Explorer"
    )

    a, b, c = st.columns(3)

    a.metric(
        "Rows",
        f"{len(df):,}",
    )

    b.metric(
        "Columns",
        f"{len(df.columns):,}",
    )

    c.metric(
        "Missing Cells",
        f"{int(df.isna().sum().sum()):,}",
    )

    search = st.text_input(
        "Search location, station, water body or country",
        "",
    )

    filtered = df

    if search:

        search_columns = [
            column
            for column in [
                "location",
                "station_code",
                "water_body",
                "country",
                "mission",
            ]
            if column in df.columns
        ]

        if search_columns:

            mask = pd.Series(
                False,
                index=df.index,
            )

            for column in search_columns:

                mask |= (
                    df[column]
                    .astype(str)
                    .str.contains(
                        search,
                        case=False,
                        na=False,
                        regex=False,
                    )
                )

            filtered = df.loc[
                mask
            ]

    st.caption(
        f"Showing {min(len(filtered), 500):,} of "
        f"{len(filtered):,} matching records."
    )

    st.dataframe(
        filtered.head(500),
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "📥 Download Filtered CSV",
        filtered.to_csv(
            index=False
        ).encode(
            "utf-8"
        ),
        "ML-SWQM-DSS_filtered_data.csv",
        "text/csv",
    )


# ============================================================
# DATA QUALITY
# ============================================================

def data_quality(
    df: pd.DataFrame,
):

    st.header(
        "🧪 Data Quality"
    )

    st.caption(
        "Quality-control overview of the processed monitoring dataset. "
        "Missing measurements are treated as missing and are not "
        "automatically interpreted as zero."
    )

    if df is None or df.empty:

        st.warning(
            "No monitoring data is available."
        )

        return

    total_cells = (
        df.shape[0]
        * df.shape[1]
    )

    missing_cells = int(
        df.isna().sum().sum()
    )

    completeness = (
        (
            total_cells
            - missing_cells
        )
        / total_cells
        * 100
        if total_cells
        else 0
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    complete_rows = int(
        df.notna()
        .all(axis=1)
        .sum()
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Samples",
        f"{len(df):,}",
    )

    b.metric(
        "Columns",
        f"{len(df.columns):,}",
    )

    c.metric(
        "Missing Cells",
        f"{missing_cells:,}",
    )

    d.metric(
        "Completeness",
        f"{completeness:.1f}%",
    )

    e, f = st.columns(2)

    e.metric(
        "Duplicate Rows",
        f"{duplicate_rows:,}",
    )

    f.metric(
        "Complete Rows",
        f"{complete_rows:,}",
    )

    st.divider()

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    st.subheader(
        "📊 Missing Values by Parameter"
    )

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_table = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing": missing.values,
            "Available": [
                len(df) - int(value)
                for value in missing.values
            ],
            "Completeness (%)": [
                round(
                    (
                        1
                        - int(value)
                        / len(df)
                    )
                    * 100,
                    1,
                )
                if len(df)
                else 0
                for value in missing.values
            ],
        }
    )

    nonzero_missing = (
        missing_table[
            missing_table["Missing"] > 0
        ]
        .head(25)
        .sort_values(
            "Missing"
        )
    )

    if not nonzero_missing.empty:

        fig = px.bar(
            nonzero_missing,
            x="Missing",
            y="Column",
            orientation="h",
            title="Columns with Missing Observations",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    else:

        st.success(
            "No missing cells were detected."
        )

    st.dataframe(
        missing_table,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # ML feature quality
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🤖 ML Input Data Quality"
    )

    try:

        (
            _,
            _,
            _,
            model_features,
        ) = load_models()

    except Exception:

        model_features = DEFAULT_FEATURES

    feature_rows = []

    for feature in model_features:

        if feature in df.columns:

            available = int(
                df[feature]
                .notna()
                .sum()
            )

            feature_rows.append(
                {
                    "Feature": feature,
                    "Parameter": display_name(
                        feature
                    ),
                    "Missing": int(
                        df[feature]
                        .isna()
                        .sum()
                    ),
                    "Available": available,
                    "Completeness (%)": round(
                        df[feature]
                        .notna()
                        .mean()
                        * 100,
                        1,
                    ),
                }
            )

        else:

            feature_rows.append(
                {
                    "Feature": feature,
                    "Parameter": display_name(
                        feature
                    ),
                    "Missing": len(df),
                    "Available": 0,
                    "Completeness (%)": 0.0,
                }
            )

    feature_quality = pd.DataFrame(
        feature_rows
    )

    st.dataframe(
        feature_quality,
        width="stretch",
        hide_index=True,
    )

    incomplete = feature_quality[
        feature_quality["Completeness (%)"] < 100
    ]

    if not incomplete.empty:

        st.warning(
            f"{len(incomplete)} ML input feature(s) contain "
            "missing observations."
        )

    else:

        st.success(
            "All ML input features are complete."
        )

    # --------------------------------------------------------
    # Location completeness
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📍 Completeness by Monitoring Location"
    )

    group_column = get_location_column(
        df
    )

    if group_column:

        rows = []

        for name, group in df.groupby(
            group_column,
            dropna=False,
        ):

            group_cells = (
                group.shape[0]
                * group.shape[1]
            )

            group_missing = int(
                group.isna()
                .sum()
                .sum()
            )

            group_completeness = (
                (
                    group_cells
                    - group_missing
                )
                / group_cells
                * 100
                if group_cells
                else 0
            )

            rows.append(
                {
                    group_column: name,
                    "Samples": len(group),
                    "Missing Cells": group_missing,
                    "Completeness (%)": round(
                        group_completeness,
                        1,
                    ),
                }
            )

        location_quality = pd.DataFrame(
            rows
        )

        st.dataframe(
            location_quality.sort_values(
                "Completeness (%)"
            ),
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No location, station_code or water_body column is available."
        )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.subheader(
        "📝 Quality-Control Interpretation"
    )

    if completeness >= 95:

        st.success(
            f"Overall dataset completeness is "
            f"{completeness:.1f}%."
        )

    elif completeness >= 80:

        st.warning(
            f"Overall dataset completeness is "
            f"{completeness:.1f}%. Review missing observations "
            "before model-dependent interpretation."
        )

    else:

        st.error(
            f"Overall dataset completeness is "
            f"{completeness:.1f}%. Substantial missing data "
            "should be investigated before model-dependent "
            "decisions are made."
        )

    if duplicate_rows:

        st.warning(
            f"{duplicate_rows:,} duplicate row(s) were detected."
        )

    else:

        st.success(
            "No duplicate rows were detected."
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

def model_info():

    st.header(
        "🤖 Model Information"
    )

    try:

        (
            regressor,
            classifier,
            scaler,
            features,
        ) = load_models()

    except Exception as exc:

        st.error(
            "Model components could not be loaded."
        )

        st.code(
            str(exc)
        )

        return

    problems = validate_model_components(
        regressor,
        classifier,
        scaler,
        features,
    )

    if problems:

        st.warning(
            "Model compatibility warnings:"
        )

        for problem in problems:

            st.write(
                f"• {problem}"
            )

    else:

        st.success(
            "Saved model components are internally compatible."
        )

    a, b, c = st.columns(3)

    a.metric(
        "Regressor Features",
        getattr(
            regressor,
            "n_features_in_",
            "N/A",
        ),
    )

    b.metric(
        "Classifier Features",
        getattr(
            classifier,
            "n_features_in_",
            "N/A",
        ),
    )

    c.metric(
        "Scaler Features",
        getattr(
            scaler,
            "n_features_in_",
            "N/A",
        ),
    )

    st.subheader(
        "Model Input Feature Order"
    )

    feature_table = pd.DataFrame(
        {
            "Order": range(
                1,
                len(features) + 1,
            ),
            "Feature": features,
            "Parameter": [
                display_name(
                    feature
                )
                for feature in features
            ],
        }
    )

    st.dataframe(
        feature_table,
        width="stretch",
        hide_index=True,
    )

    st.info(
        "The feature order stored in model_features.pkl must match "
        "the order used when the scaler and ML models were trained."
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    try:

        df = load_data()

    except Exception as exc:

        st.error(
            "Unable to load the processed dataset."
        )

        st.code(
            str(exc)
        )

        st.stop()

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    if os.path.exists(LOGO_FILE):

        st.sidebar.image(
            LOGO_FILE,
            width="stretch",
        )

    st.sidebar.title(
        "ML-SWQM-DSS"
    )

    st.sidebar.caption(
        "Surface Water Quality Monitoring & Decision Support"
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Monitoring",
            "Predictions",
            "Risk Analysis",
            "Decision Support",
            "Reports",
            "Alerts",
            "Data Explorer",
            "Data Quality",
            "Model Information",
        ],
    )

    st.sidebar.divider()

    st.sidebar.metric(
        "Processed Records",
        f"{len(df):,}",
    )

    if "country" in df.columns:

        st.sidebar.metric(
            "Countries",
            f"{df['country'].nunique(dropna=True):,}",
        )

    st.sidebar.caption(
        f"Updated: {datetime.now():%Y-%m-%d %H:%M}"
    )

    # --------------------------------------------------------
    # Page routing
    # --------------------------------------------------------

    if page == "Dashboard":

        dashboard(
            df
        )

    elif page == "Monitoring":

        monitoring(
            df
        )

    elif page == "Predictions":

        predictions(
            df
        )

    elif page == "Risk Analysis":

        risk_analysis(
            df
        )

    elif page == "Decision Support":

        decision_support(
            df
        )

    elif page == "Reports":

        reports(
            df
        )

    elif page == "Alerts":

        alerts(
            df
        )

    elif page == "Data Explorer":

        explorer(
            df
        )

    elif page == "Data Quality":

        data_quality(
            df
        )

    elif page == "Model Information":

        model_info()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
