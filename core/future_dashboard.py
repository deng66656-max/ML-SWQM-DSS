from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_DIR = Path("data/processed/future_forecasts")

SCENARIO_FILES = {
    "Baseline": FORECAST_DIR / "upper_athi_baseline_forecast_2019_2100.csv",
    "Moderate Impact": FORECAST_DIR / "upper_athi_moderate_impact_forecast_2019_2100.csv",
    "High Impact": FORECAST_DIR / "upper_athi_high_impact_forecast_2019_2100.csv",
    "Petroleum Impact": FORECAST_DIR / "upper_athi_petroleum_impact_forecast_2019_2100.csv",
}


DISPLAY_PARAMETERS = {
    "temperature": "Temperature",
    "dissolved_oxygen": "Dissolved Oxygen",
    "turbidity": "Turbidity",
    "chla": "Chlorophyll-a",
    "ph": "pH",
    "conductivity": "Conductivity",
    "tds": "TDS",
    "total_hardness": "Total Hardness",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "nitrite_nitrogen": "Nitrite Nitrogen",
    "ammonium_nitrogen": "Ammonium Nitrogen",
    "chloride": "Chloride",
    "sulphates": "Sulphates",
    "srp": "Soluble Reactive Phosphorus",
    "total_phosphorus": "Total Phosphorus",
    "nitrate_nitrogen": "Nitrate Nitrogen",
    "total_nitrogen": "Total Nitrogen",
    "tss": "Total Suspended Solids",
}


# ============================================================
# WQI CONFIGURATION
# ============================================================

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


SCREENING_LIMITS = {
    "ph": {
        "min": 6.5,
        "max": 8.5,
    },
    "dissolved_oxygen": {
        "min": 5.0,
    },
    "turbidity": {
        "max": 5.0,
    },
    "temperature": {
        "min": 15.0,
        "max": 30.0,
    },
    "tss": {
        "max": 500.0,
    },
    "total_nitrogen": {
        "max": 10.0,
    },
    "total_phosphorus": {
        "max": 1.0,
    },
}


# ============================================================
# CLASSIFICATION FUNCTIONS
# ============================================================

def wqi_category(wqi):
    if pd.isna(wqi):
        return "Unknown"

    if wqi >= 90:
        return "Excellent"

    if wqi >= 70:
        return "Good"

    if wqi >= 50:
        return "Medium"

    if wqi >= 25:
        return "Poor"

    return "Very Poor"


def risk_category(wqi):
    if pd.isna(wqi):
        return "Unknown"

    if wqi < 50:
        return "High"

    if wqi < 70:
        return "Medium"

    return "Low"


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def numeric(series):
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# PARAMETER SCORING
# ============================================================

def parameter_score(parameter, series):
    values = numeric(series)

    if parameter == "ph":
        score = 100 - (np.abs(values - 7.0) * 25)

    elif parameter == "dissolved_oxygen":
        score = (values / 8.0) * 100

    elif parameter == "turbidity":
        score = 100 - (values / 5.0) * 100

    elif parameter == "temperature":
        score = 100 - (np.abs(values - 25.0) / 15.0) * 100

    elif parameter == "conductivity":
        score = 100 - (values / 1000.0) * 100

    elif parameter == "tds":
        score = 100 - (values / 500.0) * 100

    elif parameter == "tss":
        score = 100 - (values / 500.0) * 100

    elif parameter == "total_nitrogen":
        score = 100 - (values / 10.0) * 100

    elif parameter == "total_phosphorus":
        score = 100 - (values / 1.0) * 100

    elif parameter == "chla":
        score = 100 - (values / 50.0) * 100

    else:
        score = pd.Series(100.0, index=values.index)

    return score.clip(lower=0, upper=100)


# ============================================================
# FORECAST WQI
# ============================================================

def calculate_forecast_wqi(df):
    result = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for parameter, weight in WQI_WEIGHTS.items():

        if parameter not in df.columns:
            continue

        scores = parameter_score(
            parameter,
            df[parameter],
        )

        result = result.add(
            scores.fillna(0) * weight,
            fill_value=0,
        )

        total_weight += weight

    if total_weight == 0:
        return pd.Series(np.nan, index=df.index)

    return result / total_weight


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_columns(df):
    df = df.copy()

    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    aliases = {
        "year_": "year",
        "date_year": "year",
        "station_id": "station",
        "station_name": "station",
        "do": "dissolved_oxygen",
        "dissolvedoxygen": "dissolved_oxygen",
        "chlorophyll_a": "chla",
        "chlorophyll": "chla",
        "total_n": "total_nitrogen",
        "tn": "total_nitrogen",
        "total_p": "total_phosphorus",
        "tp": "total_phosphorus",
    }

    df = df.rename(
        columns={
            old: new
            for old, new in aliases.items()
            if old in df.columns
        }
    )

    if "year" in df.columns:
        df["year"] = pd.to_numeric(
            df["year"],
            errors="coerce",
        )

    return df


# ============================================================
# LOAD FORECAST DATA
# ============================================================

@st.cache_data
def load_forecast_data():
    scenarios = {}

    for scenario_name, file_path in SCENARIO_FILES.items():

        if not file_path.exists():
            scenarios[scenario_name] = pd.DataFrame()
            continue

        try:
            df = pd.read_csv(file_path)
            df = normalize_columns(df)

            if "year" not in df.columns:
                scenarios[scenario_name] = pd.DataFrame()
                continue

            scenarios[scenario_name] = df

        except Exception:
            scenarios[scenario_name] = pd.DataFrame()

    return scenarios


# ============================================================
# LOAD OBSERVED DATA
# ============================================================

@st.cache_data
def load_observed_data():
    candidates = [
        Path("data/processed/upper_athi_water_quality.csv"),
        Path("data/processed/cleaned_data.csv"),
    ]

    for path in candidates:

        if path.exists():

            try:
                df = pd.read_csv(path)
                return normalize_columns(df)

            except Exception:
                continue

    return pd.DataFrame()


# ============================================================
# DECISION TEXT
# ============================================================

def decision_text(risk, scenario):

    if risk == "High":
        return (
            f"The {scenario} projection indicates a high forecast screening "
            "risk. Increased monitoring, investigation of contributing "
            "sources, and appropriate preventive or corrective intervention "
            "should be considered."
        )

    if risk == "Medium":
        return (
            f"The {scenario} projection indicates a medium forecast screening "
            "risk. Continued monitoring and investigation of deteriorating "
            "parameters are recommended."
        )

    if risk == "Low":
        return (
            f"The {scenario} projection indicates a low forecast screening "
            "risk under the selected scenario. Routine monitoring should "
            "continue."
        )

    return "No forecast screening decision is available."


# ============================================================
# FUTURE FORECASTING DASHBOARD
# ============================================================

def render_future_forecasting(scenarios):

    st.header(
        "Future Water-Quality Forecasting and Scenario Decision Support"
    )

    st.caption(
        "Upper Athi River Catchment, Kenya"
    )

    st.info(
        "These are scenario-based long-term projections derived from the "
        "available Upper Athi water-quality data. They should be interpreted "
        "as decision-support projections rather than as guaranteed future "
        "measurements."
    )

    # ========================================================
    # FORECAST CONTROLS
    # ========================================================

    st.subheader("Forecast Controls")

    available_scenarios = [
        scenario
        for scenario, df in scenarios.items()
        if not df.empty
    ]

    if not available_scenarios:
        st.error(
            "No forecast scenario files were found. Check the "
            "data/processed/future_forecasts folder."
        )
        return

    # --------------------------------------------------------
    # Default scenario
    # --------------------------------------------------------

    default_scenario = st.session_state.get(
        "future_forecast_scenario",
        available_scenarios[0],
    )

    if default_scenario not in available_scenarios:
        default_scenario = available_scenarios[0]

    selected_scenario = st.selectbox(
        "Select Scenario",
        available_scenarios,
        index=available_scenarios.index(default_scenario),
    )

    scenario_data = scenarios[selected_scenario].copy()

    if scenario_data.empty:
        st.warning(
            f"No forecast data are available for {selected_scenario}."
        )
        return

    min_year = int(scenario_data["year"].min())
    max_year = int(scenario_data["year"].max())

    default_year = st.session_state.get(
        "future_forecast_year",
        min_year,
    )

    if default_year < min_year or default_year > max_year:
        default_year = min_year

    selected_year = st.slider(
        "Projection Year",
        min_value=min_year,
        max_value=max_year,
        value=int(default_year),
        step=1,
    )

    available_parameters = []

    for parameter in DISPLAY_PARAMETERS:

        if parameter in scenario_data.columns:
            available_parameters.append(parameter)

    if not available_parameters:
        st.error(
            "No recognized water-quality parameters were found in the "
            "selected forecast file."
        )
        return

    default_parameter = st.session_state.get(
        "future_forecast_parameter",
        available_parameters[0],
    )

    if default_parameter not in available_parameters:
        default_parameter = available_parameters[0]

    selected_parameter = st.selectbox(
        "Water-Quality Parameter",
        available_parameters,
        index=available_parameters.index(default_parameter),
        format_func=lambda x: DISPLAY_PARAMETERS.get(
            x,
            x.replace("_", " ").title(),
        ),
    )

    # ========================================================
    # RUN FUTURE FORECAST
    # ========================================================

    st.markdown("### Run Future Forecast")

    st.caption(
        "Configure the scenario, projection year and parameter above, "
        "then click the button to execute the selected forecast view."
    )

    run_future_forecast = st.button(
        "▶ Run Future Forecast",
        type="primary",
        width="stretch",
    )

    if run_future_forecast:

        st.session_state["future_forecast_has_run"] = True
        st.session_state["future_forecast_scenario"] = selected_scenario
        st.session_state["future_forecast_year"] = selected_year
        st.session_state["future_forecast_parameter"] = selected_parameter

    if not st.session_state.get(
        "future_forecast_has_run",
        False,
    ):

        st.info(
            "Configure the forecast controls above and click "
            "**Run Future Forecast** to display the projection results."
        )

        return

    # ========================================================
    # ACTIVE FORECAST SELECTION
    # ========================================================

    active_scenario = st.session_state.get(
        "future_forecast_scenario",
        selected_scenario,
    )

    active_year = int(
        st.session_state.get(
            "future_forecast_year",
            selected_year,
        )
    )

    active_parameter = st.session_state.get(
        "future_forecast_parameter",
        selected_parameter,
    )

    if active_scenario not in scenarios:
        active_scenario = selected_scenario

    data = scenarios[active_scenario].copy()

    if data.empty:
        st.error(
            f"No forecast data are available for {active_scenario}."
        )
        return

    if active_parameter not in data.columns:
        active_parameter = selected_parameter

    # ========================================================
    # FORECAST EXECUTED
    # ========================================================

    st.success(
        f"Future forecast executed for **{active_scenario}** "
        f"— projection year **{active_year}**."
    )

    # ========================================================
    # SELECTED YEAR
    # ========================================================

    selected_rows = data[
        data["year"] == active_year
    ].copy()

    if selected_rows.empty:
        st.warning(
            f"No forecast record is available for {active_year}."
        )
        return

    # ========================================================
    # CALCULATE WQI
    # ========================================================

    if "wqi" not in data.columns:

        data["wqi"] = calculate_forecast_wqi(data)

    data["wqi_category"] = data["wqi"].apply(
        wqi_category
    )

    data["forecast_screening_risk"] = data["wqi"].apply(
        risk_category
    )

    selected_rows = data[
        data["year"] == active_year
    ].copy()

    # ========================================================
    # KEY PERFORMANCE INDICATORS
    # ========================================================

    st.subheader("Forecast Summary")

    mean_wqi = selected_rows["wqi"].mean()
    median_wqi = selected_rows["wqi"].median()

    if "station_count" in selected_rows.columns:
        station_count = int(pd.to_numeric(selected_rows["station_count"], errors="coerce").max())
    elif "station" in selected_rows.columns:
        station_count = selected_rows["station"].nunique()
    else:
        station_count = len(selected_rows)

    risk_values = selected_rows[
        "forecast_screening_risk"
    ].value_counts()

    high_risk = int(
        risk_values.get("High", 0)
    )

    medium_risk = int(
        risk_values.get("Medium", 0)
    )

    low_risk = int(
        risk_values.get("Low", 0)
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Mean Forecast WQI",
            "N/A" if pd.isna(mean_wqi)
            else f"{mean_wqi:.2f}",
        )

    with col2:
        st.metric(
            "Median Forecast WQI",
            "N/A" if pd.isna(median_wqi)
            else f"{median_wqi:.2f}",
        )

    with col3:
        st.metric(
            "Stations",
            station_count,
        )

    with col4:
        st.metric(
            "High-Risk Records",
            high_risk,
        )

    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    st.subheader("Forecast Screening Risk")

    risk_df = pd.DataFrame(
        {
            "Risk": [
                "Low",
                "Medium",
                "High",
            ],
            "Records": [
                low_risk,
                medium_risk,
                high_risk,
            ],
        }
    )

    fig_risk = px.bar(
        risk_df,
        x="Risk",
        y="Records",
        title=(
            f"Forecast Screening Risk — "
            f"{active_scenario} — {active_year}"
        ),
    )

    fig_risk.update_layout(
        xaxis_title="Forecast Screening Risk",
        yaxis_title="Number of Records",
    )

    st.plotly_chart(
        fig_risk,
        width="stretch",
    )

    # ========================================================
    # PARAMETER TABLE
    # ========================================================

    st.subheader(
        f"Forecast Parameters — {active_year}"
    )

    parameter_columns = [
        parameter
        for parameter in DISPLAY_PARAMETERS
        if parameter in selected_rows.columns
    ]

    if parameter_columns:

        parameter_summary = []

        for parameter in parameter_columns:

            values = numeric(
                selected_rows[parameter]
            )

            parameter_summary.append(
                {
                    "Parameter": DISPLAY_PARAMETERS.get(
                        parameter,
                        parameter.replace(
                            "_",
                            " ",
                        ).title(),
                    ),
                    "Mean": (
                        values.mean()
                        if not values.dropna().empty
                        else np.nan
                    ),
                    "Minimum": (
                        values.min()
                        if not values.dropna().empty
                        else np.nan
                    ),
                    "Maximum": (
                        values.max()
                        if not values.dropna().empty
                        else np.nan
                    ),
                }
            )

        parameter_table = pd.DataFrame(
            parameter_summary
        )

        st.dataframe(
            parameter_table,
            width="stretch",
            hide_index=True,
        )

    # ========================================================
    # SELECTED PARAMETER FORECAST
    # ========================================================

    st.subheader(
        f"{DISPLAY_PARAMETERS.get(active_parameter, active_parameter)} "
        "Long-Term Projection"
    )

    parameter_data = data[
        [
            "year",
            active_parameter,
        ]
    ].copy()

    parameter_data[active_parameter] = numeric(
        parameter_data[active_parameter]
    )

    parameter_data = (
        parameter_data
        .dropna(subset=[active_parameter])
        .groupby("year", as_index=False)[active_parameter]
        .median()
    )

    if not parameter_data.empty:

        fig_parameter = px.line(
            parameter_data,
            x="year",
            y=active_parameter,
            markers=True,
            title=(
                f"{DISPLAY_PARAMETERS.get(active_parameter, active_parameter)} "
                f"Projection — {active_scenario}"
            ),
        )

        fig_parameter.update_layout(
            xaxis_title="Year",
            yaxis_title=DISPLAY_PARAMETERS.get(
                active_parameter,
                active_parameter.replace(
                    "_",
                    " ",
                ).title(),
            ),
        )

        st.plotly_chart(
            fig_parameter,
            width="stretch",
        )

    # ========================================================
    # PROJECTED WQI
    # ========================================================

    st.subheader(
        "Projected Water Quality Index"
    )

    wqi_by_year = (
        data[
            [
                "year",
                "wqi",
            ]
        ]
        .dropna(subset=["wqi"])
        .groupby("year", as_index=False)["wqi"]
        .median()
    )

    if not wqi_by_year.empty:

        fig_wqi = px.line(
            wqi_by_year,
            x="year",
            y="wqi",
            markers=True,
            title=(
                f"Projected WQI — {active_scenario}"
            ),
        )

        fig_wqi.add_hline(
            y=70,
            line_dash="dash",
            annotation_text="Good / Medium boundary",
        )

        fig_wqi.add_hline(
            y=50,
            line_dash="dash",
            annotation_text="Medium / Poor boundary",
        )

        fig_wqi.update_layout(
            xaxis_title="Year",
            yaxis_title="WQI",
            yaxis_range=[0, 100],
        )

        st.plotly_chart(
            fig_wqi,
            width="stretch",
        )

    # ========================================================
    # SCENARIO COMPARISON
    # ========================================================

    st.subheader(
        f"Scenario Comparison — {active_year}"
    )

    comparison_rows = []

    for scenario_name, scenario_df in scenarios.items():

        if scenario_df.empty:
            continue

        scenario_copy = scenario_df.copy()

        if "wqi" not in scenario_copy.columns:
            scenario_copy["wqi"] = calculate_forecast_wqi(
                scenario_copy
            )

        rows = scenario_copy[
            scenario_copy["year"] == active_year
        ]

        if rows.empty:
            continue

        mean_scenario_wqi = rows["wqi"].mean()

        comparison_rows.append(
            {
                "Scenario": scenario_name,
                "Mean WQI": mean_scenario_wqi,
                "Risk": risk_category(
                    mean_scenario_wqi
                ),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    if not comparison_df.empty:

        fig_comparison = px.bar(
            comparison_df,
            x="Scenario",
            y="Mean WQI",
            color="Risk",
            title=(
                f"Scenario WQI Comparison — {active_year}"
            ),
            text_auto=".2f",
        )

        fig_comparison.update_layout(
            yaxis_title="Mean WQI",
            yaxis_range=[0, 100],
        )

        st.plotly_chart(
            fig_comparison,
            width="stretch",
        )

        st.dataframe(
            comparison_df,
            width="stretch",
            hide_index=True,
        )

    # ========================================================
    # LONG-TERM CHECKPOINTS
    # ========================================================

    st.subheader(
        "Long-Term Forecast Checkpoints"
    )

    checkpoint_years = [
        year
        for year in [
            2020,
            2030,
            2040,
            2050,
            2060,
            2070,
            2080,
            2090,
            2100,
        ]
        if year >= int(data["year"].min())
        and year <= int(data["year"].max())
    ]

    checkpoint_rows = []

    for year in checkpoint_years:

        rows = data[
            data["year"] == year
        ]

        if rows.empty:
            continue

        mean_wqi_year = rows["wqi"].mean()

        checkpoint_rows.append(
            {
                "Year": year,
                "Mean WQI": mean_wqi_year,
                "WQI Category": wqi_category(
                    mean_wqi_year
                ),
                "Forecast Screening Risk": risk_category(
                    mean_wqi_year
                ),
            }
        )

    checkpoint_df = pd.DataFrame(
        checkpoint_rows
    )

    if not checkpoint_df.empty:

        st.dataframe(
            checkpoint_df,
            width="stretch",
            hide_index=True,
        )

    # ========================================================
    # PETROLEUM ENVIRONMENTAL DSS
    # ========================================================

    st.subheader(
        "Petroleum Environmental Decision Support"
    )

    if active_scenario == "Petroleum Impact":

        st.warning(
            "The Petroleum Impact scenario is a hypothetical decision-support "
            "scenario. It does not establish that petroleum contamination "
            "was measured in the Upper Athi River Catchment."
        )

        petroleum_parameters = [
            "Total Petroleum Hydrocarbons (TPH)",
            "BTEX",
            "Polycyclic Aromatic Hydrocarbons (PAHs)",
            "Oil and Grease",
        ]

        st.markdown(
            "### Recommended Petroleum Monitoring Parameters"
        )

        for item in petroleum_parameters:
            st.write(
                f"- {item}"
            )

    else:

        st.info(
            "Petroleum-specific monitoring is recommended when petroleum "
            "activities or potential spill pathways are relevant to the "
            "selected scenario."
        )

    # ========================================================
    # PETROLEUM ENVIRONMENTAL PATHWAY
    # ========================================================

    st.markdown(
        """
### Petroleum Environmental Pathway

**Petroleum activity** → exploration / production / pipeline transport / storage / terminal operations

↓

**Potential environmental pathway** → spills / leaks / contaminated runoff / drainage / accidental releases

↓

**Surface-water receptor** → Upper Athi River Catchment

↓

**Monitoring parameters** → WQI parameters + petroleum-specific indicators

↓

**ML-SWQM-DSS**

↓

**Projected WQI + Forecast Screening Risk**

↓

**Petroleum engineering decision** → prevention / containment / monitoring / investigation / intervention
"""
    )

    # ========================================================
    # DECISION SUPPORT
    # ========================================================

    selected_mean_wqi = selected_rows["wqi"].mean()

    selected_risk = risk_category(
        selected_mean_wqi
    )

    st.markdown(
        "### Forecast Decision"
    )

    st.write(
        decision_text(
            selected_risk,
            active_scenario,
        )
    )

    # ========================================================
    # OBSERVED CONTEXT
    # ========================================================

    observed = load_observed_data()

    st.subheader(
        "Observed Data Context"
    )

    if observed.empty:

        st.info(
            "Observed Upper Athi data could not be loaded for this context panel."
        )

    else:

        observed_years = []

        if "year" in observed.columns:

            observed_years = sorted(
                observed["year"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

        st.write(
            f"Observed data years available: "
            f"{', '.join(map(str, observed_years))}"
            if observed_years
            else "Observed year information is unavailable."
        )

        if "station" in observed.columns:

            observed_station_count = observed["station"].nunique()

            st.write(
                f"Observed monitoring stations: {observed_station_count}"
            )

        elif "station_count" in observed.columns:

            observed_station_count = int(
                pd.to_numeric(
                    observed["station_count"],
                    errors="coerce",
                ).max()
            )

            st.write(
                f"Observed monitoring stations: {observed_station_count}"
            )

    # ========================================================
    # LIMITATIONS
    # ========================================================

    st.subheader(
        "Forecast Limitations"
    )

    st.markdown(
        """
- The available observed Upper Athi dataset covers a limited historical period.
- Long-term projections to 2100 contain substantial uncertainty.
- Scenario outputs are decision-support projections and should not be interpreted as guaranteed future observations.
- Forecast screening risk is derived from projected WQI and is not a regulatory compliance classification.
- Petroleum Impact projections are hypothetical and are intended for scenario analysis.
- Petroleum-specific contamination should be confirmed through appropriate field and laboratory monitoring.
- Station-level uncertainty is represented through the observed/projected distribution rather than formal confidence intervals.
"""
    )

    # ========================================================
    # DOWNLOADS
    # ========================================================

    st.subheader(
        "Forecast Downloads"
    )

    download_data = data.copy()

    csv_data = download_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Selected Scenario Forecast CSV",
        data=csv_data,
        file_name=(
            f"{active_scenario.lower().replace(' ', '_')}"
            f"_forecast.csv"
        ),
        mime="text/csv",
        width="stretch",
    )


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def show_future_forecasting():

    scenarios = load_forecast_data()

    render_future_forecasting(
        scenarios
    )
