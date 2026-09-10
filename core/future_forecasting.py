from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# ================================================================
# ML-SWQM-DSS
# UPPER ATHI RIVER CATCHMENT
# FUTURE WATER-QUALITY PROJECTION ENGINE
#
# 2017-2018 = observed data
# 2018       = baseline year
# 2019-2100  = future scenario projections
#
# IMPORTANT:
# The current dataset contains only two observed years.
# Therefore 2019-2100 are projections/scenarios, NOT validated
# long-term observations.
# ================================================================


# ================================================================
# PATHS
# ================================================================

DEFAULT_DATA_PATH = Path(
    "data/processed/upper_athi_water_quality.csv"
)

OUTPUT_DIR = Path(
    "data/processed/future_forecasts"
)


# ================================================================
# TIME
# ================================================================

OBSERVED_START = 2017
BASELINE_YEAR = 2018

FORECAST_START = 2019
FORECAST_END = 2100


# ================================================================
# PARAMETERS
# ================================================================

PARAMETERS = [
    "chla",
    "chloride",
    "chromium",
    "copper",
    "dissolved_oxygen",
    "conductivity",
    "iron",
    "manganese",
    "lead",
    "pH",
    "tds",
    "temperature",
    "total_nitrogen",
    "total_phosphorus",
    "tss",
    "turbidity",
    "zinc",
]


# ================================================================
# PARAMETER TYPES
# ================================================================

PARAMETER_TYPE = {

    "chla": "pollutant",
    "chloride": "pollutant",
    "chromium": "pollutant",
    "copper": "pollutant",
    "conductivity": "pollutant",
    "iron": "pollutant",
    "manganese": "pollutant",
    "lead": "pollutant",
    "tds": "pollutant",
    "total_nitrogen": "pollutant",
    "total_phosphorus": "pollutant",
    "tss": "pollutant",
    "turbidity": "pollutant",
    "zinc": "pollutant",

    "dissolved_oxygen": "beneficial",

    "pH": "pH",

    "temperature": "temperature",
}


# ================================================================
# NUMERICAL BOUNDS
# ================================================================

BOUNDS = {

    "chla": (0.0, 20.0),
    "chloride": (0.0, 2000.0),
    "chromium": (0.0, 5.0),
    "copper": (0.0, 5.0),
    "dissolved_oxygen": (0.5, 14.0),
    "conductivity": (20.0, 10000.0),
    "iron": (0.0, 1000.0),
    "manganese": (0.0, 500.0),
    "lead": (0.0, 5.0),
    "pH": (6.0, 9.0),
    "tds": (10.0, 7000.0),
    "temperature": (15.0, 35.0),
    "total_nitrogen": (0.0, 50.0),
    "total_phosphorus": (0.0, 20.0),
    "tss": (0.0, 5000.0),
    "turbidity": (0.0, 3000.0),
    "zinc": (0.0, 20.0),
}


# ================================================================
# SCENARIOS
# ================================================================

SCENARIOS = {

    "Baseline": 0.00,

    "Moderate Impact": 0.25,

    "High Impact": 0.50,

    "Petroleum Impact": 0.75,
}


# ================================================================
# MAXIMUM LONG-TERM CHANGE FROM 2018 BASELINE
#
# These are deliberately conservative because only two historical
# years are currently available.
# ================================================================

MAX_RELATIVE_CHANGE = {

    "chla": 0.50,
    "chloride": 0.40,
    "chromium": 0.50,
    "copper": 0.50,
    "conductivity": 0.35,
    "iron": 0.50,
    "manganese": 0.50,
    "lead": 0.50,
    "tds": 0.35,
    "total_nitrogen": 0.60,
    "total_phosphorus": 0.60,
    "tss": 0.60,
    "turbidity": 0.60,
    "zinc": 0.50,

    "dissolved_oxygen": 0.30,

    "pH": 0.05,

    "temperature": 0.08,
}


# ================================================================
# SCENARIO MAXIMUM EFFECT BY 2100
# ================================================================

SCENARIO_RELATIVE_EFFECT = {

    "chla": 0.35,
    "chloride": 0.25,
    "chromium": 0.30,
    "copper": 0.30,
    "conductivity": 0.25,
    "iron": 0.30,
    "manganese": 0.30,
    "lead": 0.35,
    "tds": 0.25,
    "total_nitrogen": 0.30,
    "total_phosphorus": 0.35,
    "tss": 0.35,
    "turbidity": 0.35,
    "zinc": 0.30,

    "dissolved_oxygen": 0.25,

    "pH": 0.04,

    "temperature": 0.05,
}


# ================================================================
# LOAD DATA
# ================================================================

def load_data(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:

    data_path = Path(data_path)

    if not data_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(
        data_path
    )

    required = [
        "station",
        "station_name",
        "year",
    ] + PARAMETERS

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "station",
            "year",
        ]
    ).copy()

    df["year"] = df[
        "year"
    ].astype(int)

    for parameter in PARAMETERS:

        df[parameter] = pd.to_numeric(
            df[parameter],
            errors="coerce",
        )

    return df


# ================================================================
# ROBUST HISTORICAL RELATIVE TREND
#
# Only the direction of 2017 -> 2018 is used.
# Its magnitude is heavily damped because the historical record
# is only two years long.
# ================================================================

def estimate_trends(
    df: pd.DataFrame,
) -> Dict[str, float]:

    trends = {}

    for parameter in PARAMETERS:

        changes = []

        for station_id, group in df.groupby(
            "station"
        ):

            old = group[
                group["year"]
                == OBSERVED_START
            ][parameter]

            new = group[
                group["year"]
                == BASELINE_YEAR
            ][parameter]

            if old.empty or new.empty:
                continue

            old_value = old.iloc[0]
            new_value = new.iloc[0]

            if (
                pd.isna(old_value)
                or pd.isna(new_value)
            ):
                continue

            old_value = float(
                old_value
            )

            new_value = float(
                new_value
            )

            denominator = max(
                abs(old_value),
                1e-6,
            )

            relative_change = (
                new_value
                - old_value
            ) / denominator

            if np.isfinite(
                relative_change
            ):

                changes.append(
                    relative_change
                )

        if not changes:

            trends[parameter] = 0.0
            continue

        values = np.asarray(
            changes,
            dtype=float,
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:

            trends[parameter] = 0.0
            continue

        low = np.quantile(
            values,
            0.10,
        )

        high = np.quantile(
            values,
            0.90,
        )

        values = np.clip(
            values,
            low,
            high,
        )

        trends[parameter] = float(
            np.median(values)
        )

    return trends


# ================================================================
# HISTORICAL TARGET
#
# The 2017-2018 direction is retained, but the magnitude is
# strongly damped and capped.
# ================================================================

def calculate_historical_target(
    parameter: str,
    baseline_value: float,
    historical_relative_change: float,
) -> float:

    parameter_type = PARAMETER_TYPE[
        parameter
    ]

    maximum_change = MAX_RELATIVE_CHANGE[
        parameter
    ]

    # ------------------------------------------------------------
    # Historical trend contribution
    # ------------------------------------------------------------

    # We only allow a fraction of the observed one-year change
    # to influence the long-term target.

    damped_change = (
        historical_relative_change
        * 0.20
    )

    damped_change = np.clip(
        damped_change,
        -maximum_change,
        maximum_change,
    )

    # ------------------------------------------------------------
    # Pollutants
    # ------------------------------------------------------------

    if parameter_type == "pollutant":

        # Do not allow baseline conditions to collapse toward zero
        # merely because one historical year happened to be lower.

        damped_change = max(
            damped_change,
            -0.20,
        )

        target = (
            baseline_value
            * (
                1.0
                + damped_change
            )
        )

    # ------------------------------------------------------------
    # Dissolved oxygen
    # ------------------------------------------------------------

    elif parameter_type == "beneficial":

        damped_change = np.clip(
            damped_change,
            -0.20,
            0.15,
        )

        target = (
            baseline_value
            * (
                1.0
                + damped_change
            )
        )

    # ------------------------------------------------------------
    # pH
    # ------------------------------------------------------------

    elif parameter_type == "pH":

        # pH is treated as an absolute environmental variable,
        # not as an ordinary concentration.

        absolute_change = (
            historical_relative_change
            * 0.05
        )

        absolute_change = np.clip(
            absolute_change,
            -0.30,
            0.30,
        )

        target = (
            baseline_value
            + absolute_change
        )

    # ------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------

    else:

        # Temperature receives an extremely conservative trend.

        absolute_change = (
            historical_relative_change
            * 0.20
        )

        absolute_change = np.clip(
            absolute_change,
            -1.0,
            1.0,
        )

        target = (
            baseline_value
            + absolute_change
        )

    lower, upper = BOUNDS[
        parameter
    ]

    return float(
        np.clip(
            target,
            lower,
            upper,
        )
    )


# ================================================================
# SCENARIO TARGET
# ================================================================

def calculate_scenario_target(
    parameter: str,
    baseline_value: float,
    historical_target: float,
    scenario: str,
) -> float:

    pressure = SCENARIOS[
        scenario
    ]

    if pressure == 0.0:

        return historical_target

    parameter_type = PARAMETER_TYPE[
        parameter
    ]

    maximum_effect = (
        SCENARIO_RELATIVE_EFFECT[
            parameter
        ]
    )

    # ------------------------------------------------------------
    # Pollutants
    # ------------------------------------------------------------

    if parameter_type == "pollutant":

        effect = (
            baseline_value
            * maximum_effect
            * pressure
        )

        target = (
            historical_target
            + effect
        )

    # ------------------------------------------------------------
    # Dissolved oxygen
    # ------------------------------------------------------------

    elif parameter_type == "beneficial":

        reduction = (
            baseline_value
            * maximum_effect
            * pressure
        )

        target = (
            historical_target
            - reduction
        )

    # ------------------------------------------------------------
    # pH
    # ------------------------------------------------------------

    elif parameter_type == "pH":

        reduction = (
            maximum_effect
            * pressure
        )

        target = (
            historical_target
            - reduction
        )

    # ------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------

    else:

        increase = (
            baseline_value
            * maximum_effect
            * pressure
        )

        target = (
            historical_target
            + increase
        )

    lower, upper = BOUNDS[
        parameter
    ]

    return float(
        np.clip(
            target,
            lower,
            upper,
        )
    )


# ================================================================
# PROJECT VALUE
# ================================================================

def project_value(
    parameter: str,
    baseline_value: float,
    historical_relative_change: float,
    scenario: str,
    years: int,
) -> float:

    lower, upper = BOUNDS[
        parameter
    ]

    if not np.isfinite(
        baseline_value
    ):

        return np.nan

    baseline_value = float(
        np.clip(
            baseline_value,
            lower,
            upper,
        )
    )

    if years <= 0:

        return baseline_value

    historical_target_value = (
        calculate_historical_target(
            parameter=parameter,
            baseline_value=baseline_value,
            historical_relative_change=(
                historical_relative_change
            ),
        )
    )

    target = calculate_scenario_target(
        parameter=parameter,
        baseline_value=baseline_value,
        historical_target=(
            historical_target_value
        ),
        scenario=scenario,
    )

    # ------------------------------------------------------------
    # Long-term asymptotic transition
    #
    # 20 years = primary adjustment period.
    # By 2100 the trajectory approaches the scenario target,
    # rather than continuing indefinitely.
    # ------------------------------------------------------------

    response = (
        1.0
        - np.exp(
            -years / 20.0
        )
    )

    value = (
        baseline_value
        + (
            target
            - baseline_value
        )
        * response
    )

    value = np.clip(
        value,
        lower,
        upper,
    )

    return float(
        value
    )


# ================================================================
# STATION FORECAST
# ================================================================

def generate_station_forecast(
    df: pd.DataFrame,
    scenario: str,
    start_year: int = FORECAST_START,
    end_year: int = FORECAST_END,
) -> pd.DataFrame:

    trends = estimate_trends(
        df
    )

    baseline = (
        df[
            df["year"]
            == BASELINE_YEAR
        ]
        .sort_values("station")
        .drop_duplicates(
            "station"
        )
        .copy()
    )

    if baseline.empty:

        raise ValueError(
            f"No observations found for "
            f"baseline year {BASELINE_YEAR}."
        )

    rows = []

    for _, station in baseline.iterrows():

        for year in range(
            start_year,
            end_year + 1,
        ):

            years = (
                year
                - BASELINE_YEAR
            )

            row = {

                "station":
                    station[
                        "station"
                    ],

                "station_name":
                    station[
                        "station_name"
                    ],

                "year":
                    year,

                "data_type":
                    "Future Projection",

                "scenario":
                    scenario,
            }

            for parameter in PARAMETERS:

                baseline_value = station[
                    parameter
                ]

                historical_relative_change = (
                    trends[
                        parameter
                    ]
                )

                row[
                    parameter
                ] = project_value(
                    parameter=parameter,
                    baseline_value=(
                        float(
                            baseline_value
                        )
                        if pd.notna(
                            baseline_value
                        )
                        else np.nan
                    ),
                    historical_relative_change=(
                        historical_relative_change
                    ),
                    scenario=scenario,
                    years=years,
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# ================================================================
# CATCHMENT AGGREGATION
# ================================================================

def aggregate_forecast(
    station_forecast: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for year, group in (
        station_forecast.groupby(
            "year"
        )
    ):

        row = {

            "year":
                int(year),

            "data_type":
                "Future Projection",

            "scenario":
                group[
                    "scenario"
                ].iloc[0],

            "station_count":
                int(
                    group[
                        "station"
                    ].nunique()
                ),
        }

        for parameter in PARAMETERS:

            values = pd.to_numeric(
                group[
                    parameter
                ],
                errors="coerce",
            ).dropna()

            if values.empty:

                row[
                    parameter
                ] = np.nan

                row[
                    f"{parameter}_lower"
                ] = np.nan

                row[
                    f"{parameter}_upper"
                ] = np.nan

            else:

                row[
                    parameter
                ] = float(
                    values.median()
                )

                row[
                    f"{parameter}_lower"
                ] = float(
                    values.quantile(
                        0.10
                    )
                )

                row[
                    f"{parameter}_upper"
                ] = float(
                    values.quantile(
                        0.90
                    )
                )

        rows.append(
            row
        )

    return (
        pd.DataFrame(rows)
        .sort_values("year")
        .reset_index(drop=True)
    )


# ================================================================
# BUILD ALL SCENARIOS
# ================================================================

def build_all_scenarios(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> Dict[str, pd.DataFrame]:

    df = load_data(
        data_path
    )

    outputs = {}

    for scenario in SCENARIOS:

        station_forecast = (
            generate_station_forecast(
                df=df,
                scenario=scenario,
                start_year=FORECAST_START,
                end_year=FORECAST_END,
            )
        )

        outputs[
            scenario
        ] = aggregate_forecast(
            station_forecast
        )

    return outputs


# ================================================================
# SAVE
# ================================================================

def save_forecasts(
    outputs: Dict[str, pd.DataFrame],
    output_dir: str | Path = OUTPUT_DIR,
) -> None:

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for scenario, data in outputs.items():

        filename = (
            scenario.lower()
            .replace(
                " ",
                "_",
            )
            .replace(
                "-",
                "_",
            )
        )

        path = (
            output_dir
            / (
                "upper_athi_"
                f"{filename}"
                "_forecast_2019_2100.csv"
            )
        )

        data.to_csv(
            path,
            index=False,
        )

        print(
            f"SAVED: {path}"
        )


# ================================================================
# DIAGNOSTIC
# ================================================================

def diagnostic_report(
    df: pd.DataFrame,
    outputs: Dict[str, pd.DataFrame],
) -> None:

    print()
    print("=" * 80)
    print(
        "ML-SWQM-DSS FUTURE "
        "WATER-QUALITY PROJECTION"
    )
    print("=" * 80)

    print(
        f"Observed period : "
        f"{df['year'].min()}-"
        f"{df['year'].max()}"
    )

    print(
        f"Baseline year   : "
        f"{BASELINE_YEAR}"
    )

    print(
        f"Forecast period : "
        f"{FORECAST_START}-"
        f"{FORECAST_END}"
    )

    print(
        f"Observed rows   : "
        f"{len(df)}"
    )

    print(
        f"Stations        : "
        f"{df['station'].nunique()}"
    )

    print()

    selected_years = [
        2019,
        2030,
        2050,
        2075,
        2100,
    ]

    for scenario, data in outputs.items():

        print("-" * 80)
        print(
            f"SCENARIO: {scenario}"
        )
        print("-" * 80)

        selected = data[
            data["year"].isin(
                selected_years
            )
        ]

        columns = [
            "year",
            "dissolved_oxygen",
            "pH",
            "temperature",
            "conductivity",
            "tds",
            "turbidity",
            "tss",
            "total_nitrogen",
            "total_phosphorus",
            "lead",
            "chromium",
        ]

        print(
            selected[
                columns
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.3f}",
            )
        )

    print()
    print("=" * 80)
    print(
        "FORECAST RANGE CHECK"
    )
    print("=" * 80)

    for scenario, data in outputs.items():

        print(
            f"\n{scenario}"
        )

        for parameter in PARAMETERS:

            values = data[
                parameter
            ].dropna()

            if values.empty:
                continue

            lower, upper = BOUNDS[
                parameter
            ]

            print(
                f"{parameter:20s} "
                f"min={values.min():.4f} "
                f"max={values.max():.4f} "
                f"allowed={lower}-{upper}"
            )

    print()
    print("=" * 80)
    print("IMPORTANT:")
    print(
        "2017-2018 = observed Upper Athi data."
    )
    print(
        "2018 = observed baseline year."
    )
    print(
        "2019-2100 = future projections/scenarios."
    )
    print(
        "Petroleum Impact is a hypothetical "
        "environmental-pressure scenario."
    )
    print(
        "It does NOT mean petroleum contamination "
        "was measured in the current dataset."
    )
    print(
        "The projections are not validated "
        "2100 observations."
    )
    print("=" * 80)
    print()


# ================================================================
# MAIN
# ================================================================

def run_forecasting_pipeline():

    df = load_data()

    outputs = (
        build_all_scenarios()
    )

    save_forecasts(
        outputs
    )

    diagnostic_report(
        df,
        outputs,
    )

    return outputs


# ================================================================
# EXECUTION
# ================================================================

if __name__ == "__main__":

    run_forecasting_pipeline()
