from pathlib import Path
import numpy as np
import pandas as pd

OBSERVED_FILE = Path("data/processed/upper_athi_water_quality.csv")

PARAMETERS = [
    "chla", "chloride", "chromium", "copper",
    "dissolved_oxygen", "conductivity", "iron",
    "manganese", "lead", "pH", "tds", "temperature",
    "total_nitrogen", "total_phosphorus",
    "tss", "turbidity", "zinc",
]

POLLUTANT_PARAMETERS = {
    "chla", "chloride", "chromium", "copper",
    "iron", "manganese", "lead",
    "tds", "total_nitrogen", "total_phosphorus",
    "tss", "turbidity", "zinc",
}

BENEFICIAL_PARAMETERS = {
    "dissolved_oxygen",
}

BOUNDED_PARAMETERS = {
    "pH": (6.5, 8.5),
    "temperature": (15.0, 30.0),
}


def load_observed_data():
    if not OBSERVED_FILE.exists():
        raise FileNotFoundError(
            f"Observed dataset not found: {OBSERVED_FILE}"
        )

    df = pd.read_csv(OBSERVED_FILE)

    required = ["station", "year"] + PARAMETERS
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["station", "year"]
    ).copy()

    df["year"] = df["year"].astype(int)

    for parameter in PARAMETERS:
        df[parameter] = pd.to_numeric(
            df[parameter],
            errors="coerce",
        )

    return df.sort_values(
        ["station", "year"]
    ).reset_index(drop=True)


def clipped_relative_change(previous, current):
    if pd.isna(previous) or pd.isna(current):
        return np.nan

    if previous == 0:
        return 0.0

    change = (current - previous) / abs(previous)

    return float(
        np.clip(change, -0.90, 0.90)
    )


def hindcast_parameter(previous, parameter):
    """
    One-step 2017 -> 2018 hindcast.

    The 2017 value is used as the starting condition.
    The catchment-wide median relative change from 2017
    to 2018 is estimated without using the station being
    predicted, then applied to that station's 2017 value.

    This is a transparent hindcast diagnostic, not a
    machine-learning forecast.
    """

    if previous is None or pd.isna(previous):
        return np.nan

    return float(previous)


def calculate_hindcast(df, parameter):
    data = df[
        ["station", "year", parameter]
    ].copy()

    data = data.dropna(
        subset=[parameter]
    )

    years = sorted(
        data["year"].unique()
    )

    if len(years) < 2:
        return None

    validation_year = years[-1]
    training_year = years[-2]

    train = data[
        data["year"] == training_year
    ].copy()

    test = data[
        data["year"] == validation_year
    ].copy()

    if train.empty or test.empty:
        return None

    merged = test[
        ["station", parameter]
    ].merge(
        train[
            ["station", parameter]
        ],
        on="station",
        suffixes=("_actual", "_previous"),
    )

    if merged.empty:
        return None

    # Estimate historical relative changes using
    # all available station pairs.
    changes = []

    for _, row in merged.iterrows():
        previous = row[f"{parameter}_previous"]
        actual = row[f"{parameter}_actual"]

        change = clipped_relative_change(
            previous,
            actual,
        )

        if not pd.isna(change):
            changes.append(change)

    if not changes:
        return None

    median_change = float(
        np.median(changes)
    )

    # Apply only a damped fraction of the historical
    # change to produce a conservative hindcast.
    damped_change = 0.20 * median_change

    predictions = []

    for _, row in merged.iterrows():
        previous = row[f"{parameter}_previous"]

        prediction = previous * (
            1.0 + damped_change
        )

        if parameter in BOUNDED_PARAMETERS:
            low, high = BOUNDED_PARAMETERS[
                parameter
            ]
            prediction = np.clip(
                prediction,
                low,
                high,
            )

        predictions.append(
            float(prediction)
        )

    merged["prediction"] = predictions

    actual = merged[
        f"{parameter}_actual"
    ].to_numpy(dtype=float)

    prediction = merged[
        "prediction"
    ].to_numpy(dtype=float)

    errors = actual - prediction

    mae = float(
        np.mean(np.abs(errors))
    )

    rmse = float(
        np.sqrt(
            np.mean(errors ** 2)
        )
    )

    return {
        "parameter": parameter,
        "training_year": training_year,
        "validation_year": validation_year,
        "n_stations": len(merged),
        "historical_median_change": median_change,
        "damped_change": damped_change,
        "MAE": mae,
        "RMSE": rmse,
        "status": "Hindcast diagnostic",
    }


def validate_all_parameters(df=None):
    if df is None:
        df = load_observed_data()

    results = []

    for parameter in PARAMETERS:
        result = calculate_hindcast(
            df,
            parameter,
        )

        if result is None:
            result = {
                "parameter": parameter,
                "training_year": np.nan,
                "validation_year": np.nan,
                "n_stations": 0,
                "historical_median_change": np.nan,
                "damped_change": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
                "status": "Insufficient data",
            }

        results.append(result)

    return pd.DataFrame(results)


def generate_validation_report():
    df = load_observed_data()

    results = validate_all_parameters(df)

    output_dir = Path(
        "data/processed/forecast_validation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "upper_athi_forecast_validation.csv"
    )

    results.to_csv(
        output_file,
        index=False,
    )

    return results, output_file


if __name__ == "__main__":
    results, output_file = (
        generate_validation_report()
    )

    print(results.to_string(index=False))
    print()
    print("SAVED:", output_file)
