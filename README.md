# ML-SWQM-DSS

## Machine Learning-Based Surface Water Quality Monitoring and Decision Support System

ML-SWQM-DSS is a machine-learning-based Surface Water Quality Monitoring and Decision Support System developed for surface-water quality assessment, environmental risk screening, and decision support.

The current implementation is focused on the **Upper Athi River Catchment in Kenya**, while maintaining a major application focus on environmental conditions associated with **petroleum exploration, production, transportation, storage, terminals, spills, and related activities**.

The system combines water-quality measurements, Water Quality Index (WQI) assessment, machine-learning prediction, risk classification, petroleum-event screening, station investigation, and reporting within an interactive Streamlit application.

---

## Project Objectives

The main objectives of ML-SWQM-DSS are to:

* Assess surface-water quality using measured water-quality parameters.
* Calculate a screening Water Quality Index (WQI).
* Apply machine learning to estimate water-quality conditions.
* Classify water-quality conditions into screening risk categories.
* Support identification of locations requiring further investigation.
* Provide petroleum-related environmental risk screening.
* Support environmental monitoring and petroleum engineering decision-making.
* Present water-quality information through an interactive dashboard.
* Generate water-quality assessment reports.

---

## Current Case Study

The current case study is the **Upper Athi River Catchment (UARC), Kenya**.

The system is informed by Upper Athi River water-quality research and monitoring data covering multiple sampling locations and water-quality parameters.

The Upper Athi River Catchment provides the environmental monitoring context, while petroleum-related activities provide an important application and decision-support context.

The system is intended to demonstrate how machine learning and water-quality indices can support environmental monitoring and petroleum-related decision-making.

---

## Petroleum Environmental Decision-Support Concept

The petroleum component follows the following conceptual pathway:

```text
Petroleum Activity
        ↓
Environmental Pathway
        ↓
Surface-Water Receptor
        ↓
Water-Quality Monitoring
        ↓
ML-SWQM-DSS
        ↓
WQI + Machine-Learning Prediction
        ↓
Risk Classification
        ↓
Petroleum / Environmental Decision
```

Potential petroleum activities considered include:

* Exploration
* Production
* Pipeline transportation
* Petroleum storage
* Petroleum terminals
* Loading and transfer operations
* Accidental spills and leaks

Potential environmental pathways include:

* Surface runoff
* Drainage systems
* Accidental releases
* Leakage
* Spill transport
* Contaminated soil and sediment
* Industrial discharge

The system can therefore be used as a screening-level decision-support tool for identifying water-quality conditions that may require additional field investigation or environmental assessment.

---

## Water-Quality Variables

The current machine-learning model uses the following ten water-quality variables:

* pH
* Dissolved oxygen
* Turbidity
* Temperature
* Electrical conductivity
* Total dissolved solids (TDS)
* Total suspended solids (TSS)
* Total nitrogen
* Total phosphorus
* Chlorophyll-a

Additional water-quality parameters may be available in the monitoring dataset for exploration, analysis, and future model development.

---

## Water Quality Index

The application calculates a screening **Water Quality Index (WQI)** using weighted water-quality parameters.

The current WQI interpretation is:

|      WQI | Classification |
| -------: | -------------- |
|   90-100 | Excellent      |
|    70-89 | Good           |
|    50-69 | Medium         |
|    25-49 | Poor           |
| Below 25 | Very Poor      |

The WQI is intended for screening and decision-support purposes. It should not be interpreted as a substitute for laboratory analysis, regulatory assessment, or professional environmental evaluation.

---

## Screening Risk Classification

The application also provides screening risk categories based on the calculated WQI.

The current screening interpretation is:

|             WQI | Screening Risk |
| --------------: | -------------- |
| Greater than 70 | Low            |
|           50-70 | Medium         |
|        Below 50 | High           |

Screening risk is intended to help identify water-quality conditions that may require additional attention or investigation.

---

## Machine Learning

The current deployed application uses a **Random Forest regression model** to estimate the Water Quality Index from selected water-quality variables.

The trained machine-learning artifacts are stored in the `models` directory.

```text
models/
├── athi_model_features.pkl
├── athi_model_metadata.pkl
└── athi_regressor.pkl
```

### Model files

`athi_model_features.pkl`

Stores the feature configuration required by the current Upper Athi model.

`athi_model_metadata.pkl`

Stores model metadata and supporting information associated with the trained model.

`athi_regressor.pkl`

Stores the trained Random Forest regression model used by the application.

The current deployment does **not** depend on the following legacy model files:

```text
model_features.pkl
saved_classifier.pkl
saved_regressor.pkl
scaler.pkl
```

Those files belong to the previous model architecture and have been removed from the active deployment structure.

---

## Current Processed Data

The current deployment uses the following processed datasets:

```text
data/processed/
├── athi_feature_importance.csv
├── athi_model_predictions.csv
└── upper_athi_water_quality.csv
```

### `upper_athi_water_quality.csv`

Contains the processed Upper Athi River water-quality dataset used by the current application.

### `athi_model_predictions.csv`

Contains model prediction outputs generated from the Upper Athi model workflow.

### `athi_feature_importance.csv`

Contains feature-importance information associated with the trained machine-learning model.

---

## Application Features

The Streamlit application provides functionality for:

* Water-quality dashboard
* Data exploration
* Water-quality parameter analysis
* WQI calculation
* Machine-learning WQI prediction
* Screening risk classification
* Petroleum-related event screening
* Station comparison
* Station investigation
* Station-priority assessment
* Petroleum environmental decision support
* Interactive charts
* Model feature-importance analysis
* PDF report generation

---

## Water-Quality Screening Criteria

The application includes screening criteria for selected parameters.

Current screening criteria include:

| Parameter        | Screening Criterion |
| ---------------- | ------------------- |
| pH               | 6.5-8.5             |
| Dissolved oxygen | ≥ 5                 |
| Turbidity        | ≤ 5 NTU             |
| Temperature      | 15-30 °C            |
| TSS              | ≤ 500               |
| Total nitrogen   | ≤ 10                |
| Total phosphorus | ≤ 1                 |

These values are used as screening indicators within the application and should be interpreted alongside applicable Kenyan environmental standards, field measurements, laboratory results, and professional environmental assessment.

---

## Petroleum Event Screening

The application includes functionality for examining potential petroleum-related water-quality events.

The petroleum screening component is designed to support assessment of conditions associated with:

* Petroleum spills
* Petroleum leaks
* Pipeline-related incidents
* Storage and terminal activities
* Petroleum handling and transfer
* Environmental monitoring near petroleum-related infrastructure

The screening output is intended to identify conditions that may warrant additional investigation.

It does not independently establish that a petroleum release has occurred.

---

## Station Investigation

The system supports comparison and investigation of monitoring locations.

Station-level analysis can assist with:

* Identifying locations with poorer water-quality conditions
* Comparing WQI values between stations
* Reviewing parameter conditions
* Identifying locations requiring further investigation
* Supporting environmental monitoring priorities

Station-priority outputs should be treated as screening recommendations rather than definitive environmental conclusions.

---

## Project Structure

The active deployment structure is:

```text
ML-SWQM-DSS/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── athi_model_features.pkl
│   ├── athi_model_metadata.pkl
│   └── athi_regressor.pkl
│
├── data/
│   └── processed/
│       ├── athi_feature_importance.csv
│       ├── athi_model_predictions.csv
│       └── upper_athi_water_quality.csv
│
└── scripts/
    ├── build_athi_dataset.py
    ├── diagnose_model_provenance.py
    ├── inspect_athi_tables.py
    ├── train_athi_model.py
    ├── validate_athi_dss.py
    ├── validate_maziba_indices.py
    ├── validate_petroleum_event_stress.py
    ├── validate_petroleum_events.py
    └── validate_station_priority.py
```

Research archives, source documents, large research datasets, backups, and historical application versions are maintained separately from the active deployment package.

---

## Development Technologies

The application is developed using:

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Joblib
* ReportLab

---

## Requirements

The current deployment dependencies are pinned in `requirements.txt`.

```text
streamlit==1.61.1
pandas==2.3.3
numpy==2.5.1
plotly==6.9.0
joblib==1.5.3
scikit-learn==1.9.0
reportlab==5.0.0
```

Pinning package versions helps maintain a consistent deployment environment.

---

## Installation

### 1. Clone the repository

```bash
git clone YOUR-REPOSITORY-URL
cd ML-SWQM-DSS
```

Replace `YOUR-REPOSITORY-URL` with the actual GitHub repository URL.

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the application

```powershell
python -m streamlit run .\app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## Local Network Access

When Streamlit is running on a Windows computer, the application can also be accessed by compatible devices on the same local network using the computer's network address.

For example:

```text
http://192.168.100.8:8501
```

The actual network address may differ depending on the user's network configuration.

---

## Cross-Platform Use

ML-SWQM-DSS is designed as a responsive web application.

The same application can therefore be accessed through a modern web browser on:

* Windows computers
* Android devices
* iPhone/iOS devices
* Tablets
* Other compatible devices

This approach avoids requiring a separate native application for each operating system.

A cloud deployment can provide remote access without requiring Python to be installed on the end user's device.

---

## Deployment

The application is suitable for deployment using a Streamlit-compatible hosting platform.

The deployment package should contain the active application, required model artifacts, processed deployment data, and `requirements.txt`.

The following should not be included in the deployment package:

* `.venv`
* Development backups
* Historical application versions
* Research archives
* Unnecessary large source files
* Legacy model artifacts
* Local environment files
* Temporary files

The `.gitignore` file is used to prevent excluded files from being committed to the repository.

---

## Data and Research Sources

The project uses processed Upper Athi River water-quality information for the current deployment.

Research source materials and datasets are maintained separately from the active application where appropriate.

Large research files may be excluded from Git because of repository size limitations while remaining available locally for research and model-development purposes.

---

## Model Limitations

The current Upper Athi dataset is relatively limited compared with the amount of data normally required for a fully operational environmental forecasting system.

Therefore, the current machine-learning model should be considered an **academic and exploratory decision-support model** rather than a fully validated operational forecasting system.

Model outputs should be interpreted together with:

* Field measurements
* Laboratory analysis
* Hydrological information
* Meteorological information
* Regulatory requirements
* Professional environmental assessment
* Site-specific petroleum activity information

Future model development should incorporate larger and more representative datasets.

---

## Future Development

Potential future improvements include:

* Additional Upper Athi River monitoring observations
* More frequent temporal observations
* Expanded monitoring-station coverage
* Meteorological variables
* Hydrological variables
* Rainfall and runoff information
* Petroleum spill and incident records
* Petroleum facility proximity information
* Total petroleum hydrocarbons (TPH)
* BTEX compounds
* PAHs
* Oil and grease measurements
* Additional heavy metals
* Improved temporal forecasting
* Model validation using independent datasets
* Automated monitoring-data ingestion
* Online database integration
* Offline data collection
* Mobile/PWA enhancements
* Expanded environmental decision-support capabilities

---

## Environmental Decision Support

ML-SWQM-DSS is designed to support, rather than replace, professional environmental and engineering judgment.

A simplified decision-support process is:

```text
Measure Water Quality
        ↓
Validate Data
        ↓
Calculate WQI
        ↓
Run ML Prediction
        ↓
Determine Screening Risk
        ↓
Investigate Station / Event
        ↓
Assess Petroleum or Environmental Context
        ↓
Determine Monitoring or Engineering Action
```

The final decision should consider the complete environmental and operational context.

---

## Academic Project

ML-SWQM-DSS is developed as an academic project focused on the application of machine learning, water-quality assessment, environmental monitoring, and decision-support concepts.

The current implementation focuses on the **Upper Athi River Catchment, Kenya**, with petroleum-related environmental monitoring and decision support forming a major application component.

The system demonstrates how measured water-quality information can be integrated with WQI calculations, machine-learning predictions, risk screening, and petroleum-related environmental assessment within a single decision-support platform.

---

## Project Status

The current version contains:

* The operational Streamlit application
* Upper Athi River processed deployment data
* Trained Upper Athi machine-learning artifacts
* Model feature information
* Model metadata
* Feature-importance results
* Model prediction results
* Validation and analysis scripts
* PDF reporting functionality
* Deployment requirements
* Git deployment configuration

The application is currently being prepared for cloud deployment and cross-platform browser access.
