# ML-SWQM-DSS

## Machine Learning-Based Surface Water Quality Monitoring and Decision Support System

ML-SWQM-DSS is a machine-learning-based decision support application designed to support surface-water quality monitoring, prediction, water-quality assessment, and risk classification.

The system is developed with a focus on environmental monitoring in areas potentially affected by petroleum exploration, production, transportation, storage, and related activities.

## Key Features

* Interactive surface-water quality monitoring dashboard
* Machine-learning-based water-quality prediction
* Random Forest classification and regression models
* Water Quality Index (WQI) assessment
* Automated water-quality risk classification
* Interactive data visualizations
* Water-quality trend analysis
* Filtering of monitoring data
* Automated PDF reporting
* Petroleum-environment monitoring focus
* Upper Athi River catchment case-study orientation
* Responsive browser-based interface

## Water Quality Parameters

The system works with water-quality variables including:

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

Additional water-quality parameters may be included where available in the monitoring dataset.

## Water Quality Index

The application calculates a screening Water Quality Index using weighted water-quality parameters.

The current WQI interpretation is:

|      WQI | Classification |
| -------: | -------------- |
|   90–100 | Excellent      |
|    70–89 | Good           |
|    50–69 | Medium         |
|    25–49 | Poor           |
| Below 25 | Very Poor      |

The system also provides screening risk categories based on the calculated WQI.

## Machine Learning

The deployed application uses trained machine-learning artifacts stored in the `models` directory.

```text
models/
├── model_features.pkl
├── saved_classifier.pkl
├── saved_regressor.pkl
└── scaler.pkl
```

These files are required by the deployed application.

## Project Structure

```text
ML-SWQM-DSS/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── models/
│   ├── model_features.pkl
│   ├── saved_classifier.pkl
│   ├── saved_regressor.pkl
│   └── scaler.pkl
└── data/
    └── processed/
        └── cleaned_data.csv
```

Research datasets and source materials are maintained separately from the deployment package.

## Installation

### 1. Clone the repository

```bash
git clone YOUR-REPOSITORY-URL
cd ML-SWQM-DSS
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the application

```powershell
python -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## Deployment

ML-SWQM-DSS is designed as a responsive web application so that the same application can be accessed from:

* Windows computers
* Android devices
* iPhone/iOS devices
* Other modern web browsers

A cloud Streamlit deployment can provide remote access without requiring Python to be installed on the user's device.

## Case Study

The project is oriented toward assessment of surface-water quality in the Upper Athi River catchment in Kenya, with particular attention to environmental conditions associated with petroleum-related activities.

The system supports monitoring and decision-making by combining measured water-quality parameters, WQI assessment, machine-learning predictions, and risk classification.

## Environmental and Petroleum Application

Potential applications include:

* Baseline surface-water quality assessment
* Monitoring near petroleum-related activities
* Detection of deteriorating water-quality conditions
* Environmental risk screening
* Support for field monitoring decisions
* Identification of locations requiring further investigation
* Reporting of water-quality conditions over time

The system is intended as a decision-support and screening tool. Model outputs should be interpreted alongside field measurements, laboratory analysis, regulatory requirements, and professional environmental assessment.

## Development Technologies

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Joblib
* ReportLab

## Project Status

The current version contains the operational Streamlit application, trained machine-learning artifacts, processed deployment dataset, and required Python dependencies.

## Academic Project

ML-SWQM-DSS is developed as an academic project with a focus on applying machine learning and environmental monitoring concepts to surface-water quality assessment within petroleum-related environments.
