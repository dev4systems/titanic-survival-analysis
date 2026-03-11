# Titanic Survival Analysis

End-to-end Titanic survival prediction with EDA, ML pipeline, and visualizations.

## Project Structure

```
titanic-survival-analysis-2/
├── data/
│   ├── train.csv          # Training data (891 samples)
│   └── test.csv           # Test data (418 samples)
├── src/
│   ├── explore.py         # EDA: univariate & bivariate analysis
│   └── pipeline.py        # Full ML pipeline
├── visualizations/        # Saved plots
├── models/                # Saved model and preprocessing artifacts
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline Overview

The ML pipeline (`src/pipeline.py`) performs:

### 1. Data Preprocessing & Cleaning

- **Missing Age**: Filled with median age by `Pclass` and `Sex`, then global median fallback
- **Missing Embarked**: Filled with mode (most frequent port)
- **Missing Fare**: Filled with median
- **Cabin**: Dropped (77% missing)
- **PassengerId, Name, Ticket**: Dropped after feature extraction

### 2. Feature Engineering

| Feature     | Description                                              |
|------------|----------------------------------------------------------|
| **Title**  | Extracted from Name (Mr, Mrs, Miss, Master, Rare)        |
| **FamilySize** | SibSp + Parch + 1                                    |
| **IsAlone**    | 1 if FamilySize == 1, else 0                         |

Categorical variables (Sex, Embarked, Title) are label-encoded.  
Numerical features are standardized with `StandardScaler`.

### 3. Model Training

Three models are trained and compared:

- **Logistic Regression**
- **Random Forest** (100 trees)
- **Gradient Boosting** (100 estimators)

An 80/20 stratified train/validation split is used.

### 4. Model Evaluation

Metrics computed for each model:

- **Accuracy** – overall correctness
- **Precision** – among predicted survivors, fraction who actually survived
- **Recall** – among actual survivors, fraction correctly predicted
- **F1 Score** – harmonic mean of precision and recall
- **ROC-AUC** – area under ROC curve

### 5. Visualizations

- `01_univariate_analysis.png` – from EDA (run `explore.py`)
- `02_bivariate_analysis.png` – from EDA (run `explore.py`)
- `03_confusion_matrices.png` – confusion matrices for each model
- `04_roc_curves.png` – ROC curves for model comparison
- `05_metrics_comparison.png` – bar chart of metrics across models

### 6. Model Persistence

The best model (by F1 score, then ROC-AUC) is saved along with preprocessing artifacts:

- `models/best_model.joblib` – trained model
- `models/encoders.joblib` – label encoders
- `models/scaler.joblib` – StandardScaler
- `models/feature_names.joblib` – feature column names

Test set predictions are written to `predictions.csv`.

## Usage

**Run EDA (creates univariate and bivariate plots):**

```bash
cd src
python explore.py
```

*Note: `explore.py` expects a `visualizations/` folder. Update the data path if needed.*

**Run the full ML pipeline:**

```bash
cd src
python pipeline.py
```

Or from the project root:

```bash
python src/pipeline.py
```

## Dependencies

- `pandas` – data manipulation
- `numpy` – numerical operations
- `scikit-learn` – preprocessing, models, evaluation
- `matplotlib` – plotting
- `seaborn` – statistical visualizations
- `joblib` – model serialization
