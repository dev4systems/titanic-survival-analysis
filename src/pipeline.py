"""
Titanic Survival Prediction - Full ML Pipeline
==============================================
Data preprocessing, feature engineering, model training, evaluation, and visualization.
"""

import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
try:
    # When running from within src/ (python pipeline.py)
    from feature_importance import plot_feature_importance
except ImportError:
    # When running from project root (python src/pipeline.py)
    from src.feature_importance import plot_feature_importance
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VIS_DIR = PROJECT_ROOT / "visualizations"
MODEL_DIR = PROJECT_ROOT / "models"

VIS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# =============================================================================
# 1. DATA PREPROCESSING AND CLEANING
# =============================================================================


def load_data():
    """Load train and test datasets."""
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path.exists() else None
    return train, test


def extract_title(name):
    """Extract title from passenger name (Mr, Mrs, Miss, Master, etc.)."""
    match = re.search(r" ([A-Za-z]+)\.", name)
    return match.group(1) if match else "Unknown"


def preprocess(df, is_train=True, encoders=None, scaler=None):
    """
    Preprocess and clean the dataset.
    - Fill missing Age with median by Pclass and Sex
    - Fill missing Embarked with mode
    - Fill missing Fare with median
    - Drop Cabin (too many missing), PassengerId, Ticket, Name (after title extraction)
    - Encode categorical variables
    """
    df = df.copy()

    # Title from Name
    df["Title"] = df["Name"].apply(extract_title)
    title_map = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Rare",
        "Rev": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Mlle": "Miss",
        "Countess": "Rare",
        "Ms": "Miss",
        "Lady": "Rare",
        "Jonkheer": "Rare",
        "Don": "Rare",
        "Dona": "Rare",
        "Mme": "Mrs",
        "Capt": "Rare",
        "Sir": "Rare",
    }
    df["Title"] = df["Title"].map(lambda x: title_map.get(x, "Rare"))

    # FamilySize and IsAlone
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Fill Age: median by Pclass and Sex
    if df["Age"].isnull().any():
        age_medians = df.groupby(["Pclass", "Sex"])["Age"].transform("median")
        df["Age"] = df["Age"].fillna(age_medians)
    # Fallback: global median
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Fill Embarked with mode
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Fill Fare (test set has 1 missing)
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Encode Sex
    if encoders is None:
        encoders = {}
    if "sex" not in encoders:
        encoders["sex"] = LabelEncoder()
        df["Sex"] = encoders["sex"].fit_transform(df["Sex"])
    else:
        df["Sex"] = encoders["sex"].transform(df["Sex"])

    # Encode Embarked
    if "embarked" not in encoders:
        encoders["embarked"] = LabelEncoder()
        df["Embarked"] = encoders["embarked"].fit_transform(df["Embarked"].astype(str))
    else:
        df["Embarked"] = encoders["embarked"].transform(df["Embarked"].astype(str))

    # Encode Title
    if "title" not in encoders:
        encoders["title"] = LabelEncoder()
        df["Title"] = encoders["title"].fit_transform(df["Title"].astype(str))
    else:
        # Handle unseen titles in test - map to first known class
        classes = list(encoders["title"].classes_)
        df["Title"] = df["Title"].apply(
            lambda x: x if str(x) in classes else classes[0]
        )
        df["Title"] = encoders["title"].transform(df["Title"].astype(str))

    # Drop columns
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Select features for modeling
    feature_cols = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]
    X = df[[c for c in feature_cols if c in df.columns]]

    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )
    else:
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )

    if is_train and "Survived" in df.columns:
        y = df["Survived"]
        return X_scaled, y, encoders, scaler
    return X_scaled, encoders, scaler


# =============================================================================
# 2. MODEL TRAINING AND EVALUATION
# =============================================================================


def train_and_evaluate_models(X_train, X_val, y_train, y_val, X_full, y_full):
    """Train Logistic Regression, Random Forest, Gradient Boosting and evaluate.
    Uses GridSearchCV for RF and GB; 5-fold CV accuracy for all models.
    """
    # Base models and grids for tuning
    rf_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    }
    gb_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    }

    models_config = [
        (
            "Logistic Regression",
            LogisticRegression(max_iter=1000, random_state=42),
            None,
        ),
        (
            "Random Forest",
            RandomForestClassifier(random_state=42),
            rf_grid,
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=42),
            gb_grid,
        ),
    ]

    results = []
    trained = {}

    for name, model, grid in models_config:
        if grid is not None:
            grid_search = GridSearchCV(
                model, grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters ({name}): {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

        # 5-fold cross-validation accuracy (on full training data for stability)
        cv_scores = cross_val_score(model, X_full, y_full, cv=5)
        cv_mean = cv_scores.mean()
        print(f"Cross Validation Accuracy: {cv_mean:.4f}")

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_proba) if y_proba is not None else 0

        results.append(
            {
                "model": name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc,
                "cv_accuracy": cv_mean,
                "y_pred": y_pred,
                "y_proba": y_proba,
            }
        )
        trained[name] = model

    return results, trained


# =============================================================================
# 3. VISUALIZATIONS
# =============================================================================


def plot_correlation_heatmap(train_df, save_path):
    """Plot correlation heatmap for numeric columns of training data (train.corr())."""
    # Use only numeric columns for correlation (same as train.corr() on mixed DataFrame)
    corr_matrix = train_df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_survival_gender(train_df, save_path):
    """Plot survival rate by gender."""
    plt.figure(figsize=(6, 5))
    sns.barplot(x="Sex", y="Survived", data=train_df)
    plt.title("Survival Rate by Gender")
    plt.ylabel("Survived (mean)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_survival_pclass(train_df, save_path):
    """Plot survival rate by passenger class."""
    plt.figure(figsize=(6, 5))
    sns.barplot(x="Pclass", y="Survived", data=train_df)
    plt.title("Survival Rate by Passenger Class")
    plt.ylabel("Survived (mean)")
    plt.xlabel("Passenger Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(results, y_val, save_path):
    """Plot confusion matrices for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = confusion_matrix(y_val, r["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Died", "Survived"],
            yticklabels=["Died", "Survived"],
        )
        ax.set_title(f"{r['model']}\nAccuracy: {r['accuracy']:.3f}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(results, y_val, save_path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(8, 6))
    for r in results:
        if r["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_val, r["y_proba"])
            plt.plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(results, save_path):
    """Bar chart comparing metrics across models."""
    df_res = pd.DataFrame(
        [
            {
                "Model": r["model"],
                "Accuracy": r["accuracy"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1 Score": r["f1_score"],
                "ROC-AUC": r["roc_auc"],
            }
            for r in results
        ]
    )
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    x = np.arange(len(df_res["Model"]))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, df_res[m], width, label=m)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(df_res["Model"], rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================


def run_pipeline():
    """Run the full ML pipeline."""
    print("=" * 60)
    print("TITANIC SURVIVAL PREDICTION - ML PIPELINE")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    train, test = load_data()
    print(f"   Train: {train.shape}, Test: {test.shape if test is not None else 'N/A'}")

    # Preprocess train
    print("\n2. Preprocessing and feature engineering...")
    X, y, encoders, scaler = preprocess(train, is_train=True)
    feature_names = list(X.columns)
    print(f"   Features: {feature_names}")

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)}, Validation: {len(X_val)}")

    # Train and evaluate models
    print("\n3. Training models...")
    results, trained = train_and_evaluate_models(
        X_train, X_val, y_train, y_val, X, y
    )

    # Print evaluation
    print("\n4. Model Evaluation:")
    print("-" * 60)
    for r in results:
        print(f"\n{r['model']}:")
        print(f"  Accuracy:  {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall:    {r['recall']:.4f}")
        print(f"  F1 Score:  {r['f1_score']:.4f}")
        print(f"  ROC-AUC:   {r['roc_auc']:.4f}")
        print(f"  CV Accuracy: {r['cv_accuracy']:.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_val, r["y_pred"], target_names=["Died", "Survived"]))

    # Model performance summary table
    summary_df = pd.DataFrame(
        [
            {
                "Model": r["model"],
                "Accuracy": round(r["accuracy"], 4),
                "Precision": round(r["precision"], 4),
                "Recall": round(r["recall"], 4),
                "F1 Score": round(r["f1_score"], 4),
                "ROC-AUC": round(r["roc_auc"], 4),
                "CV Accuracy": round(r["cv_accuracy"], 4),
            }
            for r in results
        ]
    )
    print("\n5. Model Performance Summary:")
    print(summary_df.to_string(index=False))
    summary_path = VIS_DIR / "model_performance_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   Summary saved to {summary_path}")

    # Select best model (by F1 score, then ROC-AUC)
    best = max(results, key=lambda r: (r["f1_score"], r["roc_auc"]))
    best_model = trained[best["model"]]
    print(f"\n>>> Best model: {best['model']} (F1={best['f1_score']:.4f}, AUC={best['roc_auc']:.4f})")

    # Additional visualizations (raw train data)
    print("\n6. Saving additional visualizations...")
    plot_correlation_heatmap(train, VIS_DIR / "correlation_heatmap.png")
    plot_survival_gender(train, VIS_DIR / "survival_gender.png")
    plot_survival_pclass(train, VIS_DIR / "survival_pclass.png")
    print("   Saved: correlation_heatmap.png, survival_gender.png, survival_pclass.png")

    # Feature Importance Visualization
    print("\n7. Generating feature importance visualization...")
    plot_feature_importance(feature_names)
    print("   Feature importance plot saved to visualizations/feature_importance.png")

    # Model comparison visualizations
    print("\n8. Saving model comparison visualizations...")
    plot_confusion_matrices(results, y_val, VIS_DIR / "confusion_matrices.png")
    plot_roc_curves(results, y_val, VIS_DIR / "roc_curves.png")
    plot_metrics_comparison(results, VIS_DIR / "metrics_comparison.png")
    print("   Saved: confusion_matrices.png, roc_curves.png, metrics_comparison.png")

    # Save best model and artifacts
    print("\n9. Saving best model and preprocessing artifacts...")
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    joblib.dump(encoders, MODEL_DIR / "encoders.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.joblib")
    print(f"   Best model: {best['model']} saved to models/best_model.joblib")

    # Optional: generate test predictions
    if test is not None and len(test) > 0:
        print("\n10. Generating test set predictions...")
        X_test, _, _ = preprocess(test, is_train=False, encoders=encoders, scaler=scaler)
        X_test = X_test[feature_names]  # Ensure same order
        preds = best_model.predict(X_test)
        submission = pd.DataFrame(
            {"PassengerId": test["PassengerId"], "Survived": preds}
        )
        sub_path = PROJECT_ROOT / "predictions.csv"
        submission.to_csv(sub_path, index=False)
        print(f"   Predictions saved to {sub_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
