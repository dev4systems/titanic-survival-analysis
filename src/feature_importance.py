import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
VIS_DIR = PROJECT_ROOT / "visualizations"

def plot_feature_importance(feature_names):

    model = joblib.load(MODEL_DIR / "best_model.joblib")

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    elif hasattr(model, "coef_"):
        importance = model.coef_[0]

    else:
        print("Model does not support feature importance.")
        return

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10,6))
    plt.barh(df["Feature"], df["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()

    save_path = VIS_DIR / "feature_importance.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Feature importance saved to {save_path}")