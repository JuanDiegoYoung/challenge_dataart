from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


OUTPUT_DIR = Path("challente/model_performance")
TOP_10_FEATURES = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]


def build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv("data/data.csv")
    fecha_o = pd.to_datetime(data["Fecha-O"])
    fecha_i = pd.to_datetime(data["Fecha-I"])
    data["min_diff"] = (fecha_o - fecha_i).dt.total_seconds() / 60
    data["delay"] = (data["min_diff"] > 15).astype(int)

    features = pd.concat(
        [
            pd.get_dummies(data["OPERA"], prefix="OPERA"),
            pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
            pd.get_dummies(data["MES"], prefix="MES"),
        ],
        axis=1,
    )
    return features, data["delay"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features, target = build_dataset()
    negative_count = int((target == 0).sum())
    positive_count = int((target == 1).sum())
    scale = negative_count / positive_count
    class_weight = {
        1: negative_count / len(target),
        0: positive_count / len(target),
    }

    models = [
        (
            "XGBoost all features",
            xgb.XGBClassifier(
                random_state=1,
                learning_rate=0.01,
                eval_metric="logloss",
            ),
            features,
        ),
        (
            "XGBoost top 10 balanced",
            xgb.XGBClassifier(
                random_state=1,
                learning_rate=0.01,
                scale_pos_weight=scale,
                eval_metric="logloss",
            ),
            features[TOP_10_FEATURES],
        ),
        (
            "XGBoost top 10 unbalanced",
            xgb.XGBClassifier(
                random_state=1,
                learning_rate=0.01,
                eval_metric="logloss",
            ),
            features[TOP_10_FEATURES],
        ),
        (
            "LogReg all features",
            LogisticRegression(max_iter=1000),
            features,
        ),
        (
            "LogReg top 10 balanced",
            LogisticRegression(max_iter=1000, class_weight=class_weight),
            features[TOP_10_FEATURES],
        ),
        (
            "LogReg top 10 unbalanced",
            LogisticRegression(max_iter=1000),
            features[TOP_10_FEATURES],
        ),
    ]

    rows = []
    confusions = {}
    selected_model = None

    for name, model, feature_frame in models:
        x_train, x_test, y_train, y_test = train_test_split(
            feature_frame,
            target,
            test_size=0.33,
            random_state=42,
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        rows.append(
            {
                "model": name,
                "accuracy": report["accuracy"],
                "recall_0": report["0"]["recall"],
                "f1_0": report["0"]["f1-score"],
                "recall_1": report["1"]["recall"],
                "f1_1": report["1"]["f1-score"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
            }
        )

        confusions[name] = confusion_matrix(y_test, predictions)

        if name == "XGBoost top 10 balanced":
            selected_model = model

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    plot_metrics = ["recall_0", "f1_0", "recall_1", "f1_1"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics["model"]))
    bar_width = 0.2

    for offset, metric in enumerate(plot_metrics):
        ax.bar(
            x + (offset - 1.5) * bar_width,
            metrics[metric],
            width=bar_width,
            label=metric,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics["model"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison by class metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison_metrics.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ranked = metrics.sort_values("recall_1", ascending=True)
    ax.barh(ranked["model"], ranked["recall_1"], label="recall_1")
    ax.barh(ranked["model"], ranked["f1_1"], alpha=0.7, label="f1_1")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title("Delay-class performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "delay_class_performance.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (name, matrix) in zip(axes.ravel(), confusions.items()):
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        for row in range(2):
            for col in range(2):
                ax.text(col, row, matrix[row, col], ha="center", va="center")

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85)
    fig.suptitle("Confusion matrices", y=0.98)
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if selected_model is not None:
        importances = pd.Series(
            selected_model.feature_importances_,
            index=TOP_10_FEATURES,
        ).sort_values()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(importances.index, importances.values)
        ax.set_xlabel("Importance")
        ax.set_title("Selected model feature importance")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "selected_model_feature_importance.png", dpi=160)
        plt.close(fig)

    print("Generated files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path)


if __name__ == "__main__":
    main()