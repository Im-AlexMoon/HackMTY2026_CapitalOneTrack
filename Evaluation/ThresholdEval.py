import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
DATA_DIR = "./EDA/data"
OUT_DIR = "./EDA/data/outputs"
EVAL_DIR = os.path.join(OUT_DIR, "evaluation")

TARGET = "fraud_bool"
TIME_COL = "month"
TARGET_FPR = 0.05

file_paths = [
    f"{DATA_DIR}/Base.csv",
    f"{DATA_DIR}/Variant I.csv",
    f"{DATA_DIR}/Variant II.csv",
    f"{DATA_DIR}/Variant III.csv",
    f"{DATA_DIR}/Variant IV.csv",
    f"{DATA_DIR}/Variant V.csv",
]
names = ["base", "groupsize_diff", "prev_diff", "sep", "train_prev_diff", "train_sep"]
split_names = ["random", "Random_w_time", "temporal"]

MISSING_SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]


def load_and_preprocess(path):
    df = pd.read_csv(path)
    for col in MISSING_SENTINEL_COLS:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)
    return df


def get_test_split(df, split_type):
    """Replica exactamente las particiones de test de DatasetExploratoryAnalysis_3.py"""
    if split_type == "temporal":
        # Meses 6 y 7 corresponden al test set temporal explícito del benchmark
        df_test = df[df[TIME_COL].isin([6, 7])].reset_index(drop=True)
    else:
        # Split aleatorio estratificado para 'random' y 'Random_w_time'
        _, df_test = train_test_split(
            df, test_size=0.2, stratify=df[TARGET], random_state=42
        )
    return df_test


def find_optimal_threshold_at_fpr(y_true, y_proba, target_fpr=0.05):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return thresholds[idx], fpr[idx], tpr[idx]


def plot_variant_analysis(y_true, y_proba, full_tag, opt_thresh, opt_fpr, opt_tpr, eval_dir):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"XGBoost {full_tag} (AUC = {auc_val:.4f})", color="navy", lw=2)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.scatter(
        [opt_fpr], [opt_tpr], color="red", s=100, zorder=5,
        label=f"Punto Óptimo (p={opt_thresh:.4f})\nFPR={opt_fpr:.2%}, Recall={opt_tpr:.2%}"
    )
    plt.axvline(x=TARGET_FPR, color="red", linestyle=":", alpha=0.7, label=f"Límite FPR ({TARGET_FPR:.0%})")
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
    plt.title(f"Curva ROC - [{full_tag}]")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"roc_curve_{full_tag}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(x=y_proba[y_true == 0], bins=50, color="blue", stat="density", alpha=0.4, label="Legítimo (0)")
    sns.histplot(x=y_proba[y_true == 1], bins=50, color="red", stat="density", alpha=0.5, label="Fraude (1)")
    plt.axvline(x=0.5, color="black", linestyle="--", label="Umbral 0.5")
    plt.axvline(x=opt_thresh, color="red", linestyle="-", lw=2, label=f"Umbral óptimo ({opt_thresh:.4f})")
    plt.yscale("log")
    plt.xlabel("Probabilidad Predicha")
    plt.ylabel("Densidad (log)")
    plt.title(f"Distribución Probabilidades - [{full_tag}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"prob_dist_{full_tag}.png"), dpi=300)
    plt.close()


def plot_combined_roc(roc_data_dict, eval_dir):
    plt.figure(figsize=(12, 9))
    colors = sns.color_palette("tab20", n_colors=len(roc_data_dict))
    
    for (full_tag, data), color in zip(roc_data_dict.items(), colors):
        plt.plot(data["fpr"], data["tpr"], label=f"{full_tag} (AUC={data['auc']:.3f})", lw=1.5, color=color)

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.axvline(x=TARGET_FPR, color="red", linestyle=":", alpha=0.8, label=f"Límite FPR Target ({TARGET_FPR:.0%})")
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
    plt.title("Comparación General de Curvas ROC (Variantes y Splits)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "combined_roc_curves.png"), dpi=300)
    plt.close()


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    all_summary_results = []
    roc_curves_data = {}

    print("=" * 80)
    print(" EVALUACIÓN DE SENSIBILIDAD POR MODELO Y TIPO DE SPLIT")
    print("=" * 80)

    for path, tag in zip(file_paths, names):
        df = load_and_preprocess(path)

        for name in split_names:
            full_tag = f"{name}_{tag}"
            # Corrección del nombre del archivo (guion bajo para coincidir con el script de entrenamiento)
            bundle_path = os.path.join(OUT_DIR, f"model_bundle_{name}_{tag}.joblib")
            
            if not os.path.exists(bundle_path):
                print(f"\n[!] Modelo no encontrado: {bundle_path}. Saltando.")
                continue

            print(f"\n---> Evaluando: [{full_tag}]")
            
            bundle = joblib.load(bundle_path)
            preprocessor = bundle["preprocessor"]
            selector = bundle["selector"]
            model = bundle["model"]
            feature_cols = bundle["feature_cols"].copy()

            df_test = get_test_split(df, name)

            # Para splits sin columna de tiempo, se excluye explícitamente si viniera en df_test
            if name in ["random", "temporal"] and TIME_COL in df_test.columns:
                df_test = df_test.drop(columns=[TIME_COL])

            X_test = df_test[feature_cols]
            y_test = df_test[TARGET]

            X_test_enc = preprocessor.transform(X_test)
            X_test_sel = selector.transform(X_test_enc)
            y_proba = model.predict_proba(X_test_sel)[:, 1]

            opt_thresh, actual_fpr, actual_tpr = find_optimal_threshold_at_fpr(
                y_test, y_proba, target_fpr=TARGET_FPR
            )

            pred_default = (y_proba >= 0.5).astype(int)
            pred_optimo = (y_proba >= opt_thresh).astype(int)

            fpr_def = confusion_matrix(y_test, pred_default)[0, 1] / max((y_test == 0).sum(), 1)
            rec_def = recall_score(y_test, pred_default)
            prec_def = precision_score(y_test, pred_default, zero_division=0)

            rec_opt = actual_tpr
            prec_opt = precision_score(y_test, pred_optimo, zero_division=0)
            f1_opt = f1_score(y_test, pred_optimo, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)

            all_summary_results.append({
                "variante": tag,
                "split": name,
                "roc_auc": roc_auc,
                "p_optimo": opt_thresh,
                "fpr_actual": actual_fpr,
                "recall_optimo_5%FPR": rec_opt,
                "precision_optima": prec_opt,
                "f1_optimo": f1_opt,
                "recall_defecto_0.5": rec_def,
                "precision_defecto_0.5": prec_def,
                "fpr_defecto_0.5": fpr_def,
            })

            fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)
            roc_curves_data[full_tag] = {"fpr": fpr_arr, "tpr": tpr_arr, "auc": roc_auc}

            plot_variant_analysis(y_test, y_proba, full_tag, opt_thresh, actual_fpr, actual_tpr, EVAL_DIR)
            print(f"  ✓ ROC-AUC: {roc_auc:.4f} | Recall@5%FPR: {rec_opt:.2%}")

    # Generar gráfico comparativo general
    if roc_curves_data:
        plot_combined_roc(roc_curves_data, EVAL_DIR)
        print(f"\n[+] Gráfica comparativa guardada en: {os.path.join(EVAL_DIR, 'combined_roc_curves.png')}")

    if all_summary_results:
        summary_df = pd.DataFrame(all_summary_results)
        csv_out = os.path.join(EVAL_DIR, "summary_sensitivity_analysis.csv")
        summary_df.to_csv(csv_out, index=False)
        print(f"[+] Tabla resumen exportada exitosamente a: {csv_out}")


if __name__ == "__main__":
    main()