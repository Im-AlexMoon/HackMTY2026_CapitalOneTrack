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
# Configuración de Rutas y Variantes
# ------------------------------------------------------------------
DATA_DIR = "./EDA/data"
OUT_DIR = "./EDA/data/outputs"
EVAL_DIR = os.path.join(OUT_DIR, "evaluation")

TARGET = "fraud_bool"
TARGET_FPR = 0.05  # Restricción de negocio: máximo 5% de Falsos Positivos

file_paths = [
    f"{DATA_DIR}/Base.csv",
    f"{DATA_DIR}/Variant I.csv",
    f"{DATA_DIR}/Variant II.csv",
    f"{DATA_DIR}/Variant III.csv",
    f"{DATA_DIR}/Variant IV.csv",
    f"{DATA_DIR}/Variant V.csv",
]
names = ["base", "groupsize_diff", "prev_diff", "sep", "train_prev_diff", "train_sep"]

MISSING_SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]


def load_and_preprocess(path):
    """Carga los datos y reemplaza centinelas -1 por NaN."""
    df = pd.read_csv(path)
    for col in MISSING_SENTINEL_COLS:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)
    return df


def find_optimal_threshold_at_fpr(y_true, y_proba, target_fpr=0.05):
    """Encuentra el umbral p máximo donde FPR <= target_fpr."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return thresholds[idx], fpr[idx], tpr[idx]


def plot_variant_analysis(y_true, y_proba, tag, opt_thresh, opt_fpr, opt_tpr, eval_dir):
    """Genera y guarda los gráficos individuales para una variante."""
    
    # 1. Curva ROC con punto de operación
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"XGBoost {tag} (AUC = {auc_val:.4f})", color="navy", lw=2)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    
    plt.scatter(
        [opt_fpr], [opt_tpr],
        color="red", s=100, zorder=5,
        label=f"Punto Óptimo (p={opt_thresh:.4f})\nFPR={opt_fpr:.2%}, Recall={opt_tpr:.2%}"
    )
    plt.axvline(x=TARGET_FPR, color="red", linestyle=":", alpha=0.7, label=f"Límite FPR ({TARGET_FPR:.0%})")
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
    plt.title(f"Curva ROC y Punto de Operación (FPR = 5%) - [{tag}]")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"roc_curve_{tag}.png"), dpi=300)
    plt.close()

    # 2. Distribución de Probabilidades
    plt.figure(figsize=(9, 5))
    sns.histplot(x=y_proba[y_true == 0], bins=50, color="blue", stat="density", alpha=0.4, label="Legítimo (0)")
    sns.histplot(x=y_proba[y_true == 1], bins=50, color="red", stat="density", alpha=0.5, label="Fraude (1)")
    plt.axvline(x=0.5, color="black", linestyle="--", label="Umbral por defecto (0.5)")
    plt.axvline(x=opt_thresh, color="red", linestyle="-", lw=2, label=f"Umbral óptimo ({opt_thresh:.4f})")
    
    plt.yscale("log")
    plt.xlabel("Probabilidad Predicha de Fraude")
    plt.ylabel("Densidad (escala logarítmica)")
    plt.title(f"Distribución de Probabilidades por Clase - [{tag}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"prob_dist_{tag}.png"), dpi=300)
    plt.close()


def plot_combined_roc(roc_data_dict, eval_dir):
    """Genera una gráfica comparativa con las curvas ROC de todas las variantes."""
    plt.figure(figsize=(10, 8))
    
    colors = sns.color_palette("tab10", n_colors=len(roc_data_dict))
    for (tag, data), color in zip(roc_data_dict.items(), colors):
        fpr, tpr = data["fpr"], data["tpr"]
        auc_score = data["auc"]
        plt.plot(fpr, tpr, label=f"{tag} (AUC = {auc_score:.4f})", lw=2, color=color)

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.axvline(x=TARGET_FPR, color="red", linestyle=":", alpha=0.8, label=f"Límite FPR Target ({TARGET_FPR:.0%})")
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
    plt.title("Comparación de Curvas ROC entre Variantes de BAF")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "combined_roc_curves.png"), dpi=300)
    plt.close()


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    all_summary_results = []
    roc_curves_data = {}

    print("=" * 80)
    print(" EVALUACIÓN DE UMBRALES Y ANÁLISIS DE SENSIBILIDAD POR MODELO")
    print("=" * 80)

    for path, tag in zip(file_paths, names):
        bundle_path = os.path.join(OUT_DIR, f"model_bundle_{tag}.joblib")
        
        if not os.path.exists(bundle_path):
            print(f"\n[¡ADVERTENCIA!] Archivo no encontrado: {bundle_path}. Saltando variante '{tag}'.")
            continue

        print(f"\n---> Procesando variante: [{tag}]")
        
        # 1. Cargar el bundle del modelo guardado
        bundle = joblib.load(bundle_path)
        preprocessor = bundle["preprocessor"]
        selector = bundle["selector"]
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]

        # 2. Cargar datos y replicar el split de test (20% estratificado)
        df = load_and_preprocess(path)
        _, df_test = train_test_split(
            df, test_size=0.2, stratify=df[TARGET], random_state=42
        )

        X_test = df_test[feature_cols]
        y_test = df_test[TARGET]

        # 3. Transformación e inferencia
        X_test_enc = preprocessor.transform(X_test)
        X_test_sel = selector.transform(X_test_enc)
        y_proba = model.predict_proba(X_test_sel)[:, 1]

        # 4. Búsqueda del umbral óptimo a FPR <= 5%
        opt_thresh, actual_fpr, actual_tpr = find_optimal_threshold_at_fpr(
            y_test, y_proba, target_fpr=TARGET_FPR
        )

        # 5. Predicciones bajo ambos umbrales
        pred_default = (y_proba >= 0.5).astype(int)
        pred_optimo = (y_proba >= opt_thresh).astype(int)

        # Métricas a umbral por defecto (0.5)
        fpr_def = confusion_matrix(y_test, pred_default)[0, 1] / max((y_test == 0).sum(), 1)
        rec_def = recall_score(y_test, pred_default)
        prec_def = precision_score(y_test, pred_default, zero_division=0)

        # Métricas a umbral óptimo (p_opt)
        rec_opt = actual_tpr
        prec_opt = precision_score(y_test, pred_optimo, zero_division=0)
        f1_opt = f1_score(y_test, pred_optimo, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Guardar resumen en lista
        all_summary_results.append({
            "variante": tag,
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

        # Datos para gráfica ROC combinada
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)
        roc_curves_data[tag] = {"fpr": fpr_arr, "tpr": tpr_arr, "auc": roc_auc}

        # Generar gráficos individuales
        plot_variant_analysis(y_test, y_proba, tag, opt_thresh, actual_fpr, actual_tpr, EVAL_DIR)
        print(f"  ✓ ROC-AUC: {roc_auc:.4f}")
        print(f"  ✓ Umbral Óptimo p_opt: {opt_thresh:.4f}")
        print(f"  ✓ Recall@5%FPR: {rec_opt:.2%} (vs {rec_def:.2%} a p=0.5)")
        print(f"  ✓ Precisión@5%FPR: {prec_opt:.2%}")

    # 6. Crear Gráfica ROC Combinada
    if roc_curves_data:
        plot_combined_roc(roc_curves_data, EVAL_DIR)
        print(f"\n[+] Gráfica comparativa guardada en: {EVAL_DIR}combined_roc_curves.png")

    # 7. Imprimir y Guardar Tabla Resumen
    if all_summary_results:
        summary_df = pd.DataFrame(all_summary_results)
        
        # Formatear la tabla para la consola
        print("\n" + "=" * 90)
        print(" RESUMEN COMPARATIVO DE SENSIBILIDAD POR VARIANTE (TEST SET)")
        print("=" * 90)
        display_df = summary_df.copy()
        for col in ["roc_auc", "p_optimo", "fpr_actual", "recall_optimo_5%FPR", "precision_optima", "recall_defecto_0.5", "precision_defecto_0.5"]:
            display_df[col] = display_df[col].map("{:.4f}".format)
        
        print(display_df.to_string(index=False))

        # Guardar en CSV
        csv_out = os.path.join(EVAL_DIR, "summary_sensitivity_analysis.csv")
        summary_df.to_csv(csv_out, index=False)
        print(f"\n[+] Tabla resumen exportada exitosamente a: {csv_out}")


if __name__ == "__main__":
    main()