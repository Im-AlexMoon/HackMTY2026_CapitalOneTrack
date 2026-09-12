"""
Analisis exploratorio + seleccion de caracteristicas (RFECV) para el dataset
Bank Account Fraud (Base + Variantes I-V).

Correcciones sobre la version anterior:
  1. Rutas apuntando a /mnt/user-data/uploads/ con los nombres reales de archivo.
  2. Matriz de covarianza: antes se rompia visualmente (no con excepcion) porque
     columnas como velocity_6h (varianza ~9,000,000) y zip_count_4w (~1,000,000)
     dominan la escala de color frente a columnas 0/1, dejando el heatmap plano.
     Fix: usar SymLogNorm para comprimir la escala sin perder el signo.
  3. -1 como centinela de "missing": SOLO aplica a columnas donde -1 es
     fisicamente imposible (duraciones/conteos). Verificado empiricamente:
        - prev_address_months_count      (71.3% son -1)
        - current_address_months_count   (0.4% son -1)
        - bank_months_count              (25.4% son -1)
        - session_length_in_minutes      (0.2% son -1)
        - device_distinct_emails_8w      (0.04% son -1)
     credit_risk_score e intended_balcon_amount SI tienen valores negativos
     legitimos (min -170 y -15.5 respectivamente) -> NO se tocan.
  4. Encoding one-hot para las categoricas de baja cardinalidad (2-7 categorias):
     payment_type, employment_status, housing_status, source, device_os.
  5. RFECV con XGBoost, scoring='average_precision' (no accuracy) por el
     desbalance severo (~1.1% fraude), y scale_pos_weight ajustado.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from xgboost import XGBClassifier

# ------------------------------------------------------------------
# Configuracion
# ------------------------------------------------------------------
DATA_DIR = "./EDA/data"
OUT_DIR = "./EDA/data/outputs"

file_paths = [
    f"{DATA_DIR}/Base.csv",
    f"{DATA_DIR}/Variant I.csv",
    f"{DATA_DIR}/Variant II.csv",
    f"{DATA_DIR}/Variant III.csv",
    f"{DATA_DIR}/Variant IV.csv",
    f"{DATA_DIR}/Variant V.csv",
]
names = ["base", "groupsize_diff", "prev_diff", "sep", "train_prev_diff", "train_sep"]

# Columnas donde -1 es un centinela de missing (verificado, ver docstring)
MISSING_SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

TARGET = "fraud_bool"

# Para pruebas rapidas / RFECV, usar una submuestra estratificada del set de
# seleccion de features (evita reentrenar XGBoost sobre 1M filas en cada paso
# de eliminacion). Sube este numero si tienes tiempo/computo disponible.
RFECV_SAMPLE_SIZE = 150_000
RFECV_CV_SPLITS = 3
RFECV_STEP = 1  # cuantas features elimina por iteracion; sube a 2-3 para acelerar


def load_and_clean(path):
    df = pd.read_csv(path)
    # -1 -> NaN solo en las columnas donde -1 no es un valor real posible
    for col in MISSING_SENTINEL_COLS:
        df[col] = df[col].replace(-1, np.nan)
    return df


def run_eda(df, tag):
    print(f"\n=== {tag} ===")
    print(df.head())
    print(df.info())
    print(df.describe())

    df.hist(figsize=(14, 12))
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/histogram_{tag}.png")
    plt.close()

    df_numeric = df.select_dtypes(include=["number"])

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_numeric.corr(), cmap="coolwarm", vmin=-1, vmax=1, annot=False)
    plt.title(f"Correlacion - {tag}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/correlation_matrix_{tag}.png")
    plt.close()

    # Covarianza: escalas muy distintas entre columnas (velocity_* en millones
    # vs columnas 0/1). SymLogNorm comprime la escala sin perder el signo,
    # a diferencia del heatmap plano que salia antes.
    cov = df_numeric.cov()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cov,
        cmap="coolwarm",
        norm=SymLogNorm(linthresh=1, vmin=cov.values.min(), vmax=cov.values.max()),
        annot=False,
    )
    plt.title(f"Covarianza (escala symlog) - {tag}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/covariance_matrix_{tag}.png")
    plt.close()


def build_preprocessor(feature_cols):
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    ), cat_cols, num_cols


def run_rfecv(df_train, tag):
    """
    IMPORTANTE: recibe SOLO el split de entrenamiento (df_train), nunca el
    dataframe completo. Si seleccionas features usando filas que despues
    terminan en tu test set, estas filtrando informacion del test hacia la
    seleccion de features (leakage), aunque no uses las etiquetas del test
    para entrenar el modelo final.
    """
    feature_cols = [c for c in df_train.columns if c != TARGET]

    # Submuestra estratificada SOLO para acelerar la busqueda de RFECV
    # (sigue siendo un subconjunto de df_train, nunca toca el test).
    if len(df_train) > RFECV_SAMPLE_SIZE:
        frac = RFECV_SAMPLE_SIZE / len(df_train)
        df_sample, _ = train_test_split(
            df_train, train_size=frac, stratify=df_train[TARGET], random_state=42
        )
    else:
        df_sample = df_train

    X = df_sample[feature_cols]
    y = df_sample[TARGET]

    preprocessor, cat_cols, num_cols = build_preprocessor(feature_cols)
    X_enc = preprocessor.fit_transform(X)

    # Nombres de columnas post one-hot, para poder mapear que features
    # originales sobrevivieron
    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols)) if cat_cols else []
    all_feature_names = cat_feature_names + num_cols

    pos = (y == 1).sum()
    neg = (y == 0).sum()
    scale_pos_weight = neg / max(pos, 1)

    estimator = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        tree_method="hist",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=1,  # paralelismo se maneja en RFECV (n_jobs=-1) para evitar
        random_state=42,  # sobre-suscripcion de nucleos (nested parallelism)
    )

    cv = StratifiedKFold(n_splits=RFECV_CV_SPLITS, shuffle=True, random_state=42)

    selector = RFECV(
        estimator=estimator,
        step=RFECV_STEP,
        cv=cv,
        scoring="average_precision",  # NO accuracy: fraude es ~1.1% positivo
        n_jobs=-1,
        min_features_to_select=5,
    )
    selector.fit(X_enc, y)

    selected = [f for f, keep in zip(all_feature_names, selector.support_) if keep]
    print(f"\n[{tag}] Features seleccionadas ({len(selected)}/{len(all_feature_names)}):")
    print(selected)
    print(f"[{tag}] Mejor average_precision (CV): {max(selector.cv_results_['mean_test_score']):.4f}")

    return selector, preprocessor, feature_cols, all_feature_names, selected


def train_and_evaluate(df_train, df_test, feature_cols, preprocessor, selector):
    """
    Aqui esta la respuesta a "como le indico al modelo cuales features usar":
    NO se reconstruyen los nombres de 'selected' sobre el df original (esos
    nombres son post-encoding, ej. 'payment_type_AA', y el df crudo no los
    tiene). En vez de eso:
      1. Se transforma df_train/df_test con el MISMO preprocessor ya ajustado
         en run_rfecv (solo .transform, nunca .fit_transform de nuevo, para
         no re-aprender el encoding con datos de test).
      2. selector.transform(...) aplica automaticamente la mascara booleana
         (selector.support_) que RFECV aprendio, devolviendo solo las
         columnas seleccionadas -- no hay que indexar nada a mano.
    """
    X_train_enc = preprocessor.transform(df_train[feature_cols])
    X_test_enc = preprocessor.transform(df_test[feature_cols])

    X_train_sel = selector.transform(X_train_enc)
    X_test_sel = selector.transform(X_test_enc)

    y_train = df_train[TARGET]
    y_test = df_test[TARGET]

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    clasificador = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        tree_method="hist",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    clasificador.fit(X_train_sel, y_train)

    y_pred = clasificador.predict(X_test_sel)
    y_proba = clasificador.predict_proba(X_test_sel)[:, 1]  # ROC-AUC necesita probabilidades, no clases

    roc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return clasificador, y_pred, roc, recall, precision


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    for path, tag in zip(file_paths, names):
        df = load_and_clean(path)
        run_eda(df, tag)

        # Split train/test PRIMERO -- el test nunca se toca hasta la evaluacion final
        df_train, df_test = train_test_split(
            df, test_size=0.2, stratify=df[TARGET], random_state=42
        )

        selector, preprocessor, feature_cols, all_names, selected = run_rfecv(df_train, tag)
        clf, y_pred, roc, recall, precision = train_and_evaluate(
            df_train, df_test, feature_cols, preprocessor, selector
        )

        print(f"\n[{tag}] ROC-AUC={roc:.4f}  Recall={recall:.4f}  Precision={precision:.4f}")
        results[tag] = {"n_features": len(selected), "roc_auc": roc, "recall": recall, "precision": precision}

    print("\n=== Resumen por variante ===")
    for tag, r in results.items():
        print(tag, "->", r)