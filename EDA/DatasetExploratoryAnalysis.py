"""
Analisis exploratorio + seleccion de caracteristicas (RFECV) para el dataset
Bank Account Fraud (Base + Variantes I-V).

Correcciones sobre la version anterior:
  1. Rutas apuntando a /mnt/user-data/uploads con los nombres reales de archivo.
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
  6. Evaluación de modelo con TPR Recall para balancear el error respecto a los 
     falsos positivos

     Información de versiones:
     Python 3.12.3
    numpy 2.4.4 | pandas 3.0.2 | scikit-learn 1.8.0 | xgboost 3.4.1
    scipy 1.17.1 | joblib 1.5.3 | matplotlib 3.10.8 | seaborn 0.13.2
"""

import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_curve, make_scorer
from scipy.stats import randint, uniform
from xgboost import XGBClassifier

# ------------------------------------------------------------------
# Configuracion
# ------------------------------------------------------------------
DATA_DIR = "./EDA/data"
OUT_DIR = "./EDA/data/Models_BAF"

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

CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

TARGET = "fraud_bool"
MONTH_COL = "month"

# Split temporal en vez de aleatorio: entrena con los primeros 6 meses,
# evalua con los ultimos 2 -- asi es como se disena el benchmark original de
# BAF (simula produccion real, donde el futuro nunca esta en tu training set).
# Verificado empiricamente en Base.csv: meses 0-5 = 79.5% de las filas,
# meses 6-7 = 20.5%. Ademas, la prevalencia de fraude sube de 1.13% (mes 0)
# a 1.47% (mes 7) -- hay drift real, no es un supuesto teorico.
TRAIN_MONTHS = (0, 1, 2, 3, 4, 5)
TEST_MONTHS = (6, 7)
 
 
def temporal_split(df, train_months=TRAIN_MONTHS, test_months=TEST_MONTHS):
    df_train = df[df[MONTH_COL].isin(train_months)].reset_index(drop=True)
    df_test = df[df[MONTH_COL].isin(test_months)].reset_index(drop=True)
    return df_train, df_test

TARGET_FPR = 0.05


def recall_at_fpr(y_true, y_score, max_fpr=TARGET_FPR):
    """TPR (recall) en el punto de la curva ROC donde FPR = max_fpr."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, max_fpr, side="right") - 1
    idx = max(idx, 0)
    return tpr[idx]


recall_at_fpr_scorer = make_scorer(
    recall_at_fpr, response_method="predict_proba", max_fpr=TARGET_FPR
)

RFECV_SAMPLE_SIZE = 100_000
RFECV_CV_SPLITS = 3
RFECV_STEP = 1

PARAM_DIST = {
    "n_estimators": randint(50, 501),       # enteros discretos entre 50 y 500
    "max_depth": randint(3, 9),             # enteros discretos entre 3 y 8
    "learning_rate": uniform(0.01, 0.29),   # continuo entre 0.01 y 0.30
    "scale_pos_weight": uniform(1, 20),     # continuo entre 1 y 21
    "min_child_weight": randint(1, 20),     # enteros discretos entre 1 y 19
}
N_ITER_SELECTOR_SEARCH = 8
N_ITER_FINAL_SEARCH = 30
FINAL_TUNE_SAMPLE_SIZE = 150_000


def tune_estimator_hyperparams(X, y, scoring, n_iter, base_kwargs=None, cv_splits=3):
    base_kwargs = base_kwargs or {}
    base = XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=1,
        random_state=42,
        **base_kwargs,
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X, y)
    return search.best_params_


def load_and_clean(path):
    df = pd.read_csv(path)
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
    feature_cols = [c for c in df_train.columns if c != TARGET]

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

    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols)) if cat_cols else []
    all_feature_names = cat_feature_names + num_cols

    best_params = tune_estimator_hyperparams(
        X_enc, y,
        scoring="average_precision",
        n_iter=N_ITER_SELECTOR_SEARCH
    )
    print(f"[{tag}] Mejores hiperparametros (estimador de RFECV): {best_params}")

    estimator = XGBClassifier(
        **best_params,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=1,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=RFECV_CV_SPLITS, shuffle=True, random_state=42)

    selector = RFECV(
        estimator=estimator,
        step=RFECV_STEP,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
        min_features_to_select=10,
    )
    selector.fit(X_enc, y)

    selected = [f for f, keep in zip(all_feature_names, selector.support_) if keep]
    print(f"\n[{tag}] Features seleccionadas ({len(selected)}/{len(all_feature_names)}):")
    print(selected)
    print(f"[{tag}] Mejor average_precision (CV): {max(selector.cv_results_['mean_test_score']):.4f}")

    return selector, preprocessor, feature_cols, all_feature_names, selected


def train_and_evaluate(df_train, df_test, feature_cols, preprocessor, selector):
    X_train_enc = preprocessor.transform(df_train[feature_cols])
    X_test_enc = preprocessor.transform(df_test[feature_cols])

    X_train_sel = selector.transform(X_train_enc)
    X_test_sel = selector.transform(X_test_enc)

    y_train = df_train[TARGET]
    y_test = df_test[TARGET]

    if len(y_train) > FINAL_TUNE_SAMPLE_SIZE:
        idx_tune, _ = train_test_split(
            np.arange(len(y_train)),
            train_size=FINAL_TUNE_SAMPLE_SIZE / len(y_train),
            stratify=y_train,
            random_state=42,
        )
    else:
        idx_tune = np.arange(len(y_train))

    best_params = tune_estimator_hyperparams(
        X_train_sel[idx_tune], y_train.iloc[idx_tune],
        scoring=recall_at_fpr_scorer,
        n_iter=N_ITER_FINAL_SEARCH,
    )
    print(f"[clasificador final] Mejores hiperparametros: {best_params}")

    clasificador = XGBClassifier(
        **best_params,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
    )
    clasificador.fit(X_train_sel, y_train)

    y_pred = clasificador.predict(X_test_sel)
    y_proba = clasificador.predict_proba(X_test_sel)[:, 1]

    roc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall_5fpr = recall_at_fpr(y_test, y_proba, TARGET_FPR)

    return clasificador, y_pred, roc, recall, precision, recall_5fpr


def run_pipeline(df_train, df_test, tag, split_name):
    """
    Corre RFECV + tuning + evaluacion completos para UN split ya armado
    (aleatorio o temporal) y guarda su propio bundle. Esto SI debe correr
    completo por cada split -- cada uno tiene su propio df_train, asi que
    no se puede reusar seleccion de features ni hiperparametros entre ambos.
    """
    selector, preprocessor, feature_cols, _, selected = run_rfecv(df_train, f"{tag}_{split_name}")
    clf, y_pred, roc, recall, precision, recall_5fpr = train_and_evaluate(
        df_train, df_test, feature_cols, preprocessor, selector
    )
    y_probas = clf.predict_proba

    model_bundle = {
        "preprocessor": preprocessor,
        "selector": selector,
        "model": clf,
        "feature_cols": feature_cols,
        "Probability grader": y_probas,
    }
    bundle_path = os.path.join(OUT_DIR, f"model_bundle_{split_name}_{tag}.joblib")
    joblib.dump(model_bundle, bundle_path)
    print(f"[{tag} - {split_name}] Modelo guardado en: {bundle_path}")
    print(f"[{tag} - {split_name}] ROC-AUC={roc:.4f}  Recall={recall:.4f}  Precision={precision:.4f}  Recall@5%FPR={recall_5fpr:.4f}")
 
    return {
        "n_features": len(selected), "roc_auc": roc, "recall": recall,
        "precision": precision, "recall_at_5pct_fpr": recall_5fpr,
    }
 
 
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
 
    results = {}
    for path, tag in zip(file_paths, names):
        # Cargar y hacer EDA UNA sola vez -- no depende del split, asi que
        # repetirlo por cada split es I/O y computo tirado a la basura.
        df = load_and_clean(path)
        run_eda(df, tag)


        df_train_og, df_test_og = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=42)
        results[(tag,"Random_w_time")] = run_pipeline(df_train_og, df_test_og, tag, "Random_w_time")
        # 'month' se excluye como feature en AMBOS splits, no solo el
        # temporal. Si se deja en el split aleatorio, ese modelo tiene
        # acceso a una columna que el temporal nunca ve -- y sabemos que
        # correlaciona con la prevalencia de fraude (1.13% mes 0 -> 1.47%
        # mes 7), asi que cualquier diferencia en desempeño quedaria
        # contaminada por la diferencia de features, no solo por el split.
        df_sin_month = df.drop(columns=[MONTH_COL])
 
        # --- Split aleatorio ---
        df_train_rnd, df_test_rnd = train_test_split(
            df_sin_month, test_size=0.2, stratify=df_sin_month[TARGET], random_state=42
        )
        results[(tag, "random")] = run_pipeline(df_train_rnd, df_test_rnd, tag, "random")
 
        # --- Split temporal ---
        df_train_tmp, df_test_tmp = temporal_split(df)
        df_train_tmp = df_train_tmp.drop(columns=[MONTH_COL])
        df_test_tmp = df_test_tmp.drop(columns=[MONTH_COL])
        results[(tag, "temporal")] = run_pipeline(df_train_tmp, df_test_tmp, tag, "temporal")
 
    print("\n=== Resumen comparativo (aleatorio vs temporal) ===")
    header = f"{'variante':<18}{'split':<10}{'n_feat':<8}{'ROC-AUC':<10}{'Recall':<10}{'Precision':<11}{'Recall@5%FPR':<14}"
    print(header)
    for tag in names:
        for split_name in ("random", "temporal", "Random_w_time"):
            r = results[(tag, split_name)]
            print(f"{tag:<18}{split_name:<10}{r['n_features']:<8}{r['roc_auc']:<10.4f}{r['recall']:<10.4f}{r['precision']:<11.4f}{r['recall_at_5pct_fpr']:<14.4f}")

 # Para recuperar probabilidad ejecutar model.predict_proba(X_test).
 # Esto retorna un array dimensiones Nx2, donde cada fila contiene una probabilidad y el valor predicho por el
 # clasificador binario. Revisar Example_proba_run.py para referencia