import joblib
import pandas as pd

# 1. Cargar el bundle exportado
bundle = joblib.load("./data/outputs/model_bundle_temporal_sep.joblib")
preprocessor = bundle["preprocessor"]
selector = bundle["selector"]
model = bundle["model"]
feature_cols = bundle["feature_cols"]

# 2. DEFINIR X_NUEVA: Diccionario con la nueva solicitud convertido a DataFrame de 1 fila
nueva_transaccion = {
    # Numéricas de Identidad y Perfil del Cliente
    "income": 0.6,
    "name_email_similarity": 0.88,
    "prev_address_months_count": -1,  # Centinela (-1 = Sin dirección previa)
    "current_address_months_count": 48,
    "customer_age": 38,
    "days_since_request": 0.015,
    "intended_balcon_amount": 105.0,
    "zip_count_4w": 1150,
    "velocity_6h": 3100.5,
    "velocity_24h": 5200.0,
    "velocity_4w": 4050.8,
    "bank_branch_count_8w": 12,
    "date_of_birth_distinct_emails_4w": 1,
    "credit_risk_score": 160,
    "bank_months_count": 18,
    "proposed_credit_limit": 500.0,
    "session_length_in_minutes": 3.8,
    "device_distinct_emails_8w": 1,
    "device_fraud_count": 0,
    "month": 7,  # Mes dentro del periodo de test/evaluación

    # Binarias (0 o 1)
    "email_is_free": 1,
    "phone_home_valid": 1,
    "phone_mobile_valid": 1,
    "has_other_cards": 0,
    "foreign_request": 0,
    "keep_alive_session": 1,

    # Categóricas (Baja cardinalidad)
    "payment_type": "AA",
    "employment_status": "CA",
    "housing_status": "BC",
    "source": "INTERNET",
    "device_os": "linux",
    "x1": 0.63528468,
    "x2": -1.27626537,
}

X_nueva = pd.DataFrame([nueva_transaccion])

# Reemplazar centinelas -1 en columnas que lo requieran
MISSING_SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
for col in MISSING_SENTINEL_COLS:
    if col in X_nueva.columns:
        X_nueva[col] = X_nueva[col].replace(-1, None)

# 3. Pipeline de inferencia
X_enc = preprocessor.transform(X_nueva[feature_cols])
X_sel = selector.transform(X_enc)

# Predict_proba devuelve la probabilidad de fraude p in [0, 1]
probabilidad_fraude = model.predict_proba(X_sel)[0, 1]
print(f"Riesgo de fraude estimado: {probabilidad_fraude:.2%}")