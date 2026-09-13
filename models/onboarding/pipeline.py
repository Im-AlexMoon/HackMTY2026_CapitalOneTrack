"""Exact serving feature extraction for the BAF temporal-sep bundle."""

from models.common.contracts import ContractError

FEATURES = ["income","name_email_similarity","prev_address_months_count","current_address_months_count","customer_age","days_since_request","intended_balcon_amount","payment_type","zip_count_4w","velocity_6h","velocity_24h","velocity_4w","bank_branch_count_8w","date_of_birth_distinct_emails_4w","employment_status","credit_risk_score","email_is_free","housing_status","phone_home_valid","phone_mobile_valid","bank_months_count","has_other_cards","proposed_credit_limit","foreign_request","source","session_length_in_minutes","device_os","keep_alive_session","device_distinct_emails_8w","device_fraud_count","x1","x2"]
MISSING_SENTINELS = {"prev_address_months_count","current_address_months_count","bank_months_count","session_length_in_minutes","device_distinct_emails_8w"}


def build_features(payload):
    application = payload.get("application", {})
    missing = [name for name in FEATURES if name not in application]
    if missing:
        raise ContractError(f"Application is missing BAF fields: {missing}.")
    result = {name: application[name] for name in FEATURES}
    for name in MISSING_SENTINELS:
        if result[name] == -1:
            result[name] = None
    return result


def reason_codes(payload, snapshot, raw_score):
    app = payload["application"]
    reasons = []
    if app.get("foreign_request") == 1:
        reasons.append({"code":"FOREIGN_APPLICATION","description":"Application originated outside the expected market.","kind":"observation"})
    if app.get("device_fraud_count", 0) > 0:
        reasons.append({"code":"DEVICE_FRAUD_HISTORY","description":"Application device has prior fraud activity.","kind":"observation"})
    if app.get("name_email_similarity", 1) < 0.2:
        reasons.append({"code":"LOW_NAME_EMAIL_SIMILARITY","description":"Name and email have unusually low similarity.","kind":"observation"})
    return reasons
