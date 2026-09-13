"""Curated synthetic stories with the raw fields required by real model runners."""

from datetime import datetime, timedelta, timezone


EPOCH = datetime(2026, 9, 12, 9, 0, tzinfo=timezone.utc)
SCENARIOS = [
    {"id":"sleeper_bustout","name":"The sleeper account","description":"Normal activity becomes rapid transfers; review before the final cash-out."},
    {"id":"legitimate","name":"Everyday banking","description":"Steady purchases and salary deposits across a healthy synthetic portfolio."},
    {"id":"identity_farm","name":"Connected applications","description":"Linked synthetic devices expose risky account applications at screening."},
    {"id":"false_positive","name":"An unusual, legitimate day","description":"A major purchase illustrates analyst review and policy tradeoffs."},
]


def stamp(minutes):
    return (EPOCH + timedelta(minutes=minutes)).isoformat()


def baf_profile(risky=False, credit_risk_score=55):
    profile = {
        "income":0.7,"name_email_similarity":0.91,"prev_address_months_count":-1,"current_address_months_count":84,
        "customer_age":42,"days_since_request":0.02,"intended_balcon_amount":120.0,"payment_type":"AA",
        "zip_count_4w":980,"velocity_6h":2200.0,"velocity_24h":4100.0,"velocity_4w":3900.0,
        "bank_branch_count_8w":8,"date_of_birth_distinct_emails_4w":1,"employment_status":"CA",
        "credit_risk_score":credit_risk_score,"email_is_free":0,"housing_status":"BA","phone_home_valid":1,
        "phone_mobile_valid":1,"bank_months_count":48,"has_other_cards":1,"proposed_credit_limit":500.0,
        "foreign_request":0,"source":"INTERNET","session_length_in_minutes":6.2,"device_os":"windows",
        "keep_alive_session":1,"device_distinct_emails_8w":1,"device_fraud_count":0,"x1":0.45,"x2":-0.32,
    }
    if risky:
        profile.update({
            # Dataset-derived high-risk holdout profile. It keeps the demo
            # deterministic while exercising the real exported BAF pipeline.
            "income":0.9,"name_email_similarity":0.1201057840671291,"current_address_months_count":115,
            "customer_age":30,"days_since_request":2.8598206559755925,
            "intended_balcon_amount":-0.6495234173945541,"payment_type":"AC","zip_count_4w":2201,
            "velocity_6h":2249.6942363609965,"velocity_24h":2992.081973716188,
            "velocity_4w":4255.63645979189,"bank_branch_count_8w":1,
            "date_of_birth_distinct_emails_4w":4,"employment_status":"CA","credit_risk_score":269,
            "email_is_free":1,"housing_status":"BA","phone_home_valid":0,"phone_mobile_valid":1,
            "bank_months_count":-1,"has_other_cards":0,"proposed_credit_limit":2000.0,
            "foreign_request":0,"source":"INTERNET","session_length_in_minutes":9.093918417646911,
            "device_os":"windows","keep_alive_session":0,"device_distinct_emails_8w":1,
            "device_fraud_count":0,"x1":3.5432418216782287,"x2":2.9610575013840736,
        })
    return profile


def application(account_id, alias, minute, risk=18, shared=1, mismatch=False):
    profile = baf_profile(risky=mismatch or shared > 1, credit_risk_score=risk)
    profile.update({
        "identity_mismatch":mismatch,"device_shared_count":shared,"initial_balance":20000,"synthetic":True,
        "device_id":"demo-shared-device" if shared > 1 else f"device-{account_id}",
    })
    return {"kind":"application","application_id":f"app_{account_id}","account_id":account_id,"alias":alias,
            "event_time":stamp(minute),"raw_payload":profile,
            "labels":{"source":"synthetic","purpose":"illustrative narrative only"}}


def initial_accounts():
    return [
        application("acct_sleeper","Maya Chen",0,18), application("acct_ava","Lena Foster",0,22),
        application("acct_noah","Noah Williams",0,16), application("acct_olivia","Olivia Brown",0,20),
        application("acct_maya","Maya Patel",0,19),
    ]


def events_for(scenario_id):
    events = []
    account_counts = {}

    def event(account, minute, amount, kind="purchase", description="Card purchase", suspicious=False,
              category_override=None, country_override=None, unique_merchant=False):
        index = account_counts.get(account, 0)
        account_counts[account] = index + 1
        category = category_override or ("transfer" if kind in {"transfer","cash_out","withdrawal"} else ("online_marketplace" if suspicious else "grocery"))
        raw_payload = {
            "synthetic":True,"transaction_type":"transfer" if kind != "purchase" else "purchase",
            "card_id":f"card-{account}-{'new' if suspicious and index > 5 else 'main'}",
            "device_id":f"device-{account}-{'new' if suspicious and index > 5 else 'main'}",
            "ip_address":"203.0.113.90" if suspicious and index > 5 else f"198.51.100.{10 + len(account)}",
            "merchant_id":f"merchant-{category}-{index if unique_merchant else index % 4}","merchant_category":category,
            "merchant_country":country_override or ("BR" if suspicious and index % 2 else "US"),"merchant_city":"Taylorberg",
            "merchant_latitude":-23.5505 if suspicious and index % 2 else 37.7749 + index * 0.001,
            "merchant_longitude":-46.6333 if suspicious and index % 2 else -122.4194 + index * 0.001,
        }
        events.append({"kind":"event","event_id":f"{scenario_id}-{len(events)+1:03}","account_id":account,
                       "event_time":stamp(minute),"type":kind,"amount":amount,"currency":"USD",
                       "description":description,"counterparty_id":raw_payload["merchant_id"],
                       "raw_payload":raw_payload,"labels":{"source":"synthetic"}})

    if scenario_id == "sleeper_bustout":
        for minute, amount in [(1,28),(8,34),(16,22),(24,41)]:
            event("acct_sleeper",minute,amount,description="Routine local purchase")
        events.append(application("acct_new","Avery Martin",25,17))
        event("acct_sleeper",32,1200,"deposit","Incoming transfer")
        for minute, amount in [(33,900),(34,1350),(35,1800),(36,2300),(37,2900)]:
            event("acct_sleeper",minute,amount,"transfer","Rapid outbound transfer",True)
        events.append(application("acct_blocked","Jordan Lee",38,95,8,True))
        event("acct_sleeper",39,4500,"cash_out","Final cash-out attempt",True)
    elif scenario_id == "legitimate":
        for index in range(12):
            # A 12-event, three-day cadence is inside the normal reference
            # distribution exported with the autoencoder.
            event("acct_ava",900+index*4320,100,description="Routine fuel purchase",
                  category_override="gas_station",country_override="CA",unique_merchant=True)
    elif scenario_id == "identity_farm":
        events.extend([application("acct_safe","Sam Rivera",1,21),application("acct_farm_1","Taylor A",2,91,8,True),
                       application("acct_farm_2","Taylor B",3,94,8,True),application("acct_farm_3","Taylor C",4,97,8,True)])
    elif scenario_id == "false_positive":
        for index, amount in enumerate([28,34,42,31,25,38,1200,1600,2100,2600]):
            event("acct_maya",index+1,amount,"purchase","Planned home renovation purchase",index >= 6)
    else:
        raise KeyError(scenario_id)
    return events
