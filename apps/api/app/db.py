"""Portable relational audit store; JSON becomes JSONB on PostgreSQL."""

from sqlalchemy import JSON, Float, Integer, String, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


JSON_VALUE = JSON().with_variant(JSONB(), "postgresql")


class Record:
    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(100), index=True)
    payload: Mapped[dict] = mapped_column(JSON_VALUE, default=dict)


class Application(Record, Base):
    __tablename__ = "applications"
    account_id: Mapped[str] = mapped_column(String(160), index=True)
    raw_payload: Mapped[dict] = mapped_column(JSON_VALUE)
    labels: Mapped[dict] = mapped_column(JSON_VALUE, default=dict)


class Account(Record, Base):
    __tablename__ = "accounts"


class LedgerEvent(Record, Base):
    __tablename__ = "ledger_events"
    account_id: Mapped[str] = mapped_column(String(160), index=True)
    event_time: Mapped[str] = mapped_column(String(50))
    sequence_number: Mapped[int] = mapped_column(Integer)
    raw_payload: Mapped[dict] = mapped_column(JSON_VALUE)
    labels: Mapped[dict] = mapped_column(JSON_VALUE, default=dict)


class FeatureSnapshot(Record, Base):
    __tablename__ = "feature_snapshots"
    account_id: Mapped[str] = mapped_column(String(160), index=True)
    model: Mapped[str] = mapped_column(String(40))
    preprocessing_version: Mapped[str | None] = mapped_column(String(100))
    contract_hash: Mapped[str | None] = mapped_column(String(128))


class ModelInvocation(Record, Base):
    __tablename__ = "model_invocations"
    account_id: Mapped[str] = mapped_column(String(160), index=True)
    model: Mapped[str] = mapped_column(String(40))
    risk_score: Mapped[float | None] = mapped_column(Float)


class RiskScore(Record, Base):
    __tablename__ = "risk_scores"
    account_id: Mapped[str] = mapped_column(String(160), index=True)
    risk_score: Mapped[float | None] = mapped_column(Float)


class Alert(Record, Base):
    __tablename__ = "alerts"
    account_id: Mapped[str] = mapped_column(String(160), index=True)


class AnalystReview(Record, Base):
    __tablename__ = "analyst_reviews"
    alert_id: Mapped[str] = mapped_column(String(160), index=True)


class ModelBundle(Record, Base):
    __tablename__ = "model_bundles"


class ScenarioRun(Record, Base):
    __tablename__ = "scenario_runs"


class EntityLink(Record, Base):
    __tablename__ = "entity_links"
    account_id: Mapped[str] = mapped_column(String(160), index=True)


class Policy(Base):
    __tablename__ = "policies"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON_VALUE)


def connect(url: str):
    options = {}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            options["poolclass"] = StaticPool
    engine = create_engine(url, **options)
    return engine, sessionmaker(engine, expire_on_commit=False)
