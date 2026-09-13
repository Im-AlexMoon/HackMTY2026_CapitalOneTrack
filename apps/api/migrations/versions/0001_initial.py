"""Initial relational audit schema, JSONB on PostgreSQL."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    json_type = sa.JSON().with_variant(JSONB(), 'postgresql')
    extra = {
        'applications': [('account_id', sa.String(160), False), ('raw_payload', json_type, False), ('labels', json_type, False)],
        'accounts': [],
        'ledger_events': [('account_id', sa.String(160), False), ('event_time', sa.String(50), False), ('sequence_number', sa.Integer(), False), ('raw_payload', json_type, False), ('labels', json_type, False)],
        'feature_snapshots': [('account_id', sa.String(160), False), ('model', sa.String(40), False), ('preprocessing_version', sa.String(100), True), ('contract_hash', sa.String(128), True)],
        'model_invocations': [('account_id', sa.String(160), False), ('model', sa.String(40), False), ('risk_score', sa.Float(), True)],
        'risk_scores': [('account_id', sa.String(160), False), ('risk_score', sa.Float(), True)],
        'alerts': [('account_id', sa.String(160), False)],
        'analyst_reviews': [('alert_id', sa.String(160), False)],
        'model_bundles': [], 'scenario_runs': [], 'entity_links': [('account_id', sa.String(160), False)]}
    for table, fields in extra.items():
        op.create_table(table, sa.Column('id', sa.String(160), primary_key=True), sa.Column('run_id', sa.String(100), nullable=False),
                        sa.Column('payload', json_type, nullable=False), *[sa.Column(name, typ, nullable=nullable) for name, typ, nullable in fields])
        op.create_index(f'ix_{table}_run_id', table, ['run_id'])
        for name, _, _ in fields:
            if name in {'account_id', 'alert_id'}:
                op.create_index(f'ix_{table}_{name}', table, [name])
    op.create_table('policies', sa.Column('id', sa.String(100), primary_key=True), sa.Column('payload', json_type, nullable=False))


def downgrade():
    raise RuntimeError('Destructive downgrade disabled. Restore a known database backup instead.')
