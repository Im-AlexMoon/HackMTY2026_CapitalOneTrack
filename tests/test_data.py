import json
from pathlib import Path
import pytest
from scripts.build_demo_package import build
from scripts.init_transaction_mapping import create_draft
from scripts.model_readiness import report
from scripts.prepare_demo_data import prepare
from scripts.profile_dataset import profile


def test_baf_candidates_preserve_provenance_and_separate_labels(tmp_path):
    source=tmp_path/'baf.csv';source.write_text('month,income,fraud_bool\n0,0.5,0\n7,0.8,1\n7,0.4,0\n')
    output=tmp_path/'candidate.json'
    info=prepare(source,output,'baf',limit=1)
    package=json.loads(output.read_text())
    assert info['holdout_verified'] is False
    assert 'fraud_bool' not in package['records'][0]['raw_payload']
    assert package['evaluation_only'][0]['labels']['fraud_bool']==1
    with pytest.raises(ValueError):
        prepare(source,output,'baf')


def test_momtsim_mapping_is_explicit_and_preserves_raw_fields(tmp_path):
    source=tmp_path/'events.csv';source.write_text('who,when,kind,value,truth\na,2,OUT,12,1\na,1,IN,100,0\n')
    with pytest.raises(ValueError):
        prepare(source,tmp_path/'bad.json','momtsim')
    mapping=tmp_path/'map.json';mapping.write_text(json.dumps({'columns':{'account_id':'who','event_time':'when','type':'kind','amount':'value'},'type_map':{'OUT':'transfer','IN':'deposit'},'currency':'UGX','time_format':'step','epoch':'2026-09-12T00:00:00Z','step_seconds':60,'label_columns':['truth']}))
    output=tmp_path/'ready.json';prepare(source,output,'momtsim',mapping=mapping)
    packet=json.loads(output.read_text())
    assert packet['records'][0]['type']=='deposit'
    assert 'truth' not in packet['records'][0]['raw_payload']
    assert packet['records'][0]['raw_payload']['value']==100


def test_transaction_import_selects_complete_histories_and_profiles_labels(tmp_path):
    source=tmp_path/'transactions.csv'
    source.write_text('transaction_id,customer_id,timestamp,transaction_type,amount,currency,fraud_type\n1,a,2026-01-01T00:00:00Z,IN,100,USD,none\n2,b,2026-01-01T00:01:00Z,OUT,4,USD,card_testing\n3,a,2026-01-01T00:02:00Z,OUT,12,USD,none\n')
    mapping=tmp_path/'mapping.json'
    mapping.write_text(json.dumps({'dataset_name':'Synthetic Banking Transaction Dataset with Multi-Pattern Fraud Labels for Machine Learning Research','dataset_version':'fixture-v2','source_url':'https://example.test/dataset','columns':{'event_id':'transaction_id','account_id':'customer_id','event_time':'timestamp','type':'transaction_type','amount':'amount','currency':'currency'},'type_map':{'IN':'deposit','OUT':'transfer'},'time_format':'iso8601','label_columns':['fraud_type']}))
    output=tmp_path/'candidate.json'
    info=prepare(source,output,'transactions',mapping=mapping,account_limit=1)
    packet=json.loads(output.read_text())
    assert info['complete_account_histories'] is True
    assert info['selected_account_ids']==['a'] and info['row_count']==2 and info['rows_scanned']==3
    assert all(item['account_id']=='a' for item in packet['records'])
    assert all('fraud_type' not in item['raw_payload'] for item in packet['records'])
    report_data=profile(source,mapping)
    assert report_data['transaction_summary']['distinct_accounts']==2
    assert report_data['transaction_summary']['evaluation_labels']['fraud_type']['card_testing']==1


def test_mapping_draft_demo_linker_and_readiness_report(tmp_path):
    source=tmp_path/'transactions.csv'
    source.write_text('transaction_id,customer_id,timestamp,transaction_type,amount,is_fraud\n1,a,2026-01-01T00:00:00Z,IN,100,0\n')
    draft=create_draft(source,'Dataset','v1','https://example.test')
    assert draft['columns']['account_id']=='customer_id'
    assert draft['type_map']=={'IN':None} and draft['label_columns']==['is_fraud']
    apps={'provenance':{'dataset':'BAF','holdout_verified':False},'records':[{'id':'app-1','raw_payload':{'income':1}}],'evaluation_only':[]}
    tx={'provenance':{'dataset':'Transactions','holdout_verified':False},'records':[{'event_id':'e1','account_id':'a','event_time':'2026-01-01T00:00:00+00:00','type':'deposit','amount':100,'currency':'USD','description':'Dataset replay','raw_payload':{}}],'evaluation_only':[]}
    apps_path=tmp_path/'apps.json';apps_path.write_text(json.dumps(apps))
    tx_path=tmp_path/'tx.json';tx_path.write_text(json.dumps(tx))
    links_path=tmp_path/'links.json';links_path.write_text(json.dumps({'links':[{'application_record_id':'app-1','transaction_account_id':'a','demo_account_id':'demo-a','alias':'Example'}]}))
    built=build(apps_path,tx_path,links_path,tmp_path/'demo.json')
    assert built=={'applications':1,'events':1,'holdout_verified':False}
    readiness=report(Path(__file__).resolve().parents[1])
    assert readiness['all_ready'] is False
    assert all(not item['ready'] for item in readiness['models'])
