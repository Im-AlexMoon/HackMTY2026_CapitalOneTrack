# Demo datasets and provenance

BAF is stored through Git LFS at `EDA/data/`. Retrieve missing objects with `git lfs pull`. Its authoritative [Kaggle listing](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) links the [Feedzai repository](https://github.com/feedzai/bank-account-fraud).

The monitoring notebooks use **Synthetic Banking Transaction Dataset with Multi-Pattern Fraud Labels for Machine Learning Research**. The original published source is [Mendeley Data version 2](https://data.mendeley.com/datasets/ktbthg777x/2), DOI `10.17632/ktbthg777x.2`. If the team downloaded a Kaggle mirror, record that exact URL, file name and SHA-256: a matching title does not prove byte-for-byte equivalence.

The four built-in scenarios are generated narratives, not sampled dataset records, and their scores are labelled illustrative. Do not report accuracy or lead-time measured on them as model metrics.

## Prepare BAF candidates

```powershell
python scripts/prepare_demo_data.py baf `
  --source EDA/data/Base.csv `
  --output data/curated/baf_candidates.json `
  --month 7 --limit 30
```

The output records source SHA-256, source rows, raw inputs and separate labels. It sets `holdout_verified: false`; only the split membership from the delivered notebook can change that conclusion.

## Prepare transaction candidates

Keep the downloaded file under ignored `data/raw/`. Generate a mapping draft from its actual headers:

```powershell
python scripts/init_transaction_mapping.py `
  --source data/raw/transactions.csv `
  --dataset-name "Synthetic Banking Transaction Dataset with Multi-Pattern Fraud Labels for Machine Learning Research" `
  --dataset-version "EXACT_VERSION" `
  --source-url "EXACT_DOWNLOAD_URL" `
  --output data/raw/transaction-mapping.json
```

Review every detected column. Fill `type_map`, confirm timestamp and currency semantics, and list all labels/post-outcome fields. Null mappings are intentionally invalid. Then profile and curate complete histories:

```powershell
python scripts/profile_dataset.py --source data/raw/transactions.csv --mapping data/raw/transaction-mapping.json --output data/curated/transaction-profile.json
python scripts/prepare_demo_data.py transactions --source data/raw/transactions.csv --mapping data/raw/transaction-mapping.json --output data/curated/transaction-candidates.json --account-limit 4
```

`--account-limit` scans the whole file and retains every row for the first selected accounts. Prefer explicit repeated `--account-id` values once the notebook split is known. A generic row limit is not a valid sequence holdout.

## Compose cross-dataset stories

BAF and the transaction dataset have no shared real identity. Copy `configs/datasets/demo-links.template.json` into ignored `data/curated/`, fill explicit links, and run:

```powershell
python scripts/build_demo_package.py --applications data/curated/baf_candidates.json --transactions data/curated/transaction-candidates.json --links data/curated/demo-links.json --output data/curated/demo-package.json
```

The package preserves both provenances and labels every association as synthetic. Dataset licenses remain independent of this repository; review source terms before redistributing extracts.
