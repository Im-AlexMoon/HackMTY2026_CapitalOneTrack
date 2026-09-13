# Model handoff and preprocessing

Use `models/onboarding`, `models/transaction`, or `models/sequence` as the independent bundle directory. Each has its own Dockerfile, dependency list, adapter, pipeline placeholder, feature contract, artifact directory and golden test file. Do not use the model filename as the complete handoff.

## Receive and inspect

1. Obtain the executed notebook, model artifacts, fitted transformers/selectors, actual dataset version, split membership/seed and at least two raw inputs with notebook outputs.
2. Run `python scripts/inspect_artifact.py notebook.ipynb model.pickle`. This only reads cells/opcodes and hashes; it cannot establish artifact safety or recover fitted state.
3. Classify: **green** for complete frozen pipelines, **amber** when the notebook and stored state allow exact reconstruction, **red** when learned state or feature definitions are missing. Red requires reexport from the model owner.
4. Keep team-provided binaries in `artifact/` (ignored by Git). Inspect notebook calls, feature order, units, sentinel treatment, encoding, imputation, log/scaling transformations, temporal cutoffs, window sizes and export cells. Never fit a transformer in serving.

## Complete the bundle

- `manifest.json`: set `status` to `candidate`; record model/version, preprocessing/version, dataset, target, exact Python/library versions, artifact format and SHA-256 for every artifact. Complete `dataset_provenance` with version, source URL, source file hashes and split reference. Assign the model artifact `role: "model"`; a separate Keras scaler uses `role: "preprocessor"`.
- `feature_contract.json`: set `status: "ready"`; declare ordered fields (`name`, `dtype`, `nullable`), `feature_order`, optional categories and `unknown_category: "reject" | "allow"`. Sequence contracts additionally declare length, minimum history and exact padding policy/value.
- `pipeline.py`: implement `build_features(payload)` returning one feature dictionary or an ordered sequence of dictionaries. Port only the transformations and derived variables used by the notebook. Future events and labels are removed before this call.
- Set `input_format` to `dataframe` or `numpy`. Set `preprocessing_mode` to `embedded_pipeline`, `bundle_transformer`, `pipeline_module`, or `none`. The last requires that no learned preprocessing is needed.
- Supported exports: sklearn Pipeline; pickle/joblib dictionary with `model`, `preprocessor`, optional `selector` and `feature_names`; native `.keras` with separately exported scaler; custom `pipeline.load_artifact`, `transform`, and `score` hooks for other formats. Import custom trusted code through the per-model adapter only.
- Declare score method (`predict_proba`, `predict`, `decision_function`, `reconstruction_mse`, or `custom`), positive class where relevant, direction, and normalization. `identity` requires an already bounded score; `affine` needs low/high; `logistic` needs fitted slope/intercept; `ecdf` needs the sorted validation-only reference. A padded/masked autoencoder requires a custom error function matching the notebook.
- Update that runner's `requirements.lock` and Dockerfile Python version to match its export. The environments may differ between all three models.

## Verify and activate

Each entry of `tests/golden_cases.json` must contain `input` (the canonical runner request), `expected: {risk_score, model_vector}`, and optional `atol` (default 1e-7). At least two are required; test nulls/categories and minimum sequence history in additional cases. The repository currently includes deterministic integration baselines generated from reviewed artifacts and the ported preprocessing. Replace or confirm them with independent Colab outputs before claiming formal notebook parity.

Inside the exact runner environment, mount only the bundle being validated as writable so the receipt can be created:

```sh
docker compose run --rm \
  -v ./models/onboarding:/app/models/onboarding:rw \
  onboarding python /app/scripts/validate_model_bundle.py \
  /app/models/onboarding --trusted --write-verification
```

Repeat for transaction and sequence. Validation loads trusted artifacts, compares numerical feature vectors and predictions, and writes a content-hashed verification receipt. Any later bundle change invalidates it. Empty tests and pending bundles exit nonzero. No fictitious pickle is shipped.

Set `DEMO_MODE=false` and `ALLOW_TRUSTED_ARTIFACTS=true` for verified services, rebuild, and restart. Check `/health`, `/metadata`, `/validate`, `/predict`. Until the bundle passes, `ready` remains false and scores remain null. The API must also use external mode; it refuses synthetic runner scores there. Use `/api/applications` and `/api/events` for real-dataset input. Synthetic preset replay is deliberately disabled in external mode.

Both the raw vector and transformed model vector are recorded with contract/preprocessing versions for every inference. This is the evidence needed to diagnose differences between notebook and deployment.
