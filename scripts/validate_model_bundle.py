"""Validate golden integration cases inside the matching trusted runner environment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.common.bundle import bundle_hash
from models.common.contracts import PredictionRequest
from models.common.service import Runner


def validate(directory, trusted=False, write=False):
    import numpy as np

    manifest = json.loads((directory / 'manifest.json').read_text())
    runner = Runner(manifest['model'], directory, demo=False, trusted=trusted, require_verified=False)
    if not runner.ready:
        raise ValueError(runner.problem)
    cases = json.loads((directory / 'tests/golden_cases.json').read_text())['cases']
    if len(cases) < 2:
        raise ValueError('At least two golden integration cases are required.')
    for i, case in enumerate(cases):
        result = runner.predict(PredictionRequest.model_validate(case['input'])).model_dump()
        expected = case['expected']
        if result['status'] != 'ok' or 'risk_score' not in expected or 'model_vector' not in expected:
            raise ValueError(f'Case {i}: status must be ok; expected risk_score and model_vector are required. {result}')
        tolerance = case.get('atol', 1e-7)
        if not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 0.001:
            raise ValueError('Golden tolerance must be between zero and 0.001.')
        np.testing.assert_allclose(result['risk_score'], expected['risk_score'], atol=tolerance, rtol=0)
        actual_vector = np.asarray(result['features']['model_vector'], dtype=float)
        expected_vector = np.asarray(expected['model_vector'], dtype=float)
        if actual_vector.shape != expected_vector.shape:
            raise ValueError(f'Case {i}: model input shape mismatch {actual_vector.shape} vs {expected_vector.shape}')
        np.testing.assert_allclose(actual_vector, expected_vector, atol=tolerance, rtol=0, equal_nan=True)
    report = {'status': 'passed', 'cases': len(cases), 'bundle_hash': bundle_hash(directory, runner.manifest, runner.contract)}
    if write:
        (directory / 'verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--trusted', action='store_true', help='Acknowledge reviewed local code and artifact execution, inside the runner.')
    parser.add_argument('--write-verification', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.directory, args.trusted, args.write_verification), indent=2))
    except Exception as exc:
        print(json.dumps({'status': 'not_ready', 'detail': str(exc)}, indent=2))
        sys.exit(1)
