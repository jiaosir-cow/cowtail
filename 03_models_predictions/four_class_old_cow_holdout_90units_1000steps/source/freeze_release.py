"""Freeze a complete, matched four-state cohort; do not package local test truth."""
import hashlib
import json
from pathlib import Path
import random
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = 'PATH_X'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    preparation = json.loads((ROOT / 'inputs/PREPARATION.json').read_text(encoding='utf-8'))
    assert preparation['status'] == 'complete'
    assert preparation['array_shape'] == [29790, 3, 256]
    assert preparation['eligible_event_count'] == 882
    assert preparation['protocol_text_sha256'] == sha(ROOT / 'PROTOCOL.md')
    assert preparation['preparer_code_sha256'] == sha(ROOT / 'source/prepare_budget_inputs.py')
    assert preparation['source_sha256_before'] == preparation['source_sha256_after']
    for relative, expected in preparation['artifact_sha256'].items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT.resolve()) and sha(path) == expected, relative
    units = {}
    for budget in preparation['budgets']:
        for method in ('scratch', 'act', 'ufro'):
            fold, rep, percent = budget['fold_id'], budget['replicate'], budget['nominal_percent']
            unit_id = f'{fold}_s{rep}_b{percent:03d}_{method}'
            units[unit_id] = {'unit_id': unit_id, 'fold_id': fold, 'test_cow': budget['test_cow'],
                'replicate': rep, 'percent': percent, 'method': method,
                'budget_dir': budget['budget_dir'], 'fold_dir': f'inputs/folds/{fold}',
                'target_seed': 20260909 + 10000 * budget['fold_number'] + 100 * rep,
                'checkpoint': None if method == 'scratch' else f'inputs/checkpoints/{method}_seed{rep}.pt'}
    assert len(units) == 90
    write_new(ROOT / 'configs/units.json', {'units': units})
    groups = [tuple(f'{b["fold_id"]}_s{b["replicate"]}_b{b["nominal_percent"]:03d}_{m}'
                    for m in ('scratch', 'act', 'ufro')) for b in preparation['budgets']]
    rng = random.Random(20260909)
    rng.shuffle(groups)
    ordered = []
    for group in groups:
        group = list(group)
        rng.shuffle(group)
        ordered.extend(group)
    jobs = [{'label': 'gpu_smoke', 'lane': 'gpu', 'gpu': True,
        'script': REMOTE + '/source/budget_runner.py', 'arguments': ['--gpu-smoke', '--device', 'cuda'],
        'complete_path': REMOTE + '/engineering/gpu_smoke/COMPLETE.json', 'cwd': REMOTE,
        'threads': 2, 'gpu_allocator_gib': 2, 'gpu_pacing_duty': 0.5}]
    for unit_id in ordered:
        jobs.append({'label': unit_id, 'lane': 'gpu', 'gpu': True,
            'script': REMOTE + '/source/budget_runner.py', 'arguments': ['--unit', unit_id, '--device', 'cuda'],
            'complete_path': REMOTE + f'/results/{unit_id}/COMPLETE.json', 'cwd': REMOTE,
            'requires': ['gpu_smoke'], 'threads': 2, 'gpu_allocator_gib': 2, 'gpu_pacing_duty': 0.5})
    jobs.append({'label': 'finalize_predictions', 'lane': 'gate', 'gpu': False,
        'script': REMOTE + '/source/finalize_predictions.py', 'arguments': [],
        'complete_path': REMOTE + '/export/COMPLETE.json', 'cwd': REMOTE,
        'requires': list(units), 'threads': 2})
    write_new(ROOT / 'configs/queue.json', jobs)
    source_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / 'source').rglob('*.py'))
                     if '__pycache__' not in p.parts}
    input_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / 'inputs').rglob('*'))
                    if p.is_file() and p.name != 'BUNDLE_SHA256SUMS.json'
                    and not p.name.endswith(('.tmp', '.partial'))}
    test_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / 'tests').rglob('*.py'))}
    config_hashes = {f'configs/{name}': sha(ROOT / 'configs' / name) for name in ('units.json', 'queue.json')}
    artifacts = {**source_hashes, **input_hashes, **test_hashes, **config_hashes}
    assert not any(name.startswith('analysis/') or 'test_truth' in name for name in artifacts)
    protocol = {'protocol_id': 'four_state_original_interval_budget_20260909_v1',
        'protocol_text_sha256': sha(ROOT / 'PROTOCOL.md'), 'artifact_sha256': artifacts,
        'source_sha256': source_hashes, 'input_sha256': input_hashes,
        'units': list(units), 'execution_order': ordered, 'class_order': ['standing', 'lying', 'walking', 'feeding'],
        'parameter_count': 828, 'window_count': 29790, 'eligible_event_count': 882,
        'primary_step': 1000, 'sensitivity_step': 3000, 'batch_size': 32, 'optimizer': 'Adam',
        'learning_rate': 0.001, 'gradient_clip': 1.0, 'cpu_threads': 2, 'interop_threads': 1,
        'gpu_allocator_gib': 2, 'gpu_duty': 0.5, 'new_validation_labels_used': 0,
        'historical_development_labels_have_been_used': True,
        'test_truth_packaged': False, 'other_fold_training_labels_exist': True,
        'preparation_sha256': sha(ROOT / 'inputs/PREPARATION.json'),
        'local_test_truth_sha256': sha(ROOT / 'analysis/test_truth.csv'),
        'budget_unit': 'original eligible event_id; nested class-blind uniform permutation',
        'label_saving_rule': 'lowest observed budget meeting both full-scratch MCC and macroF1 for each test cow; step1000 only; descriptive'}
    write_new(ROOT / 'configs/FROZEN_PROTOCOL.json', protocol)
    write_new(ROOT / 'source/SOURCE_SHA256SUMS.json', {k.removeprefix('source/'): v for k, v in source_hashes.items()})
    write_new(ROOT / 'inputs/BUNDLE_SHA256SUMS.json', {k.removeprefix('inputs/'): v for k, v in input_hashes.items()})
    for relative, expected in artifacts.items():
        assert sha(ROOT / relative) == expected, relative
    (ROOT / 'delivery').mkdir(exist_ok=True)
    archive_path = ROOT / 'delivery/four_state_budget_release_v1.zip'
    paths = [ROOT / 'PROTOCOL.md']
    for folder in ('source', 'inputs', 'configs', 'tests'):
        paths.extend(p for p in sorted((ROOT / folder).rglob('*')) if p.is_file()
                     and '__pycache__' not in p.parts and not p.name.endswith(('.tmp', '.partial')))
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    receipt = {'file': archive_path.name, 'bytes': archive_path.stat().st_size, 'sha256': sha(archive_path),
        'protocol_sha256': sha(ROOT / 'configs/FROZEN_PROTOCOL.json'), 'members': len(paths),
        'formal_units': len(units), 'queue_jobs': len(jobs), 'local_test_truth_packaged': False}
    write_new(ROOT / 'delivery/RELEASE_RECEIPT.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
