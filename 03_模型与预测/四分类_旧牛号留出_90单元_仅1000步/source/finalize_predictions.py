"""Archive every fixed endpoint before held-out scoring; guarded CPU work only."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import zipfile


def main():
    from windows_launcher import verify_worker_membership
    guard = verify_worker_membership(os.environ.get('DATASET_X'))
    from phase2_common import sha, check_complete
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / 'configs/FROZEN_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    units = json.loads((root / 'configs/units.json').read_text())['units']
    assert len(units) == 90 and set(units) == set(protocol['units'])
    receipts = {}
    for unit_id in protocol['units']:
        folder = root / 'results' / unit_id
        assert check_complete(folder), unit_id
        path = folder / 'COMPLETE.json'
        receipt = json.loads(path.read_text())
        assert receipt['unit_id'] == unit_id and receipt['parameter_count'] == 828
        assert receipt['endpoint_steps'] == [1000, 3000]
        assert receipt['binding']['protocol_sha256'] == sha(protocol_path)
        receipts[unit_id] = sha(path)
    output = root / 'export'
    output.mkdir(exist_ok=False)
    archive_path = output / 'all_budget_predictions.zip'
    paths = [root / 'PROTOCOL.md']
    for folder in ('results', 'engineering', 'source', 'configs', 'tests'):
        paths.extend(p for p in sorted((root / folder).rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and 'runtime_cache' not in p.parts and not p.name.endswith('.tmp'))
    supervisor = root / 'supervisor'
    paths.append(supervisor / 'baseline.json')
    for label in ['gpu_smoke', *protocol['units']]:
        for suffix in ('.done.json', '.config.json', '.config.json.worker.json', '.log'):
            path = supervisor / (label + suffix)
            assert path.is_file(), path
            paths.append(path)
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
    result = {'status': 'complete', 'formal_units': 90, 'endpoint_steps': [1000, 3000],
        'protocol_sha256': sha(protocol_path), 'unit_receipt_sha256': receipts,
        'archive': archive_path.name, 'archive_bytes': archive_path.stat().st_size,
        'archive_sha256': sha(archive_path), 'guard': guard, 'held_out_scoring': False,
        'utc': datetime.now(timezone.utc).isoformat(),
        'live_status_note': 'Fetch final STATUS and resource log separately after this finalizer exits.'}
    receipt_path = output / 'PREDICTIONS_FROZEN.json'
    receipt_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    (output / 'COMPLETE.json').write_text(json.dumps({'status': 'complete', 'artifact_sha256': {
        archive_path.name: sha(archive_path), receipt_path.name: sha(receipt_path)}}, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
