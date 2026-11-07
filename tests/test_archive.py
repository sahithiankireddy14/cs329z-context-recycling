"""The offline reproduction gate must reject damaged or incomplete archives."""
import hashlib
import json
from pathlib import Path
import runpy
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
verify = runpy.run_path(str(ROOT / 'scripts/verify_results_v4.py'))['verify']


def test_saved_archive():
    verify(ROOT / 'artifacts/v4')


@pytest.mark.parametrize('refresh_checksum', [False, True])
def test_reject_missing_prediction(tmp_path, refresh_checksum):
    archive = tmp_path / 'archive'
    shutil.copytree(ROOT / 'artifacts/v4', archive)
    name = 'sweagent_full_jev_predictions_v4.jsonl'
    target = archive / name
    target.write_text('\n'.join(target.read_text().splitlines()[1:]) + '\n')
    if refresh_checksum:
        checksums = json.loads((archive / 'checksums.json').read_text())
        checksums[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        (archive / 'checksums.json').write_text(json.dumps(checksums))
    message = 'Incomplete or duplicate' if refresh_checksum else 'checksum mismatch'
    with pytest.raises(ValueError, match=message):
        verify(archive)
