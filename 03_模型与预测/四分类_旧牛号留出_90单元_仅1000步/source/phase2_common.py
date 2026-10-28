"""Minimal receipt reader retained from the STA supervisor contract."""
import hashlib
import json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''): h.update(chunk)
    return h.hexdigest()

def check_complete(root):
    root=Path(root).resolve()
    receipt=root/'COMPLETE.json'
    if not receipt.is_file(): return False
    obj=json.loads(receipt.read_text(encoding='utf-8'))
    hashes=obj.get('artifact_sha256',{})
    if obj.get('status')!='complete' or not hashes: raise RuntimeError('Invalid completion receipt')
    for name,digest in hashes.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path)!=digest:
            raise RuntimeError('Completed artifact mismatch: '+str(path))
    return True
