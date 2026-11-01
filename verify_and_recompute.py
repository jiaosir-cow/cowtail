"""Portable, read-only verification of this folder; never train or run a model.

Python dependencies for --recompute-four-state: numpy, pandas, scikit-learn.
Run python -B verify_and_recompute.py --recompute-four-state from any working directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def check(ok,message):
    if not bool(ok):raise ValueError(message)

def verify_files():
    m=json.loads((ROOT/'FILE_MANIFEST.json').read_text(encoding='utf-8'))
    count=0
    for name,record in m['files'].items():
        p=(ROOT/name).resolve()
        check(p.is_relative_to(ROOT.resolve()) and p.is_file(),'Missing/unsafe file: '+name)
        check(p.stat().st_size==record['size_bytes'] and sha(p)==record['sha256'],'Changed file: '+name)
        count+=1
    return count

def recompute():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef
    data=ROOT/'01_data'
    archived=ROOT/'03_models_predictions/four_class_old_cow_holdout_90units_1000steps'
    w=pd.read_csv(data/'windows.csv',keep_default_na=False)
    old=pd.read_csv(archived/'inputs/windows.csv',keep_default_na=False)
    expected=pd.read_csv(ROOT/'02_results/four_class_unified_old_cow_holdout/metrics_by_replicate.csv',keep_default_na=False)
    check(len(w)==12736 and w.window_id.is_unique,'Wrong selected dataset')
    combined=np.load(data/'acceleration.npy',mmap_mode='r',allow_pickle=False)
    source=np.load(archived/'inputs/acceleration.npy',mmap_mode='r',allow_pickle=False)
    old_ids=old.set_index('window_id').row_index
    check(np.array_equal(np.asarray(combined),np.asarray(source[old_ids.loc[w.window_id].to_numpy(int)])), 'Selected sensor tensor differs from archived source')
    for cow in ['cowA','cowB','cowC','cowD']:
        cw=pd.read_csv(data/cow/'windows.csv',keep_default_na=False)
        x=np.load(data/cow/'acceleration.npy',mmap_mode='r',allow_pickle=False)
        check(np.array_equal(x,np.asarray(combined[cw.package_row_index.to_numpy(int)])),cow+' data differs')
        check(cw.row_index.tolist()==list(range(len(cw))),cow+' local row order differs')
    selected=w.set_index('window_id')
    rows=0;units=0;max_error=0.0
    for folder in sorted((archived/'results').iterdir()):
        if not folder.is_dir():continue
        match=re.fullmatch(r'(f0[12])_s([012])_b(005|010|025|050|100)_(scratch|act|ufro)',folder.name)
        check(match is not None,'Unexpected unit name')
        fold,rep,percent,method=match.groups();rep=int(rep);percent=int(percent)
        receipt=json.loads((folder/'COMPLETE.json').read_text(encoding='utf-8-sig'))
        endpoint=folder/receipt['attempt']/'step1000'
        for name in ['test_indices.npy','test_state_probabilities.npy','prediction_metadata.json','checkpoint.pt']:
            p=endpoint/name
            check(sha(p)==receipt['artifact_sha256'][p.relative_to(folder).as_posix()],'Endpoint hash differs: '+folder.name+'/'+name)
        indices=np.load(endpoint/'test_indices.npy',allow_pickle=False)
        p=np.load(endpoint/'test_state_probabilities.npy',allow_pickle=False)
        ids=old.set_index('row_index').loc[indices].window_id
        keep=ids.isin(selected.index).to_numpy()
        sub=selected.loc[ids[keep]].reset_index()
        predicted=p[keep].argmax(axis=1)
        for cow in sub.display_cow.unique():
            mask=sub.display_cow.eq(cow).to_numpy()
            case=sub[mask]
            truth=case.target4.to_numpy(int);pred=predicted[mask]
            for weighting in ['window','event_balanced_window']:
                weights=None if weighting=='window' else 1.0/case.event_id.map(case.event_id.value_counts()).to_numpy(float)
                vals={'accuracy':accuracy_score(truth,pred,sample_weight=weights),
                      'macro_f1_four':f1_score(truth,pred,labels=range(4),average='macro',zero_division=0,sample_weight=weights),
                      'macro_f1_supported':f1_score(truth,pred,labels=sorted(set(truth.tolist())),average='macro',zero_division=0,sample_weight=weights),
                      'mcc':matthews_corrcoef(truth,pred,sample_weight=weights)}
                target=expected[expected.group.eq(cow)&expected.percent.eq(percent)&expected.method.eq(method)&expected.replicate.eq(rep)&expected.weighting.eq(weighting)]
                check(len(target)==1,'Missing/duplicate expected metric row')
                for name,value in vals.items():
                    error=abs(float(value)-float(target.iloc[0][name]));max_error=max(max_error,error)
                    check(error<2e-12,'Recomputed score differs: '+str((cow,percent,method,rep,weighting,name)))
                rows+=1
        units+=1
    check(units==90 and rows==len(expected)==360,'Unexpected unit or score count')
    return {'units':units,'recomputed_rows':rows,'metrics_per_row':4,'max_absolute_error':max_error,
            'sensor_tensor_exactly_matches_archived_source':True,'training_executed':False,'inference_executed':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recompute-four-state',action='store_true');args=p.parse_args()
    result={'file_verification':'PASS','verified_files':verify_files()}
    if args.recompute_four_state:result['four_state_recomputation']=recompute()
    print(json.dumps(result,ensure_ascii=False,indent=2))
