"""Package four displayed cases without modifying source data or inventing identities."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
SOURCE=PROJECT/'datasets/DATASET_X'
ANNOTATIONS=Path(r'PATH_X')
DISPLAY={'cowA':'cowA','cowB':'cowB','cowC':'cowC','cowD':'cowD'}
STATES=['STANDING','LYING','WALKING','FEEDING']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def check(ok,msg):
    if not bool(ok):raise ValueError(msg)

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def write_csv(path,table):
    path.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(path,index=False,encoding='utf-8-sig',mode='x',lineterminator='\n')

def write_array(path,array):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.save(f,array,allow_pickle=False)

def rename_display(frame):
    f=frame.copy()
    f['source_analysis_group']=f.analysis_group
    f['source_cow_alias']=f.cow_alias
    f['source_registered_cow_id']=f.cow_id
    if 'display_name' in f:f['source_display_name']=f.display_name
    f['display_cow']=f.analysis_group.map(DISPLAY)
    f['analysis_group']=f.display_cow
    f['cow_alias']=f.display_cow
    if 'display_name' in f:f['display_name']=f.display_cow
    # cow_id and all legacy/time/label fields are never rewritten by a display rename.
    return f

def main():
    data=ROOT/'01_数据'
    check(not (data/'DATA_MANIFEST.json').exists(),'Dataset already prepared; preserve the existing package')
    paths={n:SOURCE/n for n in ['windows.csv','events.csv','acceleration.npy']}
    paths['canonical_annotations']=ANNOTATIONS
    before={n:sha(p) for n,p in paths.items()}
    w0=pd.read_csv(paths['windows.csv'],keep_default_na=False)
    e0=pd.read_csv(paths['events.csv'],keep_default_na=False)
    a0=pd.read_csv(ANNOTATIONS,keep_default_na=False)
    w=w0[w0.analysis_group.isin(DISPLAY)].copy()
    e=e0[e0.analysis_group.isin(DISPLAY)].copy()
    check(len(w)==12736 and len(e)==364,'Unexpected selected sample count')
    check(w.window_id.is_unique and e.event_id.is_unique,'Duplicate IDs')
    rows=w.row_index.to_numpy(np.int64)
    w['source_dataset_row_index']=rows
    w['row_index']=np.arange(len(w),dtype=np.int64)
    w['package_row_index']=w.row_index
    w=rename_display(w);e=rename_display(e)
    check(set(w.event_id)==set(e.event_id),'Events and windows disagree')
    check(w.groupby(['device_mac','session_id']).display_cow.nunique().max()==1,'One session enters several cases')
    sessions=w[['device_mac','session_id','source_analysis_group','display_cow','source_cow_alias','source_registered_cow_id','legacy_cow_id']].drop_duplicates()
    check(len(sessions)==31,'Unexpected session count')
    group_by_session={tuple(v[:2]):v[2] for v in sessions[['device_mac','session_id','display_cow']].itertuples(index=False,name=None)}
    selected=a0[[tuple(v) in group_by_session for v in a0[['device_mac','session_id']].to_numpy()]].copy()
    selected['display_cow']=[group_by_session[tuple(v)] for v in selected[['device_mac','session_id']].to_numpy()]
    selected['source_analysis_group']=selected.display_cow.map({v:k for k,v in DISPLAY.items()})
    selected['included_in_current_window_evaluation']=selected.event_id.isin(e.event_id)
    check(len(selected)==468,'Unexpected canonical annotation count')
    check(set(e.event_id).issubset(set(selected.event_id)),'Selected evaluation event absent from canonical labels')
    source_array=np.load(paths['acceleration.npy'],mmap_mode='r',allow_pickle=False)
    check(source_array.shape==(28911,3,256) and source_array.dtype==np.float32,'Sensor tensor changed')
    tensor=np.asarray(source_array[rows])
    check(np.isfinite(tensor).all(),'Nonfinite sensor input')
    data.mkdir(parents=True,exist_ok=True)
    write_array(data/'acceleration.npy',tensor)
    write_csv(data/'windows.csv',w);write_csv(data/'events.csv',e)
    write_csv(data/'canonical_annotations.csv',selected)
    write_csv(data/'session_identity_mapping.csv',sessions)
    summaries=[]
    for cow in ['cowA','cowB','cowC','cowD']:
        folder=data/cow
        cw=w[w.display_cow.eq(cow)].copy();ce=e[e.display_cow.eq(cow)].copy();ca=selected[selected.display_cow.eq(cow)].copy()
        package_rows=cw.package_row_index.to_numpy(np.int64)
        cw['row_index']=np.arange(len(cw),dtype=np.int64)
        write_array(folder/'acceleration.npy',tensor[package_rows])
        write_array(folder/'package_row_indices.npy',package_rows)
        write_array(folder/'source_dataset_row_indices.npy',cw.source_dataset_row_index.to_numpy(np.int64))
        write_csv(folder/'windows.csv',cw);write_csv(folder/'events.csv',ce);write_csv(folder/'all_annotations.csv',ca)
        truth=cw[['row_index','package_row_index','window_id','event_id','target4','original_state','display_cow','source_analysis_group','source_registered_cow_id','legacy_cow_id']]
        write_csv(folder/'test_truth.csv',truth)
        ex=ca[ca.layer.eq('body_state') & ~ca.included_in_current_window_evaluation].copy()
        ex['exclusion_status']='Not in previously audited fully-contained window set; original annotation retained'
        write_csv(folder/'body_annotations_outside_evaluation.csv',ex)
        support={s:int(cw.target4.eq(i).sum()) for i,s in enumerate(STATES)}
        info={'display_cow':cow,'source_analysis_group':str(cw.source_analysis_group.iloc[0]),
              'source_registered_cow_ids':sorted(cw.source_registered_cow_id.unique()),
              'legacy_cow_ids':sorted(cw.legacy_cow_id.unique()),'windows':len(cw),'eligible_intervals':len(ce),
              'canonical_annotations':len(ca),'body_state_annotations':int(ca.layer.eq('body_state').sum()),
              'sessions':len(cw[['device_mac','session_id']].drop_duplicates()),'class_window_support':support,
              'missing_truth_classes':[s for s,c in support.items() if c==0],
              'identity_basis':'User statement: four field-recorded cattle; original source mappings retained',
              'independently_verified_identity':False,'sensor_input':'float32 tri-axial acceleration, 256 samples/window'}
        write_json(folder/'CASE.json',info);summaries.append(info)
        (folder/'README.md').write_text(f'# {cow}\n\n本目录包含本案例所用的传感窗口、标签、全部对应母标签及来源索引。\n\n'
            f'可用区间：{len(ce)}；窗口：{len(cw)}；全部对应母标签：{len(ca)}。\n\n'
            '`row_index`对应本目录acceleration.npy，`package_row_index`对应上级合并数组。原始标签内容及来源牛号保持不变。\n\n'
            '身份声明和来源对应见CASE.json；实验结果见根目录02_实验结果。\n'+
            ('\n本案例没有采食真值，不能作为完整四分类验证。\n' if cow=='cowD' else ''),encoding='utf-8')
    summary=[]
    for v in summaries:
        row={k:val for k,val in v.items() if not isinstance(val,(dict,list))}
        row.update({s:v['class_window_support'][s] for s in STATES});summary.append(row)
    write_csv(data/'四牛标签与窗口数量.csv',pd.DataFrame(summary))
    write_json(ROOT/'身份与命名约定.json',{
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'display_cows':['cowA','cowB','cowC','cowD'],
        'display_rename':{'cowA':'cowA','cowB':'cowB','cowC':'cowC','cowD':'cowD'},
        'user_statement':'这四个牛就是完整的四个牧场实测的牛',
        'identity_basis':'user-attested field records','identities_independently_verified':False,
        'naming_is_not_evidence_of_additional_independent_animals':True,
        'source_registered_identifiers_preserved':True,
        'new_C_must_not_automatically_reuse_old_six_cow_C_id':'The historical COW_OTHER=COWID_01 is a different source grouping; this rename does not establish that ID mapping.',
        'source_lineage':summaries})
    after={n:sha(p) for n,p in paths.items()}
    check(before==after,'Read-only source changed')
    write_json(data/'DATA_MANIFEST.json',{'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
        'array_shape':list(tensor.shape),'windows':len(w),'eligible_intervals':len(e),'canonical_annotations':len(selected),
        'source_sha256_before':before,'source_sha256_after':after,'source_paths':{n:str(p) for n,p in paths.items()},
        'sources_unchanged':True,'numeric_sensor_values_unchanged':True,'label_values_unchanged':True,
        'session_or_event_crosses_display_cases':False,'display_name_only_changed':True,'class_order':STATES,
        'cases':summaries,'artifact_sha256':{p.relative_to(data).as_posix():sha(p) for p in data.rglob('*') if p.is_file()},
        'script_sha256':sha(Path(__file__))})
    print(json.dumps({'status':'PASS','windows':len(w),'events':len(e),'canonical_annotations':len(selected),'cases':summaries},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
