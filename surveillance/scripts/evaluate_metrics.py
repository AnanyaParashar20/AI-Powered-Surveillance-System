"""Evaluation script to compute executive metrics and precision-recall bar chart.

Run:
  python surveillance/scripts/evaluate_metrics.py
"""
import os, glob, json
from collections import defaultdict
import pandas as pd

PRED_DIR = os.path.join("surveillance","data","output")
GT_DIR = os.path.join("surveillance","data","avenue","gt")
IOU_THRESHOLD = 0.30
ANOMALY_LABEL = "anomaly"
EVALUATION_MODE = "event"  # or "frame"

def load_ground_truth_intervals(gt_dir: str) -> pd.DataFrame:
    intervals = []
    for path in sorted(glob.glob(os.path.join(gt_dir, "*.json"))):
        video = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path,'r') as f: data=json.load(f)
        except Exception: continue
        frames=[]
        for fid, boxes in data.items():
            if any(len(b)==4 and b[2]*b[3]>0 for b in boxes):
                try: frames.append(int(fid))
                except: pass
        if not frames: continue
        frames.sort(); start=prev=frames[0]
        for fr in frames[1:]:
            if fr==prev+1: prev=fr
            else:
                intervals.append({"video":video,"start_frame":start,"end_frame":prev,"type":ANOMALY_LABEL})
                start=prev=fr
        intervals.append({"video":video,"start_frame":start,"end_frame":prev,"type":ANOMALY_LABEL})
    return pd.DataFrame(intervals)

def load_predictions(pred_dir: str) -> pd.DataFrame:
    rows=[]
    for path in sorted(glob.glob(os.path.join(pred_dir,"events_*.csv"))):
        try: df=pd.read_csv(path)
        except Exception: continue
        if df.empty: continue
        parts=os.path.basename(path).split('_')
        vid=parts[1] if len(parts)>1 else 'unk'
        df['video']=vid
        rows.append(df)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

def collapse_pred_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["video","type","start_frame","end_frame","score"])
    if 'frame_idx' not in df.columns and 'frame' in df.columns:
        df=df.rename(columns={'frame':'frame_idx'})
    df=df.sort_values(["video","type","track_id","frame_idx"]).reset_index(drop=True)
    intervals=[]
    for (video,etype,track), g in df.groupby(["video","type","track_id"], dropna=False):
        frames=g['frame_idx'].astype(int).tolist()
        scores=g['score'].tolist() if 'score' in g.columns else [0.0]*len(g)
        start=prev=frames[0]; acc=[scores[0]]
        for fr,sc in zip(frames[1:],scores[1:]):
            if fr==prev+1:
                prev=fr; acc.append(sc)
            else:
                intervals.append({"video":video,"type":etype,"start_frame":start,"end_frame":prev,"score":float(sum(acc)/len(acc))})
                start=prev=fr; acc=[sc]
        intervals.append({"video":video,"type":etype,"start_frame":start,"end_frame":prev,"score":float(sum(acc)/len(acc))})
    return pd.DataFrame(intervals)

def temporal_iou(a0,a1,b0,b1):
    inter=max(0,min(a1,b1)-max(a0,b0)+1)
    union=(a1-a0+1)+(b1-b0+1)-inter
    return inter/union if union>0 else 0.0

def evaluate_event_level(pred_int, gt_int):
    results=defaultdict(lambda:{"TP":0,"FP":0,"FN":0})
    gt_by_video={v:g for v,g in gt_int.groupby('video')}
    matched=set()
    for _,p in pred_int.iterrows():
        cand=gt_by_video.get(p.video)
        best=0.0; best_j=None
        if cand is not None:
            for j,(idx,g) in enumerate(cand.iterrows()):
                i=temporal_iou(p.start_frame,p.end_frame,g.start_frame,g.end_frame)
                if i>best: best=i; best_j=idx
        if best>=IOU_THRESHOLD:
            results[p.type]['TP']+=1; matched.add(best_j)
        else:
            results[p.type]['FP']+=1
    for idx in gt_int.index:
        if idx not in matched:
            results[ANOMALY_LABEL]['FN']+=1
    rows=[]
    for t,c in results.items():
        TP,FP,FN=c['TP'],c['FP'],c['FN']
        prec=TP/(TP+FP) if TP+FP else 0.0
        rec=TP/(TP+FN) if TP+FN else 0.0
        f1=2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        rows.append({"type":t,"TP":TP,"FP":FP,"FN":FN,"precision":prec,"recall":rec,"f1":f1})
    df=pd.DataFrame(rows).sort_values('type')
    macro=df[['precision','recall','f1']].mean()
    df.loc[len(df)]=["(macro)",'-','-','-',macro.precision,macro.recall,macro.f1]
    return df

def evaluate_frame_level(pred_df, gt_int):
    gt_frames=set()
    for _,r in gt_int.iterrows():
        for fr in range(r.start_frame,r.end_frame+1):
            gt_frames.add((r.video,fr))
    rows=[]
    for etype,g in pred_df.groupby('type'):
        pred_frames={(row.video,int(row.frame_idx)) for _,row in g.iterrows()}
        TP=len(pred_frames & gt_frames)
        FP=len(pred_frames-gt_frames)
        FN=len(gt_frames-pred_frames)
        prec=TP/(TP+FP) if TP+FP else 0.0
        rec=TP/(TP+FN) if TP+FN else 0.0
        f1=2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        rows.append({"type":etype,"TP":TP,"FP":FP,"FN":FN,"precision":prec,"recall":rec,"f1":f1})
    df=pd.DataFrame(rows)
    macro=df[['precision','recall','f1']].mean()
    df.loc[len(df)]=["(macro)",'-','-','-',macro.precision,macro.recall,macro.f1]
    return df

def plot_precision_recall(df):
    try:
        import plotly.express as px
    except ImportError:
        print('[WARN] plotly not installed; skipping chart')
        return
    pr=df[df.type!="(macro)"][['type','precision','recall']]
    if pr.empty: return
    pr_long=pr.melt(id_vars='type', value_vars=['precision','recall'], var_name='metric', value_name='value')
    fig=px.bar(pr_long,x='type',y='value',color='metric',barmode='group',text=pr_long['value'].map(lambda v:f"{v:.2f}"),color_discrete_map={'precision':'#dc2626','recall':'#1d4ed8'})
    fig.update_layout(title='Precision & Recall per Event Type',yaxis=dict(range=[0,1]))
    try:
        fig.write_image('precision_recall.png')
        print('[INFO] Saved precision_recall.png')
    except Exception:
        print('[INFO] Static export unavailable (install kaleido for PNG).')
    try: fig.show()
    except Exception: pass

def main():
    print('[INFO] Loading ground truth ...')
    gt=load_ground_truth_intervals(GT_DIR)
    if gt.empty:
        print('[ERROR] No GT intervals.'); return
    print(f'[INFO] GT intervals: {len(gt)}')
    print('[INFO] Loading predictions ...')
    preds=load_predictions(PRED_DIR)
    if preds.empty:
        print('[ERROR] No predictions.'); return
    print(f'[INFO] Prediction rows: {len(preds)}')
    if EVALUATION_MODE=='event':
        pred_int=collapse_pred_events(preds)
        print(f'[INFO] Pred intervals: {len(pred_int)}')
        metrics=evaluate_event_level(pred_int, gt)
    else:
        metrics=evaluate_frame_level(preds, gt)
    print('\nExecutive Metrics Table:')
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x,float) else str(x)))
    metrics.to_csv('evaluation_metrics.csv', index=False)
    print('[INFO] Saved evaluation_metrics.csv')
    plot_precision_recall(metrics)
    print('[INFO] Done.')

if __name__=='__main__':
    main()
