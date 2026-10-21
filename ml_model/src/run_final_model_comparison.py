from pathlib import Path
from urllib.parse import urlsplit
import json, re, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
SEED=42; MAX_LEN=160; VOCAB=128; EPOCHS=4; BATCH=512
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"ml_model"/"external_data"
OUT=ROOT/"ml_model"/"results"/"final_model_comparison"
PRED=OUT/"predictions"; GRAPHS=OUT/"graphs"
PRIMARY=DATA/"labelled_test"/"phishing_url_dataset.csv"
SECOND=DATA/"alternative_test"/"clean_external_test.csv"
NAMES=["Logistic Regression","Random Forest","Extra Trees","Linear SVM","XGBoost","LightGBM","HistGradientBoosting","CatBoost","FNN","DNN","Wide & Deep","TabNet","Character CNN","URL Transformer","Random Forest + XGBoost","Random Forest + Character CNN","Character CNN + XGBoost","Random Forest + Character CNN + XGBoost","CNN-LSTM","GNN","Vision Transformer","Multimodal Fusion"]
WORDS=("login","verify","update","secure","account","bank","password","signin","confirm","wallet","paypal","support","recover","auth","security","invoice")
def norm(x):
    if pd.isna(x): return ""
    s=str(x).strip()
    if not s:return ""
    if "://" not in s:s="https://"+s
    try:
        p=urlsplit(s); h=(p.hostname or "").lower()
        if not h:return s.lower().split("#",1)[0]
        try: port=p.port
        except ValueError: port=None
        return f"{p.scheme.lower()}://{h}{':'+str(port) if port else ''}{p.path}"+(f"?{p.query}" if p.query else "")
    except Exception:return s.lower().split("#",1)[0]
def host(u):
    try:return (urlsplit(u).hostname or "").lower()
    except Exception:return ""
def slug(s):return re.sub(r"[^a-z0-9]+","_",s.lower()).strip("_")
def load():
    a=pd.read_csv(PRIMARY,usecols=["URL","Label"]); a["target"]=pd.to_numeric(a.Label,errors="coerce").map({0:1,1:0})
    b=pd.read_csv(SECOND,usecols=["URL","Label"]); b["target"]=pd.to_numeric(b.Label,errors="coerce").map({0:0,1:1})
    d=pd.concat([a[["URL","target"]],b[["URL","target"]]],ignore_index=True); d.URL=d.URL.map(norm)
    d=d[d.URL.ne("")&d.target.isin([0,1])].copy()
    conflicts=d.groupby("URL").target.nunique(); bad=set(conflicts[conflicts>1].index); d=d[~d.URL.isin(bad)].drop_duplicates("URL").reset_index(drop=True)
    d["target"]=d.target.astype(int); d["host"]=d.URL.map(host); d=d[d.host.ne("")].reset_index(drop=True)
    if d.target.nunique()!=2:raise ValueError("Need both labels")
    return d,len(bad)
def feats(u):
    s=str(u); low=s.lower()
    try:
        p=urlsplit(s); h=(p.hostname or "").lower(); tld=h.rsplit(".",1)[-1] if "." in h else ""
    except Exception:h="";tld=""
    return {"length":len(s),"dots":s.count("."),"hyphens":s.count("-"),"slashes":s.count("/"),"equals":s.count("="),"https":int(low.startswith("https://")),"http":int(low.startswith("http://")),"at":int("@" in s),"digits":sum(c.isdigit() for c in s),"suspicious_present":int(any(w in low for w in WORDS)),"suspicious_count":sum(low.count(w) for w in WORDS),"suspicious_tld":int(tld in {"zip","mov","top","xyz","click","work","country","gq","tk","fit","ml","ga","cf"}),"subdomains":int(len(h.split("."))>=4),"hyphen_domain":int("-" in h),"digit_domain":int(any(c.isdigit() for c in h)),"host_length":len(h),"long_url":int(len(s)>60),"ip_host":int(bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}",h))),"lookalike_domain":0}
def chars(urls):
    a=np.zeros((len(urls),MAX_LEN),dtype=np.int64)
    for i,u in enumerate(urls):
        for j,c in enumerate(str(u).lower()[:MAX_LEN]): a[i,j]=ord(c)+1 if ord(c)<VOCAB-1 else 1
    return a
def neural(name,Xtr,ytr,Xte,Ctr,Cte):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset,DataLoader
    torch.manual_seed(SEED); torch.set_num_threads(4)
    nf=Xtr.shape[1]
    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.emb=nn.Embedding(VOCAB+1,24,padding_idx=0)
            self.conv=nn.Sequential(nn.Conv1d(24,64,5,padding=2),nn.ReLU(),nn.Conv1d(64,64,3,padding=1),nn.ReLU())
            self.lstm=nn.LSTM(24,32,batch_first=True,bidirectional=True)
            self.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(24,4,64,batch_first=True),1)
            self.vitproj=nn.Linear(24,32); self.vit=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,64,batch_first=True),1)
            self.msg=nn.Linear(24,24); self.gout=nn.Linear(24,32)
            dims={"Character CNN":128,"CNN-LSTM":192,"URL Transformer":24,"Vision Transformer":32,"GNN":32,"Multimodal Fusion":128+nf}
            if name in ("FNN","DNN","Wide & Deep","TabNet"):
                self.deep=nn.Sequential(nn.Linear(nf,128 if name!="FNN" else 64),nn.ReLU(),nn.Dropout(.15),nn.Linear(128 if name!="FNN" else 64,64),nn.ReLU(),nn.Linear(64,1))
                self.wide=nn.Linear(nf,1)
            else:self.head=nn.Sequential(nn.Linear(dims[name],64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,1))
        def forward(self,x,c):
            if name in ("FNN","DNN","TabNet"):return self.deep(x)
            if name=="Wide & Deep":return self.wide(x)+self.deep(x)
            e=self.emb(c)
            if name=="Character CNN":
                z=self.conv(e.transpose(1,2)); z=torch.cat([z.mean(-1),z.amax(-1)],1)
            elif name=="CNN-LSTM":
                z=self.conv(e.transpose(1,2)); z=torch.cat([z.mean(-1),z.amax(-1)],1); l,_=self.lstm(e); z=torch.cat([z,l.mean(1)],1)
            elif name=="URL Transformer":z=self.tr(e).mean(1)
            elif name=="Vision Transformer":
                # ViT-style encoder on a 2D grid of URL characters, not webpage screenshots.
                z=self.vit(self.vitproj(e.reshape(e.shape[0],10,16,24).mean(2))).mean(1)
            elif name=="GNN":
                # Character-chain graph message passing; not a website relationship graph.
                nb=torch.zeros_like(e); nb[:,1:]+=e[:,:-1]; nb[:,:-1]+=e[:,1:]; z=torch.relu(self.msg(e+nb)).mean(1); z=self.gout(z)
            elif name=="Multimodal Fusion":
                q=self.conv(e.transpose(1,2)); z=torch.cat([q.mean(-1),q.amax(-1),x],1)
            return self.head(z)
    m=Net()
    pos=max(1,int((ytr==1).sum())); neg=max(1,int((ytr==0).sum()))
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos))
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    tx=torch.tensor(Xtr,dtype=torch.float32); tc=torch.tensor(Ctr,dtype=torch.long); ty=torch.tensor(ytr.to_numpy(),dtype=torch.float32).view(-1,1)
    dl=DataLoader(TensorDataset(tx,tc,ty),batch_size=BATCH,shuffle=True)
    m.train()
    for ep in range(EPOCHS):
        total=0
        for x,c,y in dl:
            opt.zero_grad(); z=m(x,c); loss=lossfn(z,y); loss.backward(); opt.step(); total+=float(loss.detach())
        print(f"    epoch {ep+1}/{EPOCHS}, mean batch loss={total/max(1,len(dl)):.4f}",flush=True)
    m.eval(); probs=[]
    with torch.no_grad():
        for i in range(0,len(Xte),BATCH):
            probs.extend(torch.sigmoid(m(torch.tensor(Xte[i:i+BATCH],dtype=torch.float32),torch.tensor(Cte[i:i+BATCH],dtype=torch.long))).numpy().ravel())
    p=np.asarray(probs); return (p>=.5).astype(int),p
def main():
    OUT.mkdir(parents=True,exist_ok=True);PRED.mkdir(parents=True,exist_ok=True);GRAPHS.mkdir(parents=True,exist_ok=True)
    print("Loading 208K combined URL data...",flush=True);d,conf=load()
    print(f"Usable URLs={len(d):,}; labels={d.target.value_counts().sort_index().to_dict()}; conflicts excluded={conf}",flush=True)
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(d,d.target,groups=d.host))
    train=d.iloc[tr].reset_index(drop=True);test=d.iloc[te].reset_index(drop=True)
    if set(train.host)&set(test.host):raise RuntimeError("Host leakage detected")
    print(f"Hostname-grouped split: train={len(tr):,}, holdout={len(te):,}, overlap=0",flush=True)
    print("Extracting features and URL character sequences...",flush=True)
    F=pd.DataFrame([feats(u) for u in d.URL]).astype(np.float32)
    Xtr=F.iloc[tr].to_numpy();Xte=F.iloc[te].to_numpy();ytr=train.target;yte=test.target
    C=chars(d.URL.tolist());Ctr=C[tr];Cte=C[te]
    models={}
    models["Logistic Regression"]=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,class_weight="balanced",random_state=SEED))
    models["Random Forest"]=RandomForestClassifier(n_estimators=180,n_jobs=-1,class_weight="balanced",random_state=SEED)
    models["Extra Trees"]=ExtraTreesClassifier(n_estimators=180,n_jobs=-1,class_weight="balanced",random_state=SEED)
    models["Linear SVM"]=make_pipeline(StandardScaler(),LinearSVC(class_weight="balanced",random_state=SEED))
    models["HistGradientBoosting"]=HistGradientBoostingClassifier(max_iter=120,random_state=SEED)
    for nm,mod in [("XGBoost","xgboost"),("LightGBM","lightgbm"),("CatBoost","catboost")]:
        try:
            if nm=="XGBoost":
                from xgboost import XGBClassifier;models[nm]=XGBClassifier(n_estimators=120,max_depth=5,learning_rate=.08,eval_metric="logloss",n_jobs=4,random_state=SEED)
            elif nm=="LightGBM":
                from lightgbm import LGBMClassifier;models[nm]=LGBMClassifier(n_estimators=120,verbosity=-1,n_jobs=4,random_state=SEED)
            else:
                from catboost import CatBoostClassifier;models[nm]=CatBoostClassifier(iterations=120,verbose=False,thread_count=4,random_seed=SEED)
        except Exception as e: print(f"Dependency issue {nm}: {e}",flush=True)
    for n in ["FNN","DNN","Wide & Deep","TabNet","Character CNN","URL Transformer","CNN-LSTM","GNN","Vision Transformer","Multimodal Fusion"]:models[n]="NEURAL"
    # Soft-voting ensemble definitions.
    for n in ["Random Forest + XGBoost","Random Forest + Character CNN","Character CNN + XGBoost","Random Forest + Character CNN + XGBoost"]:models[n]="ENSEMBLE"
    rows=[]
    for i,name in enumerate(NAMES,1):
        print(f"\n[{i}/22] {name}",flush=True);start=time.perf_counter()
        try:
            if name not in models:raise RuntimeError("Model dependency/implementation missing")
            if name=="TabNet":
                from pytorch_tabnet.tab_model import TabNetClassifier
                tm=TabNetClassifier(seed=SEED,verbose=0,device_name="cpu")
                tm.fit(Xtr.astype(np.float32),ytr.to_numpy().astype(np.int64),
                       max_epochs=15,batch_size=2048,virtual_batch_size=256,
                       num_workers=0,drop_last=False)
                score=tm.predict_proba(Xte.astype(np.float32))[:,1]
                pred=(score>=0.5).astype(int);note="Actual pytorch-tabnet TabNetClassifier on lexical features"
            elif models[name]=="NEURAL":
                pred,score=neural(name,Xtr,ytr,Xte,Ctr,Cte); note="PyTorch URL/feature architecture"
            elif models[name]=="ENSEMBLE":
                import xgboost
                parts=[]
                if "Random Forest" in name:
                    rf=RandomForestClassifier(n_estimators=120,n_jobs=-1,class_weight="balanced",random_state=SEED);rf.fit(Xtr,ytr);parts.append(rf.predict_proba(Xte)[:,1])
                if "XGBoost" in name:
                    xg=xgboost.XGBClassifier(n_estimators=100,max_depth=5,learning_rate=.08,eval_metric="logloss",n_jobs=4,random_state=SEED);xg.fit(Xtr,ytr);parts.append(xg.predict_proba(Xte)[:,1])
                if "Character CNN" in name:
                    _,cp=neural("Character CNN",Xtr,ytr,Xte,Ctr,Cte);parts.append(cp)
                score=np.mean(np.vstack(parts),axis=0);pred=(score>=.5).astype(int);note="Soft-voting probabilities; CNN is trained separately"
            else:
                m=models[name];m.fit(Xtr,ytr);pred=np.asarray(m.predict(Xte)).astype(int)
                if hasattr(m,"predict_proba"):
                    pp=m.predict_proba(Xte);score=pp[:,list(m.classes_).index(1)]
                elif hasattr(m,"decision_function"):score=np.asarray(m.decision_function(Xte)).ravel()
                else:score=None
                note="URL lexical-feature classifier"
            tn,fp,fn,tp=confusion_matrix(yte,pred,labels=[0,1]).ravel()
            try:auc=float(roc_auc_score(yte,score)) if score is not None else np.nan
            except Exception:auc=np.nan
            row={"model":name,"split":"hostname_grouped_holdout","status":"OK","n_samples":len(yte),"accuracy":accuracy_score(yte,pred),"precision_phishing":precision_score(yte,pred,zero_division=0),"recall_phishing":recall_score(yte,pred,zero_division=0),"f1_phishing":f1_score(yte,pred,zero_division=0),"false_positive_rate":fp/(fp+tn) if tn+fp else np.nan,"roc_auc":auc,"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"train_seconds":round(time.perf_counter()-start,2),"notes":note,"error":""}
            pd.DataFrame({"URL":test.URL,"actual_label":yte,"predicted_label":pred,"phishing_score":score if score is not None else np.nan}).to_csv(PRED/f"{slug(name)}_holdout.csv",index=False)
            print(f"  accuracy={row['accuracy']:.4f} F1={row['f1_phishing']:.4f} FPR={row['false_positive_rate']:.4f} time={row['train_seconds']:.1f}s",flush=True)
        except Exception as e:
            row={"model":name,"split":"hostname_grouped_holdout","status":"ERROR","error":f"{type(e).__name__}: {e}","train_seconds":round(time.perf_counter()-start,2)}
            print("  ERROR:",row["error"],flush=True)
        rows.append(row);pd.DataFrame(rows).to_csv(OUT/"model_comparison_metrics.csv",index=False)
    metrics=pd.DataFrame(rows);metrics.to_json(OUT/"model_comparison_metrics.json",orient="records",indent=2)
    pd.DataFrame([{"model":r["model"],"status":r["status"],"error":r.get("error","")} for r in rows]).to_csv(OUT/"model_status.csv",index=False)
    pd.DataFrame([{"item":"unique_urls","value":len(d)},{"item":"legitimate","value":int((d.target==0).sum())},{"item":"phishing","value":int((d.target==1).sum())},{"item":"train_rows","value":len(tr)},{"item":"holdout_rows","value":len(te)},{"item":"train_hosts","value":train.host.nunique()},{"item":"holdout_hosts","value":test.host.nunique()},{"item":"shared_hosts","value":0},{"item":"conflicting_label_urls_excluded","value":conf}]).to_csv(OUT/"dataset_summary.csv",index=False)
    good=metrics[metrics.status=="OK"]
    for col,title,file in [("accuracy","Accuracy","accuracy_comparison.png"),("precision_phishing","Phishing precision","precision_comparison.png"),("recall_phishing","Phishing recall","recall_comparison.png"),("f1_phishing","Phishing F1","f1_score_comparison.png"),("false_positive_rate","False-positive rate","false_positive_rate_comparison.png"),("roc_auc","ROC-AUC","roc_auc_comparison.png"),("train_seconds","Training time (seconds)","training_time_comparison.png")]:
        dd=good.dropna(subset=[col]).sort_values(col)
        if dd.empty:continue
        plt.figure(figsize=(12,max(5,.38*len(dd))));plt.barh(dd.model,dd[col]);plt.xlabel(title);plt.title(title+" — hostname-grouped holdout");plt.gca().invert_yaxis();plt.tight_layout();plt.savefig(GRAPHS/file,dpi=160);plt.close()
    summary={"experiment":"22-model internal comparison","dataset_rows":len(d),"train_rows":len(tr),"holdout_rows":len(te),"hostname_overlap":0,"models_requested":22,"models_successfully_evaluated":int((metrics.status=="OK").sum()),"labels":{"0":"legitimate","1":"phishing"},"model_notes":{"TabNet":"Uses pytorch-tabnet TabNetClassifier on lexical features.","GNN":"Character-chain message passing, not website relationship graph.","Vision Transformer":"Transformer over a 2D URL-character grid, not screenshot pixels.","Multimodal Fusion":"Combines URL lexical features and character sequence."}}
    (OUT/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"\nFINISHED. {int((metrics.status=='OK').sum())}/22 ran successfully.",flush=True)
    print("Metrics:",OUT/"model_comparison_metrics.csv",flush=True);print("Graphs:",GRAPHS,flush=True)
if __name__=="__main__":main()
