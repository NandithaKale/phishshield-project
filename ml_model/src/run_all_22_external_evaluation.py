from pathlib import Path
from urllib.parse import urlsplit
import json, re, time, warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")
SEED, MAX_LEN, VOCAB, EPOCHS, BATCH, THRESHOLD = 42, 160, 128, 4, 512, 0.5
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml_model" / "external_data"
OUT = ROOT / "ml_model" / "results" / "all_22_external_evaluation"
PRED = OUT / "predictions"
PRIMARY = DATA / "labelled_test" / "phishing_url_dataset.csv"
SECOND = DATA / "alternative_test" / "clean_external_test.csv"
LEGITPHISH = DATA / "legitphish" / "url_features_extracted1.csv"
BALANCED = DATA / "new_test" / "balanced_urls_340k.csv"
OPENPHISH = DATA / "openphish_feed.txt"
NAMES = [
    "Logistic Regression", "Random Forest", "Extra Trees", "Linear SVM",
    "XGBoost", "LightGBM", "HistGradientBoosting", "CatBoost", "FNN", "DNN",
    "Wide & Deep", "TabNet", "Character CNN", "URL Transformer",
    "Random Forest + XGBoost", "Random Forest + Character CNN",
    "Character CNN + XGBoost", "Random Forest + Character CNN + XGBoost",
    "CNN-LSTM", "GNN", "Vision Transformer", "Multimodal Fusion",
]
WORDS = ("login","verify","update","secure","account","bank","password","signin","confirm","wallet","paypal","support","recover","auth","security","invoice")


def log(s): print("[External22]", s, flush=True)
def norm(x):
    if pd.isna(x): return ""
    s = str(x).strip()
    if not s: return ""
    if "://" not in s: s = "https://" + s
    try:
        p = urlsplit(s); h = (p.hostname or "").lower()
        if not h: return s.lower().split("#",1)[0]
        try: port = p.port
        except ValueError: port = None
        return f"{p.scheme.lower()}://{h}{':' + str(port) if port else ''}{p.path}" + (f"?{p.query}" if p.query else "")
    except Exception: return s.lower().split("#",1)[0]
def host(u):
    try: return (urlsplit(u).hostname or "").lower()
    except Exception: return ""
def slug(s): return re.sub(r"[^a-z0-9]+","_",s.lower()).strip("_")
def read_csv(p):
    try: return pd.read_csv(p,encoding="utf-8-sig",low_memory=False)
    except UnicodeDecodeError: return pd.read_csv(p,encoding="latin-1",low_memory=False)

def load_training():
    if not PRIMARY.exists() or not SECOND.exists(): raise FileNotFoundError("Required training CSV is missing.")
    a, b = read_csv(PRIMARY), read_csv(SECOND)
    if not {"URL","Label"}.issubset(a.columns) or not {"URL","Label"}.issubset(b.columns):
        raise ValueError("Training CSVs must have URL and Label columns.")
    a["target"] = pd.to_numeric(a.Label,errors="coerce").map({0:1,1:0})
    b["target"] = pd.to_numeric(b.Label,errors="coerce").map({0:0,1:1})
    d = pd.concat([a[["URL","target"]],b[["URL","target"]]],ignore_index=True)
    d["URL"] = d.URL.map(norm); d = d[d.URL.ne("") & d.target.isin([0,1])].copy()
    conflicts = d.groupby("URL").target.nunique()
    bad = set(conflicts[conflicts > 1].index)
    d = d[~d.URL.isin(bad)].drop_duplicates("URL").reset_index(drop=True)
    d["target"] = d.target.astype(int); d["host"] = d.URL.map(host)
    d = d[d.host.ne("")].reset_index(drop=True)
    if d.target.nunique() != 2: raise ValueError("Training data must contain both classes.")
    log(f"Training rows={len(d):,}; labels={d.target.value_counts().to_dict()}; conflicts excluded={len(bad):,}")
    return d

def feats(u):
    s=str(u); low=s.lower()
    try:
        p=urlsplit(s); h=(p.hostname or "").lower(); tld=h.rsplit(".",1)[-1] if "." in h else ""
    except Exception: h=""; tld=""
    return {"length":len(s),"dots":s.count("."),"hyphens":s.count("-"),"slashes":s.count("/"),"equals":s.count("="),"https":int(low.startswith("https://")),"http":int(low.startswith("http://")),"at":int("@" in s),"digits":sum(c.isdigit() for c in s),"suspicious_present":int(any(w in low for w in WORDS)),"suspicious_count":sum(low.count(w) for w in WORDS),"suspicious_tld":int(tld in {"zip","mov","top","xyz","click","work","country","gq","tk","fit","ml","ga","cf"}),"subdomains":int(len(h.split("."))>=4),"hyphen_domain":int("-" in h),"digit_domain":int(any(c.isdigit() for c in h)),"host_length":len(h),"long_url":int(len(s)>60),"ip_host":int(bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}",h))),"lookalike_domain":0}
FEATURES=list(feats("https://example.com").keys())
def feature_matrix(urls,label):
    rows=[feats(u) for u in urls]
    log(f"{label}: extracted {len(rows):,} lexical feature rows")
    return pd.DataFrame(rows,columns=FEATURES).astype(np.float32).to_numpy()
def chars(urls,label):
    out=np.zeros((len(urls),MAX_LEN),dtype=np.int64)
    for i,u in enumerate(urls):
        for j,c in enumerate(str(u).lower()[:MAX_LEN]): out[i,j]=ord(c)+1 if ord(c)<VOCAB-1 else 1
    log(f"{label}: encoded {len(urls):,} URLs as characters")
    return out

def load_external():
    out={}
    if LEGITPHISH.exists():
        d=read_csv(LEGITPHISH)
        if {"URL","ClassLabel"}.issubset(d.columns):
            d["y"]=pd.to_numeric(d.ClassLabel,errors="coerce").map({0:1,1:0})
            d["URL"]=d.URL.map(norm); d=d.dropna(subset=["y"]); d=d[d.URL.ne("")].copy()
            d["y"]=d.y.astype(int); out["LegitPhish"]=d[["URL","y"]].drop_duplicates("URL")
    if BALANCED.exists():
        d=read_csv(BALANCED); cols={str(c).lower():c for c in d.columns}
        if "url" in cols and "label" in cols:
            d=d[[cols["url"],cols["label"]]].copy(); d.columns=["URL","raw"]
            d["y"]=d.raw.astype(str).str.strip().str.lower().map({"legitimate":0,"phishing":1})
            d["URL"]=d.URL.map(norm); d=d.dropna(subset=["y"]); d=d[d.URL.ne("")].copy()
            d["y"]=d.y.astype(int); out["Balanced_340k"]=d[["URL","y"]].drop_duplicates("URL")
    if OPENPHISH.exists():
        urls=[norm(x) for x in OPENPHISH.read_text(encoding="utf-8",errors="replace").splitlines() if x.strip()]
        d=pd.DataFrame({"URL":urls}); d=d[d.URL.ne("")].drop_duplicates("URL")
        if len(d): out["OpenPhish"]=d.reset_index(drop=True)
    if not out: raise RuntimeError("No supported external datasets were found.")
    for n,d in out.items(): log(f"External {n}: {len(d):,} rows")
    log("Web Fraud omitted because verdict 0/1 meaning has not been verified.")
    return out

def score(model,X):
    if hasattr(model,"predict_proba"):
        p=model.predict_proba(X); return p[:,list(model.classes_).index(1)]
    if hasattr(model,"decision_function"): return np.asarray(model.decision_function(X)).ravel()
    return np.asarray(model.predict(X),dtype=float)
def get_metrics(y,pred,scores):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    try: auc=float(roc_auc_score(y,scores))
    except Exception: auc=np.nan
    return {"n_samples":len(y),"accuracy":accuracy_score(y,pred),"precision_phishing":precision_score(y,pred,zero_division=0),"recall_phishing":recall_score(y,pred,zero_division=0),"f1_phishing":f1_score(y,pred,zero_division=0),"false_positive_rate":fp/(fp+tn) if tn+fp else np.nan,"roc_auc":auc,"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}

def classic_models():
    m={
      "Logistic Regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,class_weight="balanced",random_state=SEED)),
      "Random Forest":RandomForestClassifier(n_estimators=180,n_jobs=-1,class_weight="balanced",random_state=SEED),
      "Extra Trees":ExtraTreesClassifier(n_estimators=180,n_jobs=-1,class_weight="balanced",random_state=SEED),
      "Linear SVM":make_pipeline(StandardScaler(),LinearSVC(class_weight="balanced",random_state=SEED)),
      "HistGradientBoosting":HistGradientBoostingClassifier(max_iter=120,random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        m["XGBoost"]=XGBClassifier(n_estimators=120,max_depth=5,learning_rate=.08,eval_metric="logloss",n_jobs=4,random_state=SEED)
    except Exception as e: log(f"XGBoost unavailable: {e}")
    try:
        from lightgbm import LGBMClassifier
        m["LightGBM"]=LGBMClassifier(n_estimators=120,verbosity=-1,n_jobs=4,random_state=SEED)
    except Exception as e: log(f"LightGBM unavailable: {e}")
    try:
        from catboost import CatBoostClassifier
        m["CatBoost"]=CatBoostClassifier(iterations=120,verbose=False,thread_count=4,random_seed=SEED)
    except Exception as e: log(f"CatBoost unavailable: {e}")
    return m

def train_neural(name,X,y,C):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset,DataLoader
    torch.manual_seed(SEED); torch.set_num_threads(4)
    nf=X.shape[1]
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb=nn.Embedding(VOCAB+1,24,padding_idx=0)
            self.conv=nn.Sequential(nn.Conv1d(24,64,5,padding=2),nn.ReLU(),nn.Conv1d(64,64,3,padding=1),nn.ReLU())
            self.lstm=nn.LSTM(24,32,batch_first=True,bidirectional=True)
            self.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(24,4,64,batch_first=True),1)
            self.vitproj=nn.Linear(24,32); self.vit=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,64,batch_first=True),1)
            self.msg=nn.Linear(24,24); self.gout=nn.Linear(24,32)
            dims={"Character CNN":128,"CNN-LSTM":192,"URL Transformer":24,"Vision Transformer":32,"GNN":32,"Multimodal Fusion":128+nf}
            if name in ("FNN","DNN","Wide & Deep","TabNet"):
                width=64 if name=="FNN" else 128
                self.deep=nn.Sequential(nn.Linear(nf,width),nn.ReLU(),nn.Dropout(.15),nn.Linear(width,64),nn.ReLU(),nn.Linear(64,1))
                self.wide=nn.Linear(nf,1)
            else:self.head=nn.Sequential(nn.Linear(dims[name],64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,1))
        def forward(self,x,c):
            if name in ("FNN","DNN","TabNet"): return self.deep(x)
            if name=="Wide & Deep": return self.wide(x)+self.deep(x)
            e=self.emb(c)
            if name=="Character CNN":
                z=self.conv(e.transpose(1,2)); z=torch.cat([z.mean(-1),z.amax(-1)],1)
            elif name=="CNN-LSTM":
                z=self.conv(e.transpose(1,2)); z=torch.cat([z.mean(-1),z.amax(-1)],1); l,_=self.lstm(e); z=torch.cat([z,l.mean(1)],1)
            elif name=="URL Transformer": z=self.tr(e).mean(1)
            elif name=="Vision Transformer": z=self.vit(self.vitproj(e.reshape(e.shape[0],10,16,24).mean(2))).mean(1)
            elif name=="GNN":
                nb=torch.zeros_like(e); nb[:,1:]+=e[:,:-1]; nb[:,:-1]+=e[:,1:]; z=torch.relu(self.msg(e+nb)).mean(1); z=self.gout(z)
            elif name=="Multimodal Fusion":
                q=self.conv(e.transpose(1,2)); z=torch.cat([q.mean(-1),q.amax(-1),x],1)
            else: raise ValueError(name)
            return self.head(z)
    model=Net()
    pos=max(1,int((y==1).sum())); neg=max(1,int((y==0).sum()))
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos,dtype=torch.float32))
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    tx=torch.tensor(X,dtype=torch.float32); tc=torch.tensor(C,dtype=torch.long); ty=torch.tensor(y,dtype=torch.float32).view(-1,1)
    dl=DataLoader(TensorDataset(tx,tc,ty),batch_size=BATCH,shuffle=True)
    model.train()
    for ep in range(EPOCHS):
        total=0.0
        for xb,cb,yb in dl:
            opt.zero_grad(); loss=lossfn(model(xb,cb),yb); loss.backward(); opt.step(); total+=float(loss.detach())
        log(f"{name}: epoch {ep+1}/{EPOCHS} loss={total/max(1,len(dl)):.4f}")
    model.eval()
    def predict(Xtest,Ctest,label):
        probs=[]
        with torch.no_grad():
            for i in range(0,len(Xtest),BATCH):
                xb=torch.tensor(Xtest[i:i+BATCH],dtype=torch.float32)
                cb=torch.tensor(Ctest[i:i+BATCH],dtype=torch.long)
                probs.extend(torch.sigmoid(model(xb,cb)).numpy().ravel())
        log(f"{name}: predicted {len(Xtest):,} URLs on {label}")
        return np.asarray(probs)
    return predict

def main():
    OUT.mkdir(parents=True,exist_ok=True); PRED.mkdir(parents=True,exist_ok=True)
    train=load_training(); ext=load_external()
    train_urls=train.URL.tolist(); y=train.target.to_numpy(dtype=int)
    X=feature_matrix(train_urls,"training"); C=chars(train_urls,"training")
    train_urls_set=set(train.URL); train_hosts=set(train.host)
    prepared={}
    for name,df in ext.items():
        df=df.copy(); df["URL"]=df.URL.map(norm); df=df[df.URL.ne("")].drop_duplicates("URL")
        overlap=df.URL.isin(train_urls_set); removed=int(overlap.sum()); df=df.loc[~overlap].copy()
        df["host"]=df.URL.map(host); host_overlap=int(df.host.isin(train_hosts).sum())
        if df.empty: log(f"{name}: no rows after exact overlap removal"); continue
        log(f"{name}: {len(df):,} retained; {removed:,} exact overlaps removed; {host_overlap:,} retained URLs share a training host")
        prepared[name]={"df":df.reset_index(drop=True),"X":feature_matrix(df.URL.tolist(),name),"C":chars(df.URL.tolist(),name),"removed":removed,"host_overlap":host_overlap}
    classic=classic_models(); rows=[]; statuses=[]
    for idx,name in enumerate(NAMES,1):
        start=time.perf_counter(); log(f"\n[{idx}/22] {name}")
        try:
            # Train each architecture once on the same internal training set.
            if name in classic:
                model=classic[name]; model.fit(X,y)
                def predict_fn(item,label): return score(model,item["X"])
            elif name=="TabNet":
                from pytorch_tabnet.tab_model import TabNetClassifier
                model=TabNetClassifier(seed=SEED,verbose=0,device_name="cpu")
                model.fit(X.astype(np.float32),y.astype(np.int64),max_epochs=15,batch_size=2048,virtual_batch_size=256,num_workers=0,drop_last=False)
                def predict_fn(item,label): return model.predict_proba(item["X"].astype(np.float32))[:,1]
            elif name in ("FNN","DNN","Wide & Deep","Character CNN","URL Transformer","CNN-LSTM","GNN","Vision Transformer","Multimodal Fusion"):
                neural_predict=train_neural(name,X,y,C)
                def predict_fn(item,label): return neural_predict(item["X"],item["C"],label)
            elif name in ("Random Forest + XGBoost","Random Forest + Character CNN","Character CNN + XGBoost","Random Forest + Character CNN + XGBoost"):
                parts=[]
                rf=xgb=None
                if "Random Forest" in name:
                    rf=RandomForestClassifier(n_estimators=120,n_jobs=-1,class_weight="balanced",random_state=SEED); rf.fit(X,y)
                if "XGBoost" in name:
                    from xgboost import XGBClassifier
                    xgb=XGBClassifier(n_estimators=100,max_depth=5,learning_rate=.08,eval_metric="logloss",n_jobs=4,random_state=SEED); xgb.fit(X,y)
                cnn_predict=train_neural("Character CNN",X,y,C) if "Character CNN" in name else None
                def predict_fn(item,label):
                    ps=[]
                    if rf is not None: ps.append(rf.predict_proba(item["X"])[:,list(rf.classes_).index(1)])
                    if xgb is not None: ps.append(xgb.predict_proba(item["X"])[:,list(xgb.classes_).index(1)])
                    if cnn_predict is not None: ps.append(cnn_predict(item["X"],item["C"],label))
                    return np.mean(np.vstack(ps),axis=0)
            else: raise RuntimeError("No implementation for model")
            elapsed=round(time.perf_counter()-start,2)
            for ds,item in prepared.items():
                df=item["df"]; scores=predict_fn(item,ds); pred=(scores>=THRESHOLD).astype(int)
                row={"model":name,"dataset":ds,"status":"OK","training_seconds":elapsed,"exact_url_overlaps_removed":item["removed"],"training_host_overlap_count":item["host_overlap"]}
                if "y" in df.columns: row.update(get_metrics(df.y.to_numpy(dtype=int),pred,scores))
                else: row.update({"n_samples":len(df),"predicted_phishing":int(pred.sum()),"predicted_legitimate":int(len(pred)-pred.sum()),"phishing_detection_rate":float(pred.mean()) if len(pred) else np.nan,"note":"Phishing-only feed; accuracy and FPR are not available."})
                rows.append(row)
                out=pd.DataFrame({"URL":df.URL.to_numpy(),"predicted_label":pred,"phishing_score":scores})
                if "y" in df.columns: out["actual_label"]=df.y.to_numpy(dtype=int)
                out.to_csv(PRED/f"{slug(name)}_{ds.lower()}_predictions.csv",index=False)
                log(f"{name} / {ds}: " + (f"accuracy={row['accuracy']:.4f}, F1={row['f1_phishing']:.4f}, FPR={row['false_positive_rate']:.4f}" if "y" in df.columns else f"predicted phishing={pred.sum()}/{len(pred)}"))
            statuses.append({"model":name,"status":"OK","error":"","elapsed_seconds":elapsed})
        except Exception as e:
            error=f"{type(e).__name__}: {e}"; log(f"ERROR {name}: {error}")
            statuses.append({"model":name,"status":"ERROR","error":error,"elapsed_seconds":round(time.perf_counter()-start,2)})
            rows.append({"model":name,"dataset":"ALL","status":"ERROR","error":error})
        pd.DataFrame(rows).to_csv(OUT/"external_metrics.csv",index=False)
        pd.DataFrame(statuses).to_csv(OUT/"model_status.csv",index=False)
    summary={"experiment":"Freshly retrained 22-model external evaluation","models_requested":22,"models_completed":sum(s["status"]=="OK" for s in statuses),"label_convention":{"0":"legitimate","1":"phishing"},"threshold":THRESHOLD,"feature_names":FEATURES,"datasets":{n:len(i["df"]) for n,i in prepared.items()},"limitations":["Fresh models were retrained; weights from previous run were not saved/reused.","OpenPhish is phishing-only and cannot independently establish accuracy or false-positive rate.","Web Fraud skipped because verdict mapping was not verified.","lookalike_domain remains fixed at zero, matching the original comparison.","GNN is character-chain message passing; Vision Transformer operates on URL characters, not screenshots; fusion combines lexical features and URL characters."]}
    (OUT/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    log(f"FINISHED: {summary['models_completed']}/22 models completed.")
    log(f"Metrics: {OUT/'external_metrics.csv'}")
    log(f"Statuses: {OUT/'model_status.csv'}")
    log(f"Predictions: {PRED}")
if __name__=="__main__": main()
