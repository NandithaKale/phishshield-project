"""
PhishShield COMPLETE comparison runner.
Run from repo root:
    python ml_model/src/run_all_models_seen_unseen.py

Writes separate files:
  ml_model/final_all_models/seen_test_metrics.csv
  ml_model/final_all_models/unseen_test_metrics.csv
  ml_model/final_all_models/model_status.csv
  ml_model/final_all_models/phishshield_all_models.xlsx

Data policy:
- Internal CSV: good=0 legitimate, bad=1 phishing.
- A stratified 80/20 split is used for seen-test comparison; training rows are
  balanced with class weights (not by throwing away the majority class).
- For external evaluation, models are refit on ALL cleaned internal URLs after
  removing exact URL overlap with external datasets.
- External source labels are configured as 0=phishing, 1=legitimate and mapped
  to PhishShield convention 0=legitimate, 1=phishing. VERIFY source documentation.
- GNN, ViT, and multimodal fusion are explicitly marked NOT RUN because this
  repository's supplied inputs are URLs + 19 URL features, not graph nodes/edges,
  page screenshots, HTML/DOM, or multimodal tensors. Do not fabricate those inputs.
"""

from __future__ import annotations
import os, sys, time, json, warnings, traceback
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
    GradientBoostingClassifier, VotingClassifier, StackingClassifier, RandomForestRegressor
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import joblib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml_model" / "final_all_models"
OUT.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = ROOT / "ml_model" / "dataset" / "phishing.csv"
BALANCED_EXTERNAL_FILE = ROOT / "ml_model" / "external_data" / "labelled_test" / "phishing_url_dataset.csv"
LEGITPHISH_FILE = ROOT / "ml_model" / "external_data" / "legitphish" / "url_features_extracted1.csv"
BALANCED_EXTERNAL_LABEL_MAP = {0: 1, 1: 0}
LEGITPHISH_LABEL_MAP = {0: 1, 1: 0}
N_FEATURES = 19
SEED = 42
MAX_NEURAL_TRAIN = 160_000  # cap only neural fitting for CPU runtime; classic models use all rows
MAX_URL_LEN = 200
VOCAB_SIZE = 128


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_extractor():
    sys.path.insert(0, str(ROOT))
    from shared.feature_extractor import extract_features
    return extract_features


def url_text(x):
    return "" if pd.isna(x) else str(x).strip()


def internal_label(x):
    if pd.isna(x): return None
    s = str(x).strip().lower()
    if s == "good": return 0
    if s == "bad": return 1
    if s in ("0", "0.0"): return 0
    if s in ("1", "1.0"): return 1
    return None


def read_csv_fallback(path, **kwargs):
    err = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            log(f"Read {path.name} with encoding={enc}")
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            err = e
    raise RuntimeError(f"Unable to read {path}: {err}")


def load_internal():
    d = read_csv_fallback(TRAIN_FILE, usecols=["URL", "Label"], on_bad_lines="warn")
    d["URL"] = d["URL"].map(url_text)
    d["target"] = d["Label"].map(internal_label)
    d = d[(d.URL != "") & d.target.notna()].copy()
    d["target"] = d.target.astype(int)
    return d.drop_duplicates("URL").reset_index(drop=True)[["URL", "target"]]


def load_external(path, label_col, label_map, name):
    if not path.exists():
        log(f"{name}: file missing, skipping")
        return None
    d = read_csv_fallback(path, on_bad_lines="warn")
    if "URL" not in d or label_col not in d:
        log(f"{name}: expected URL and {label_col} columns missing; skipping")
        return None
    d = d[["URL", label_col]].copy()
    d["URL"] = d.URL.map(url_text)
    d = d[(d.URL != "") & d[label_col].notna()].copy()
    d["target"] = d[label_col].map(lambda x: label_map.get(int(float(x))) if str(x).strip() else np.nan)
    d = d[d.target.notna()].copy()
    d["target"] = d.target.astype(int)
    return d.drop_duplicates("URL").reset_index(drop=True)[["URL", "target"]]


def feature_matrix(urls, extractor, cache, name):
    rows, bad = [], 0
    for i, raw in enumerate(urls):
        u = url_text(raw)
        if u not in cache:
            try:
                f = np.asarray(extractor(u), dtype=np.float32).reshape(-1)
                if not u or f.size != N_FEATURES or not np.isfinite(f).all():
                    raise ValueError("invalid URL/features")
            except Exception as e:
                bad += 1
                f = np.zeros(N_FEATURES, dtype=np.float32)
                if bad <= 5: log(f"Warning {name} row {i}: {u[:70]!r} -> zero features ({e})")
            cache[u] = f
        rows.append(cache[u])
        if (i+1) % 50000 == 0: log(f"{name} features {i+1:,}/{len(urls):,}")
    X = np.vstack(rows).astype(np.float32) if rows else np.empty((0, N_FEATURES), np.float32)
    log(f"{name}: X={X.shape}; invalid URL rows={bad:,}")
    return X, bad


def url_chars(urls):
    X = np.zeros((len(urls), MAX_URL_LEN), dtype=np.int32)
    for i, u in enumerate(urls):
        for j, ch in enumerate(str(u)[:MAX_URL_LEN]):
            code = ord(ch)
            X[i, j] = code if 0 < code < VOCAB_SIZE else 1
    return X


def make_tf_model(kind):
    import tensorflow as tf
    inp = tf.keras.Input(shape=(MAX_URL_LEN,), dtype="int32")
    x = tf.keras.layers.Embedding(VOCAB_SIZE, 32, mask_zero=False)(inp)
    if kind == "Character CNN":
        x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
    elif kind == "CNN-LSTM":
        x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
    elif kind == "FNN":
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.25)(x)
    elif kind == "DNN":
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
    elif kind == "Wide & Deep":
        wide = tf.keras.layers.Flatten()(x)
        deep = tf.keras.layers.GlobalAveragePooling1D()(x)
        deep = tf.keras.layers.Dense(128, activation="relu")(deep)
        x = tf.keras.layers.Concatenate()([wide, deep])
    elif kind == "URL Transformer":
        pos = tf.range(start=0, limit=MAX_URL_LEN, delta=1)
        pos_emb = tf.keras.layers.Embedding(MAX_URL_LEN, 32)(pos)
        x = x + pos_emb
        att = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + att)
        ff = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(ff)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
    else:
        raise ValueError(kind)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out, name=kind.replace(" ", "_"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def models_classic():
    m = {
        "Logistic Regression": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=700, class_weight="balanced", solver="liblinear", random_state=SEED)),
        "Random Forest": RandomForestClassifier(n_estimators=160, n_jobs=-1, class_weight="balanced_subsample", random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=160, n_jobs=-1, class_weight="balanced", random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=2, random_state=SEED),
        "Gaussian Naive Bayes": make_pipeline(SimpleImputer(), GaussianNB()),
        "Linear SVM (calibrated)": make_pipeline(SimpleImputer(), StandardScaler(), CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED), cv=3)),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=140, learning_rate=0.08, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=2, random_state=SEED),
        "MLP (feature-based)": make_pipeline(SimpleImputer(), StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=35, early_stopping=True, random_state=SEED)),
    }
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = XGBClassifier(n_estimators=180, max_depth=7, learning_rate=.08, subsample=.85, colsample_bytree=.9, eval_metric="logloss", n_jobs=-1, tree_method="hist", random_state=SEED)
    except Exception as e: log(f"XGBoost unavailable: {e}")
    try:
        from lightgbm import LGBMClassifier
        m["LightGBM"] = LGBMClassifier(n_estimators=180, learning_rate=.08, num_leaves=31, verbosity=-1, n_jobs=-1, random_state=SEED, class_weight="balanced")
    except Exception as e: log(f"LightGBM unavailable: {e}")
    try:
        from catboost import CatBoostClassifier
        m["CatBoost"] = CatBoostClassifier(iterations=180, depth=7, learning_rate=.08, verbose=False, thread_count=-1, random_seed=SEED, auto_class_weights="Balanced")
    except Exception as e: log(f"CatBoost unavailable: {e}")
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        m["TabNet"] = TabNetClassifier(n_d=16, n_a=16, n_steps=3, gamma=1.3, seed=SEED, verbose=0)
    except Exception as e: log(f"TabNet unavailable: {e}")
    return m


def score_metrics(model_name, dataset, y, pred, score, train_s=None, infer_s=None, size=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    try: auc = roc_auc_score(y, score)
    except Exception: auc = np.nan
    return {"model":model_name,"dataset":dataset,"n_samples":len(y),
        "accuracy":accuracy_score(y,pred),"precision_phishing":precision_score(y,pred,zero_division=0),
        "phishing_recall":recall_score(y,pred,zero_division=0),"f1_phishing":f1_score(y,pred,zero_division=0),
        "false_positive_rate":fp/(fp+tn) if fp+tn else np.nan,"roc_auc":auc,
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
        "train_seconds":train_s,"inference_seconds":infer_s,"model_size_mb":size,"status":"OK"}


def get_scores(model, X):
    if hasattr(model, "predict_proba"): return model.predict_proba(X)[:,1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return (z-z.min())/(np.ptp(z)+1e-12)
    return model.predict(X).astype(float)


def main():
    log("Loading internal and external datasets")
    extractor = load_extractor()
    internal = load_internal()
    ext1 = load_external(BALANCED_EXTERNAL_FILE, "Label", BALANCED_EXTERNAL_LABEL_MAP, "BalancedExternal")
    ext2 = load_external(LEGITPHISH_FILE, "ClassLabel", LEGITPHISH_LABEL_MAP, "LegitPhish")
    external = [(n,d) for n,d in [("BalancedExternal",ext1),("LegitPhish",ext2)] if d is not None]
    external_urls = set()
    for _,d in external: external_urls.update(d.URL.str.lower().tolist())
    before = len(internal)
    if external_urls: internal = internal[~internal.URL.str.lower().isin(external_urls)].reset_index(drop=True)
    log(f"Internal cleaned rows={before:,}; removed external URL overlap={before-len(internal):,}; remaining={len(internal):,}; class counts={internal.target.value_counts().sort_index().to_dict()}")

    # Seen test split, feature extraction shared safely through URL-keyed cache.
    tr_df, te_df = train_test_split(internal, test_size=.20, random_state=SEED, stratify=internal.target)
    cache = {}
    Xtr, _ = feature_matrix(tr_df.URL.tolist(), extractor, cache, "SeenTrain")
    Xte, _ = feature_matrix(te_df.URL.tolist(), extractor, cache, "SeenTest")
    ytr = tr_df.target.to_numpy(dtype=np.int32); yte = te_df.target.to_numpy(dtype=np.int32)
    Xfull, _ = feature_matrix(internal.URL.tolist(), extractor, cache, "FullTrain")
    yfull = internal.target.to_numpy(dtype=np.int32)
    Xext = {}
    for n,d in external:
        Xext[n], _ = feature_matrix(d.URL.tolist(), extractor, cache, n)

    classic = models_classic()
    seen_rows, unseen_rows, status_rows = [], [], []
    seen_scores = {}
    unseen_scores = {ds: {} for ds, _ in external}
    # Train/evaluate classic models on seen split; then refit on full data for unseen.
    for name, model in classic.items():
        try:
            log(f"[SEEN] Training {name} on {len(ytr):,}")
            t0=time.perf_counter()
            if name == "TabNet":
                model.fit(Xtr.astype(np.float32), ytr, max_epochs=35, patience=7, batch_size=2048, virtual_batch_size=256)
            else:
                sw = compute_sample_weight("balanced", ytr) if name in ("HistGradientBoosting","Gradient Boosting","XGBoost","CatBoost") else None
                if sw is not None and name not in ("XGBoost","CatBoost"):
                    model.fit(Xtr,ytr,sample_weight=sw)
                elif name == "XGBoost":
                    # Fit without explicit sample weights; the held-out split remains stratified.
                    model.fit(Xtr,ytr)
                elif name == "CatBoost":
                    model.fit(Xtr,ytr)
                else: model.fit(Xtr,ytr)
            ts=time.perf_counter()-t0
            t0=time.perf_counter(); sc=get_scores(model,Xte); ps=time.perf_counter()-t0
            pr=(sc>=.5).astype(int)
            seen_rows.append(score_metrics(name,"SeenTest",yte,pr,sc,ts,ps))
            seen_scores[name]=sc
            status_rows.append({"model":name,"family":"traditional/tabular","seen_status":"OK","unseen_status":"PENDING","notes":""})
            # Refit model on ALL cleaned internal rows for external-only evaluation.
            log(f"[UNSEEN] Refitting {name} on all {len(yfull):,} internal rows")
            full_model = clone(model)
            t0=time.perf_counter()
            if name == "TabNet":
                full_model.fit(Xfull.astype(np.float32), yfull, max_epochs=35, patience=7, batch_size=2048, virtual_batch_size=256)
            else:
                sw = compute_sample_weight("balanced", yfull) if name in ("HistGradientBoosting","Gradient Boosting","XGBoost","CatBoost") else None
                if sw is not None and name in ("HistGradientBoosting","Gradient Boosting"):
                    full_model.fit(Xfull,yfull,sample_weight=sw)
                else: full_model.fit(Xfull,yfull)
            full_ts=time.perf_counter()-t0
            model_path=OUT/(name.lower().replace(" ","_").replace("+","plus").replace("(","").replace(")","")+".joblib")
            try: joblib.dump(full_model,model_path,compress=2); size=model_path.stat().st_size/(1024*1024)
            except Exception: size=np.nan
            for ds,d in external:
                t0=time.perf_counter(); sc2=get_scores(full_model,Xext[ds]); inf=time.perf_counter()-t0
                pr2=(sc2>=.5).astype(int)
                row=score_metrics(name,ds,d.target.to_numpy(dtype=int),pr2,sc2,full_ts,inf,size)
                unseen_rows.append(row)
                unseen_scores[ds][name] = sc2
            # keep probability outputs for the requested ensembles
            del full_model
        except Exception as e:
            log(f"ERROR {name}: {e}"); traceback.print_exc()
            status_rows.append({"model":name,"family":"traditional/tabular","seen_status":"ERROR","unseen_status":"ERROR","notes":str(e)})

    # Neural URL models. For speed on CPU, fit neural models on a stratified sample
    # up to MAX_NEURAL_TRAIN, clearly marked in status; classical models use full rows.
    neural_names=["FNN","DNN","Wide & Deep","Character CNN","CNN-LSTM","URL Transformer"]
    try:
        import tensorflow as tf
        n_neural=min(len(tr_df),MAX_NEURAL_TRAIN)
        if n_neural < len(tr_df):
            nn_df, _ = train_test_split(tr_df, train_size=n_neural, random_state=SEED, stratify=tr_df.target)
        else: nn_df=tr_df
        Xn=url_chars(nn_df.URL.tolist()); yn=nn_df.target.to_numpy(dtype=np.int32)
        Xte_chars=url_chars(te_df.URL.tolist())
        # Build external character arrays once
        Xext_chars={ds:url_chars(d.URL.tolist()) for ds,d in external}
        neural_models={}
        for name in neural_names:
            try:
                log(f"[SEEN] Training {name} on {len(yn):,} URL strings")
                tf.keras.backend.clear_session()
                net=make_tf_model(name)
                t0=time.perf_counter()
                net.fit(Xn,yn,epochs=4,batch_size=512,validation_split=.05,verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=1,restore_best_weights=True)])
                ts=time.perf_counter()-t0
                t0=time.perf_counter(); sc=net.predict(Xte_chars,batch_size=1024,verbose=0).reshape(-1); inf=time.perf_counter()-t0
                seen_rows.append(score_metrics(name,"SeenTest",yte,(sc>=.5).astype(int),sc,ts,inf))
                seen_scores[name] = sc
                # Fit full internal data (capped for CPU) for external tests
                n_full=min(len(internal),MAX_NEURAL_TRAIN)
                if n_full < len(internal):
                    nn_full,_=train_test_split(internal,train_size=n_full,random_state=SEED,stratify=internal.target)
                else: nn_full=internal
                Xnf=url_chars(nn_full.URL.tolist()); ynf=nn_full.target.to_numpy(dtype=np.int32)
                tf.keras.backend.clear_session()
                full_net=make_tf_model(name)
                t0=time.perf_counter()
                full_net.fit(Xnf,ynf,epochs=4,batch_size=512,validation_split=.05,verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=1,restore_best_weights=True)])
                full_ts=time.perf_counter()-t0
                for ds,d in external:
                    t0=time.perf_counter(); sc2=full_net.predict(Xext_chars[ds],batch_size=1024,verbose=0).reshape(-1); inf=time.perf_counter()-t0
                    unseen_rows.append(score_metrics(name,ds,d.target.to_numpy(dtype=int),(sc2>=.5).astype(int),sc2,full_ts,inf))
                    unseen_scores[ds][name] = sc2
                model_path=OUT/(name.lower().replace(" ","_").replace("&","and").replace("-","_")+".keras")
                full_net.save(model_path)
                status_rows.append({"model":name,"family":"character/neural","seen_status":"OK","unseen_status":"OK","notes":f"Neural models trained on up to {MAX_NEURAL_TRAIN:,} URLs for CPU runtime; not all internal rows."})
                del net, full_net
            except Exception as e:
                log(f"Neural model {name} failed: {e}")
                status_rows.append({"model":name,"family":"character/neural","seen_status":"ERROR","unseen_status":"ERROR","notes":str(e)})
    except Exception as e:
        log(f"TensorFlow neural models unavailable: {e}")
        for name in neural_names:
            status_rows.append({"model":name,"family":"character/neural","seen_status":"NOT RUN","unseen_status":"NOT RUN","notes":str(e)})

    # Probability-average ensembles for each dataset, using outputs from the same split.
    ensemble_specs = [
        ("RF + XGBoost", ["Random Forest", "XGBoost"]),
        ("RF + CNN", ["Random Forest", "Character CNN"]),
        ("CNN + XGBoost", ["Character CNN", "XGBoost"]),
        ("RF + CNN + XGBoost", ["Random Forest", "Character CNN", "XGBoost"]),
    ]
    for ensemble_name, members in ensemble_specs:
        if all(m in seen_scores for m in members):
            scores = np.mean([seen_scores[m] for m in members], axis=0)
            seen_rows.append(score_metrics(ensemble_name, "SeenTest", yte, (scores >= .5).astype(int), scores))
            seen_status = "OK"
        else:
            seen_status = "NOT RUN"
        ext_statuses = []
        for ds, d in external:
            if all(m in unseen_scores.get(ds, {}) for m in members):
                scores = np.mean([unseen_scores[ds][m] for m in members], axis=0)
                unseen_rows.append(score_metrics(ensemble_name, ds, d.target.to_numpy(dtype=int), (scores >= .5).astype(int), scores))
                ext_statuses.append("OK")
            else:
                ext_statuses.append("NOT RUN")
        status_rows.append({
            "model": ensemble_name, "family": "ensemble",
            "seen_status": seen_status,
            "unseen_status": "OK" if ext_statuses and all(x == "OK" for x in ext_statuses) else "PARTIAL/NOT RUN",
            "notes": "Arithmetic mean of member phishing scores; validate probability calibration on external data."
        })

    # Explicitly include requested research models and explain why unavailable.
    for name, note in [
        ("GNN","NOT RUN: no graph representation (nodes/edges) is supplied by the URL-only dataset."),
        ("Vision Transformer","NOT RUN: no webpage screenshots/image tensors are supplied."),
        ("Multimodal Fusion","NOT RUN: no aligned URL + HTML/DOM + screenshot feature dataset is supplied."),
    ]:
        status_rows.append({"model":name,"family":"research/advanced","seen_status":"NOT RUN","unseen_status":"NOT RUN","notes":note})

    seen=pd.DataFrame(seen_rows); unseen=pd.DataFrame(unseen_rows); statuses=pd.DataFrame(status_rows)
    seen.to_csv(OUT/"seen_test_metrics.csv",index=False)
    unseen.to_csv(OUT/"unseen_test_metrics.csv",index=False)
    statuses.to_csv(OUT/"model_status.csv",index=False)
    # Add prediction-free consolidated workbook.
    try:
        with pd.ExcelWriter(OUT/"phishshield_all_models.xlsx",engine="openpyxl") as w:
            seen.to_excel(w,sheet_name="Seen Test",index=False)
            unseen.to_excel(w,sheet_name="Unseen Test",index=False)
            statuses.to_excel(w,sheet_name="Model Coverage",index=False)
            pd.DataFrame([{
                "internal_training_rows":len(internal),
                "seen_train_rows":len(tr_df),"seen_test_rows":len(te_df),
                "external_balanced_rows":len(ext1) if ext1 is not None else 0,
                "external_legitphish_rows":len(ext2) if ext2 is not None else 0,
                "external_label_maps":json.dumps({"BalancedExternal":BALANCED_EXTERNAL_LABEL_MAP,"LegitPhish":LEGITPHISH_LABEL_MAP}),
                "note":"Verify source label mappings. GNN/ViT/multimodal marked NOT RUN because required inputs are absent.",
            }]).to_excel(w,sheet_name="Run Info",index=False)
    except Exception as e: log(f"Workbook failed: {e}")
    log("COMPLETE")
    log(f"Seen metrics: {OUT/'seen_test_metrics.csv'}")
    log(f"Unseen metrics: {OUT/'unseen_test_metrics.csv'}")
    log(f"Model coverage: {OUT/'model_status.csv'}")
    log(f"Workbook: {OUT/'phishshield_all_models.xlsx'}")


if __name__=="__main__":
    main()
