import json, joblib, warnings, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score)
from textutil import clean, tokenize

warnings.filterwarnings("ignore")
OUT = "outputs/"
df = pd.read_csv("data/tweets.csv")
R = {}

# ---------- Part 1: exploration
R["shape"] = list(df.shape)
R["dtypes"] = {k: str(v) for k, v in df.dtypes.items()}
R["nulls"] = df.isnull().sum().to_dict()
R["dups"] = int(df.text.duplicated().sum())
vc = df.target.value_counts().sort_index()
R["class_counts"] = {int(k): int(v) for k, v in vc.items()}
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Non-disaster (0)", "Disaster (1)"], vc.values, color=["#8aa4c8", "#d9534f"])
for i, v in enumerate(vc.values):
    ax.text(i, v + 40, f"{v} ({v/len(df):.0%})", ha="center")
ax.set_title("Class distribution"); ax.set_ylabel("Tweets")
plt.tight_layout(); plt.savefig(OUT + "class_dist.png", dpi=150); plt.close()

conf = df.groupby("text").target.nunique()
R["conflicting_dup_texts"] = int((conf > 1).sum())
df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
R["rows_after_dedup"] = len(df)
df["clean"] = df.text.map(clean)
df["tokens"] = df.clean.map(tokenize)


def topwords(label, n=15):
    c = Counter(w for toks in df[df.target == label].tokens for w in toks if w not in ENGLISH_STOP_WORDS)
    return c.most_common(n)


td, tn = topwords(1), topwords(0)
R["top_disaster"], R["top_non"] = td, tn
fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, tw, col, t in [(axs[0], td, "#d9534f", "Disaster"), (axs[1], tn, "#8aa4c8", "Non-disaster")]:
    ax.barh([w for w, _ in tw][::-1], [c for _, c in tw][::-1], color=col)
    ax.set_title("Top words: " + t)
plt.tight_layout(); plt.savefig(OUT + "top_words.png", dpi=150); plt.close()

df["len"] = df.text.str.len()
fig, ax = plt.subplots(figsize=(5.5, 4))
for l, c, n in [(0, "#8aa4c8", "Non-disaster"), (1, "#d9534f", "Disaster")]:
    ax.hist(df[df.target == l].len, bins=30, alpha=.65, color=c, label=n)
ax.set_title("Tweet length (characters)"); ax.legend()
plt.tight_layout(); plt.savefig(OUT + "length_hist.png", dpi=150); plt.close()

# ---------- extra features
POS = set("love great good happy best amazing beautiful win safe thank nice lol fun awesome hope".split())
NEG = set("dead death killed kill die died fire flood crash attack bomb terror war injured fatal burn destroy disaster emergency panic fear hurt storm collapse".split())


def extra(t, toks):
    return [len(t), len(toks), t.count("#"), t.count("@"), int("http" in t.lower()), t.count("!"), t.count("?"),
            sum(c.isupper() for c in t) / max(len(t), 1),
            sum(w in POS for w in toks) - sum(w in NEG for w in toks)]


X_extra = np.array([extra(t, k) for t, k in zip(df.text, df.tokens)], float)
df["kw"] = df.keyword.fillna("").str.replace("%20", " ")

# ---------- split
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.2, stratify=df.target, random_state=42)
y = df.target.values
ytr, yte = y[tr], y[te]
R["train"], R["test"] = len(tr), len(te)

corpus = (df.clean + " " + df.kw).values
tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=60000)
Xw_tr = tfw.fit_transform(corpus[tr]); Xw_te = tfw.transform(corpus[te])
Xc_tr = tfc.fit_transform(corpus[tr]); Xc_te = tfc.transform(corpus[te])
sc = StandardScaler().fit(X_extra[tr])
E_tr = csr_matrix(sc.transform(X_extra[tr])); E_te = csr_matrix(sc.transform(X_extra[te]))
svd = TruncatedSVD(150, random_state=42).fit(Xw_tr)
S_tr, S_te = svd.transform(Xw_tr), svd.transform(Xw_te)
R["tfidf_vocab"] = len(tfw.vocabulary_)
sets = {
    "TF-IDF words": (Xw_tr, Xw_te),
    "TF-IDF word+char": (hstack([Xw_tr, Xc_tr]).tocsr(), hstack([Xw_te, Xc_te]).tocsr()),
    "TF-IDF word+char+extra": (hstack([Xw_tr, Xc_tr, E_tr]).tocsr(), hstack([Xw_te, Xc_te, E_te]).tocsr()),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
abl = {n: float(cross_val_score(LogisticRegression(C=3, max_iter=2000), a, ytr, cv=cv, scoring="f1").mean()) for n, (a, _) in sets.items()}
R["feature_ablation_f1_cv"] = abl
best_fs = max(abl, key=abl.get)
R["best_feature_set"] = best_fs
Xtr, Xte = sets[best_fs]
R["extras_only_f1_cv"] = float(cross_val_score(LogisticRegression(max_iter=2000), E_tr, ytr, cv=cv, scoring="f1").mean())
R["lsa_only_f1_cv"] = float(cross_val_score(LogisticRegression(max_iter=2000), S_tr, ytr, cv=cv, scoring="f1").mean())

# ---------- models + grid search
grids = {
    "Logistic Regression": (LogisticRegression(max_iter=3000), {"C": [0.3, 1, 3, 10]}, Xtr, Xte),
    "Linear SVM": (LinearSVC(), {"C": [0.05, 0.1, 0.3, 1]}, Xtr, Xte),
    "Naive Bayes": (MultinomialNB(), {"alpha": [0.1, 0.3, 1]}, Xw_tr, Xw_te),
    "Random Forest": (RandomForestClassifier(random_state=42, n_jobs=-1),
                      {"n_estimators": [200, 400], "max_depth": [None, 40]},
                      hstack([Xw_tr, E_tr]).tocsr(), hstack([Xw_te, E_te]).tocsr()),
    "Neural Net (MLP on LSA)": (MLPClassifier(max_iter=400, early_stopping=True, random_state=42),
                                {"hidden_layer_sizes": [(64,), (128, 64)], "alpha": [1e-4, 1e-2]}, S_tr, S_te),
}
res, fitted, scores, preds = {}, {}, {}, {}
for n, (m, g, a, b) in grids.items():
    gs = GridSearchCV(m, g, cv=cv, scoring="f1", n_jobs=-1).fit(a, ytr)
    p = gs.predict(b)
    s = gs.decision_function(b) if n == "Linear SVM" else gs.predict_proba(b)[:, 1]
    res[n] = dict(best_params={k: str(v) for k, v in gs.best_params_.items()}, cv_f1=float(gs.best_score_),
                  train_f1=float(f1_score(ytr, gs.predict(a))),
                  accuracy=float(accuracy_score(yte, p)), precision=float(precision_score(yte, p)),
                  recall=float(recall_score(yte, p)), f1=float(f1_score(yte, p)),
                  roc_auc=float(auc(*roc_curve(yte, s)[:2])), pr_auc=float(average_precision_score(yte, s)))
    fitted[n], scores[n], preds[n] = gs.best_estimator_, s, p
    print(n, res[n], flush=True)
R["models"] = res
best = max(res, key=lambda k: res[k]["f1"])
R["best_model"] = best

# ---------- evaluation plots
cols = ["#1f77b4", "#d9534f", "#2ca02c", "#9467bd", "#ff7f0e"]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for (n, s), c in zip(scores.items(), cols):
    f, t, _ = roc_curve(yte, s); axs[0].plot(f, t, c=c, label=f"{n} ({res[n]['roc_auc']:.3f})")
    p_, r_, _ = precision_recall_curve(yte, s); axs[1].plot(r_, p_, c=c, label=f"{n} ({res[n]['pr_auc']:.3f})")
axs[0].plot([0, 1], [0, 1], "k--", lw=.8)
axs[0].set(title="ROC curves (AUC)", xlabel="False positive rate", ylabel="True positive rate")
axs[1].set(title="Precision-Recall curves (AP)", xlabel="Recall", ylabel="Precision")
axs[0].legend(fontsize=7); axs[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT + "roc_pr.png", dpi=150); plt.close()

fig, axs = plt.subplots(1, 5, figsize=(17, 3.6))
for ax, n in zip(axs, preds):
    ConfusionMatrixDisplay(confusion_matrix(yte, preds[n]), display_labels=["Non", "Disaster"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(n, fontsize=9)
plt.tight_layout(); plt.savefig(OUT + "confusion_all.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(yte, preds[best]), display_labels=["Non", "Disaster"]).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Best: " + best)
plt.tight_layout(); plt.savefig(OUT + "confusion_best.png", dpi=150); plt.close()
R["best_confusion"] = confusion_matrix(yte, preds[best]).tolist()

names = list(res)
fig, ax = plt.subplots(figsize=(9, 4)); w = .2
for i, mt in enumerate(["accuracy", "precision", "recall", "f1"]):
    ax.bar(np.arange(len(names)) + i * w, [res[n][mt] for n in names], w, label=mt)
ax.set_xticks(np.arange(len(names)) + 1.5 * w)
ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
ax.set_ylim(.5, 1); ax.legend(ncol=4); ax.set_title("Test-set metrics by model")
plt.tight_layout(); plt.savefig(OUT + "model_compare.png", dpi=150); plt.close()

# ---------- validation: learning curve for a linear model on the chosen features
lin = LogisticRegression(max_iter=3000, C=float(res["Logistic Regression"]["best_params"]["C"]))
ts, trs, vas = learning_curve(lin, Xtr, ytr, cv=cv, scoring="f1", train_sizes=np.linspace(.2, 1, 5), n_jobs=-1)
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(ts, trs.mean(1), "o-", label="Train F1"); ax.plot(ts, vas.mean(1), "o-", label="CV F1")
ax.set(title="Learning curve: Logistic Regression", xlabel="Training samples", ylabel="F1"); ax.legend()
plt.tight_layout(); plt.savefig(OUT + "learning_curve.png", dpi=150); plt.close()
R["learning_curve"] = dict(train=float(trs.mean(1)[-1]), cv=float(vas.mean(1)[-1]))
R["overfit_gap_best"] = res[best]["train_f1"] - res[best]["f1"]

# ---------- final raw-text pipeline for deployment
C_best = float(res["Logistic Regression"]["best_params"]["C"])
final = Pipeline([
    ("tf", FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english", preprocessor=clean)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=60000, preprocessor=clean))])),
    ("clf", LogisticRegression(max_iter=3000, C=C_best))])
raw = df.text.values
final.fit(raw[tr], ytr)
fp = final.predict(raw[te])
R["deploy_model_test"] = dict(accuracy=float(accuracy_score(yte, fp)), precision=float(precision_score(yte, fp)),
                              recall=float(recall_score(yte, fp)), f1=float(f1_score(yte, fp)))
rng = np.random.default_rng(0)


def noisy(t):
    t = t.upper() if rng.random() < .3 else t
    t = re.sub(r"(\w)", lambda m: m.group(1) * 2 if rng.random() < .04 else m.group(1), t)
    return t + " http://t.co/xyz #lol @user"


R["robust_f1_noisy"] = float(f1_score(yte, final.predict([noisy(t) for t in raw[te]])))
probes = ["Forest fire near my house, evacuation ordered", "Your new song is fire lol",
          "Earthquake of magnitude 6.1 hits the coast, buildings collapsed", "I'm flooded with work this week",
          "Massive flood waters rising, people trapped on rooftops", "That concert was a total bomb haha",
          "Wildfire smoke covers the city, residents told to evacuate", "Just ate a burning hot curry, my mouth is on fire"]
R["probes"] = [(p, float(final.predict_proba([p])[0, 1])) for p in probes]
wrong = np.where(fp != yte)[0][:8]
R["errors"] = [(raw[te][i], int(yte[i])) for i in wrong]
fn = final.named_steps["tf"].get_feature_names_out(); co = final.named_steps["clf"].coef_[0]
wo = sorted([(f[3:], float(c)) for f, c in zip(fn, co) if f.startswith("w__")], key=lambda x: x[1])
R["top_coef_pos"], R["top_coef_neg"] = wo[-12:][::-1], wo[:12]
joblib.dump(final, "app/model.pkl")
json.dump(R, open(OUT + "results.json", "w"), indent=1, default=str)
df[["id", "text", "clean", "target"]].to_csv(OUT + "prepared_data.csv", index=False)
print(json.dumps({k: R[k] for k in ["best_model", "best_feature_set", "feature_ablation_f1_cv", "deploy_model_test",
                                    "robust_f1_noisy", "probes", "extras_only_f1_cv", "lsa_only_f1_cv"]}, indent=1))
