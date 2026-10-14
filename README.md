# Disaster Tweet Classification

An NLP project that classifies tweets as being about a **real disaster** (1) or **not** (0), with a Flask web app for live predictions.

## Results (held-out test set, 1,501 tweets)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.818 | 0.819 | 0.736 | **0.775** | 0.868 |
| Linear SVM | 0.811 | 0.810 | 0.728 | 0.767 | 0.866 |
| Naive Bayes | 0.819 | 0.849 | 0.702 | 0.768 | 0.858 |
| Random Forest | 0.799 | 0.876 | 0.616 | 0.723 | 0.853 |
| Neural Net (MLP on LSA) | 0.771 | 0.749 | 0.695 | 0.721 | 0.834 |

The deployed model is a raw-text pipeline (word + character TF-IDF, then logistic regression). It scores 0.808 accuracy and 0.767 F1 on the test set, and 0.735 F1 on a noisy version of the test set.

Known limits: the model overfits (train F1 0.93 vs CV F1 0.76), and figurative uses such as "on fire" or "bomb" are sometimes still over-scored.

## Approach

1. **Exploration and preparation:** class balance, top words, duplicate removal, text cleaning (URLs, mentions, punctuation), tokenising, stratified 80/20 split.
2. **Features:** word 1-2 gram TF-IDF, character 2-5 gram TF-IDF, tweet statistics (length, hashtags, mentions, URL, caps ratio, a lexicon sentiment score), and LSA vectors. Word2Vec/GloVe were not used because they need extra downloads.
3. **Models:** five candidates tuned with 5-fold grid search on F1.
4. **Evaluation and validation:** metrics, ROC and PR curves, confusion matrices, learning curve, noise-robustness test, metaphor probes.
5. **Deployment:** pickled pipeline served by Flask.

## Repository layout

```
data/tweets.csv          dataset (7,613 labelled tweets)
pipeline.py              full analysis, training and evaluation
textutil.py              text-cleaning helpers (shared with the app)
outputs/                 charts, results.json, prepared_data.csv
app/app.py               Flask backend (GET /, POST /predict)
app/templates/index.html web UI
app/model.pkl            trained model
make_ppt.py              builds the presentation
Disaster_Tweet_Classification.pptx   project presentation
1789547632.pdf           original project brief
Procfile                 for Heroku-style deployment
```

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib scipy joblib flask python-pptx

python pipeline.py        # retrain, regenerate outputs/ and app/model.pkl
python app/app.py         # web app at http://localhost:5000
python make_ppt.py        # rebuild the deck
```

Python 3.10 was used. `python pipeline.py` must be run from the project root.

API example:

```bash
curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
     -d '{"text": "Forest fire near La Ronge Sask. Canada"}'
# {"disaster": true, "label": "Real disaster", "probability": 0.9882}
```

## Deployment

The app is not deployed to a live host. The `Procfile` and `app/requirements.txt` are set up for Heroku or Render (`gunicorn --chdir app app:app`).

## Next steps

Fine-tune a transformer such as DistilBERT, use pretrained GloVe/Word2Vec features, tune the decision threshold for the acceptable false-alarm rate, and add location and time signals.
