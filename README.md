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

## Examples: input tweet and model output

The model returns the probability that a tweet is about a real disaster. A probability of 0.5 or higher is labelled **Real disaster**. These are real outputs from the saved model.

**Correctly classified**

| Tweet | P(disaster) | Prediction |
|---|---|---|
| Forest fire near La Ronge Sask. Canada | 0.988 | Real disaster |
| 13,000 people receive #wildfires evacuation orders in California | 0.991 | Real disaster |
| Earthquake of magnitude 6.1 hits the coast, buildings collapsed | 0.979 | Real disaster |
| Massive flood waters rising, people trapped on rooftops | 0.890 | Real disaster |
| Three people died from the heat wave so far | 0.880 | Real disaster |
| Your new song is fire lol | 0.008 | Not a disaster |
| I'm flooded with work this week | 0.268 | Not a disaster |
| What a goooooooaaaaaal!!!!!!! | 0.061 | Not a disaster |
| I love fruits | 0.019 | Not a disaster |

**Hard cases the model gets wrong.** The figurative use of a disaster word still fools it:

| Tweet | P(disaster) | Prediction | Truth |
|---|---|---|---|
| That concert was a total bomb haha | 0.688 | Real disaster | Not a disaster |
| Just ate a burning hot curry, my mouth is on fire | 0.717 | Real disaster | Not a disaster |

A contextual model such as BERT would handle these better (see Next steps).

**Why it decides this way.** The words that push the score most towards "disaster" are *fires, earthquake, hiroshima, bombing, japan, shot, near*. The words that push towards "not a disaster" include *new, jobs, art, finally, wants, cake*. The cleaning step removes URLs, @mentions, `#` signs and punctuation first, so `#wildfires` is read as `wildfires`.

**Web app.** Paste a tweet into the box and click Classify. You get a verdict and a probability bar, for example:

```
Input : Wildfire smoke covers the city, residents told to evacuate
Output: Real disaster, disaster probability 97.0%
```

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

curl -X POST localhost:5000/predict -H "Content-Type: application/json" \n     -d '{"text": "Your new song is fire lol"}'
# {"disaster": false, "label": "Not a disaster", "probability": 0.0077}
```

## Deployment

The app is not deployed to a live host. The `Procfile` and `app/requirements.txt` are set up for Heroku or Render (`gunicorn --chdir app app:app`).

## Next steps

Fine-tune a transformer such as DistilBERT, use pretrained GloVe/Word2Vec features, tune the decision threshold for the acceptable false-alarm rate, and add location and time signals.
