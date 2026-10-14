import re


def clean(t):
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = t.replace("#", " ")
    t = re.sub(r"&\w+;", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokenize(t):
    return [w for w in t.split() if len(w) > 1]
