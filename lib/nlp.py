import re
import nltk
ps = nltk.stem.PorterStemmer()

def split(src):
    tks = [i.strip() for i in re.split(r'[^A-Za-z0-9_]', src) if i.strip()]
    return tks

def processed(text,stemming=False):
    # print(text)
    text = re.sub("[ ]{1,}",r' ',text)
    text = re.sub(r'\W+|\d+', ' ', text.strip().lower())
    tokens = [token.strip()  for token in text.split(" ")]
    tokens = [token for token in tokens if len(token) > 0]
    if stemming:
        tokens = [ps.stem(i) for i in tokens if i]

    return " ".join(tokens)

def lower(src):
    src = src.lower()
    tks = split(src)
    return ' '.join(tks)
