Python 3.13.2 (v3.13.2:4f8bb3947cf, Feb  4 2025, 11:51:10) [Clang 15.0.0 (clang-1500.3.9.4)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
#!/usr/bin/env python
# pip install pandas scikit-learn spacy==3.7.3
# python -m spacy download tr_core_news_sm
import pandas as pd, spacy, re, string
... from sklearn.model_selection import train_test_split
... from sklearn.feature_extraction.text import TfidfVectorizer
... from sklearn.linear_model import LogisticRegression
... from sklearn.metrics import classification_report, f1_score
... import joblib
... 
... DATA_PATH = "tweets.csv"  # columns: text,label (0/1)
... nlp = spacy.load("tr_core_news_sm")
... tbl = str.maketrans("", "", string.punctuation)
... 
... def clean(text):
...     doc = nlp(text.lower())
...     tokens = [t.lemma_.translate(tbl) for t in doc if not t.is_stop and t.is_alpha]
...     return " ".join(tokens)
... 
... def main():
...     df = pd.read_csv(DATA_PATH).dropna()
...     df["clean"] = df["text"].apply(clean)
...     X_train, X_test, y_train, y_test = train_test_split(
...         df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
...     )
... 
...     tfidf = TfidfVectorizer(min_df=3, ngram_range=(1,2))
...     Xtr = tfidf.fit_transform(X_train)
...     Xte = tfidf.transform(X_test)
... 
...     clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y_train)
...     y_pred = clf.predict(Xte)
...     print(classification_report(y_test, y_pred))
...     print("macro-F1:", round(f1_score(y_test, y_pred, average="macro"), 3))
... 
...     joblib.dump({"vec": tfidf, "model": clf}, "sentiment_pipeline.joblib")
...     print("Pipeline saved → sentiment_pipeline.joblib")
... 
... if __name__ == "__main__":
...     main()
