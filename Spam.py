import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df

df = load_data()

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["clean_msg"] = df["message"].apply(clean_text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["clean_msg"])
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.set_page_config(page_title="Spam Detector", page_icon="📨", layout="centered")
st.title("📨 Spam Email Detection App")
st.write("Check if your message is **Spam or Ham** using Machine Learning.")

st.markdown("---")

user_input = st.text_area("✉️ Type or paste your email/message below:")

if st.button("🔍 Check Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please type a message first!")
    else:
        cleaned = clean_text(user_input)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0]

        spam_conf = proba[1] * 100
        ham_conf = proba[0] * 100

        if pred == 1:
            st.error(f"🚫 This message is **SPAM!**")
            st.progress(int(spam_conf))
            st.write(f"🔴 Confidence: **{spam_conf:.2f}%** Spam")
        else:
            st.success(f"✅ This message looks **HAM (Not Spam)**!")
            st.progress(int(ham_conf))
            st.write(f"🟢 Confidence: **{ham_conf:.2f}%** Ham")

st.markdown("---")
st.subheader("📊 Model Info")
st.write(f"Model used: Multinomial Naive Bayes")
st.write(f"Accuracy on test data: **{acc*100:.2f}%**")

st.markdown("---")
st.caption("Created by Akshay Kumar 💻 | Powered by Streamlit + Scikit-learn")
