import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import string

ps = PorterStemmer()


def transform_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    y = []
    for i in message:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    message = y[:]
    y.clear()

    for i in message:
        y.append(ps.stem(i))

    return " ".join(y)

cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):

    transform_sms = transform_message(input_sms)

    vector_input = cv.transform([transform_sms])

    result = model.predict(vector_input)[0]

    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")