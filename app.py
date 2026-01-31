import os
import re
import urllib.request
from flask import *
import sqlite3
from werkzeug.utils import secure_filename
from nltk.corpus import stopwords

import re
import nltk
nltk.download('stopwords')
import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


app = Flask(__name__)

app.secret_key = "secret key"


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/logon')
def logon():
    return render_template('signup.html')


@app.route("/signup", methods=["post"])
def signup():
    username = request.form['user']
    name = request.form['name']
    email = request.form['email']
    number = request.form["mobile"]
    password = request.form['password']
    role = "student"
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`,'role') VALUES (?, ?, ?, ?, ?,?)",
                (username, email, password, number, name, role))
    con.commit()
    con.close()
    return render_template("index.html")


@app.route("/signin", methods=["post"])
def signin():
    mail1 = request.form['user']
    password1 = request.form['password']
    con = sqlite3.connect('signup.db')
    data = 0
    data = con.execute(
        "select `user`, `password`,role from info where `user` = ? AND `password` = ?", (mail1, password1,)).fetchall()
    print(data)
    if mail1 == 'admin' and password1 == 'admin':
        session['username'] = "admin"
        return redirect("userlogin")
    elif mail1 == str(data[0][0]) and password1 == str(data[0][1]):
        print(data)
        session['username'] = data[0][0]
        return redirect("userlogin")
    else:
        return render_template("signup.html")


@app.route("/userlogin")
def userlogin():
    return render_template("student.html")
#################################


def process_text(s):
    # Check string to see if they are a punctuation
    nopunc = [char for char in s if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # Convert string to lowercase and remove stopwords
    clean_string = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    clean_string = ' '.join(clean_string)
    return clean_string
#################################

cv=joblib.load('count_vectorizer.pkl')
bnb = joblib.load('bernoulli_naive_bayes_model.pkl')

def prediction_input_processing(text):
    result = bnb.predict(cv.transform([process_text(text)]).toarray())[0]
    if result == 1:
        return 'Yes this News is fake'
    return 'No, It is not fake'


@app.route('/predict', methods=["POST"])
def predict():
    y = request.form["t"]
    t = prediction_input_processing(y)
    return render_template("student.html", t=t)


####################
@app.route('/logout')
def home():
    session.pop('username', None)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
