from flask import Flask, render_template, request, redirect, session
import sqlite3
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------

app = Flask(__name__)
app.secret_key = "secret123"

# -------------------------------------------------
# DATABASE (RENDER SAFE)
# -------------------------------------------------

DB_PATH = os.path.join("/tmp", "users.db")

def get_db():
    return sqlite3.connect(DB_PATH)

def init_db():
    con = get_db()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    con.commit()
    con.close()

init_db()

# -------------------------------------------------
# LOAD & PREPROCESS DATA (CRITICAL FIX)
# -------------------------------------------------

data = pd.read_csv("gym recommendation.csv")

# Encode categorical columns
label_enc = LabelEncoder()
for col in ['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type']:
    data[col] = label_enc.fit_transform(data[col])

# Scale numerical columns
scaler = StandardScaler()
data[['Age', 'Height', 'Weight', 'BMI']] = scaler.fit_transform(
    data[['Age', 'Height', 'Weight', 'BMI']]
)

FEATURES = [
    'Sex', 'Age', 'Height', 'Weight',
    'Hypertension', 'Diabetes', 'BMI',
    'Level', 'Fitness Goal', 'Fitness Type'
]

# -------------------------------------------------
# ROUTES
# -------------------------------------------------

@app.route("/")
def home():
    return redirect("/login")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        con = get_db()
        cur = con.cursor()
        cur.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )
        user = cur.fetchone()
        con.close()

        if user:
            session["user"] = username
            return redirect("/predict")

    return render_template("login.html")

# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        con = get_db()
        cur = con.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password)
            )
            con.commit()
        except:
            pass
        con.close()

        return redirect("/login")

    return render_template("signup.html")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":
        user_input = {
            'Sex': int(request.form["sex"]),
            'Age': float(request.form["age"]),
            'Height': float(request.form["height"]),
            'Weight': float(request.form["weight"]),
            'Hypertension': int(request.form["bp"]),
            'Diabetes': int(request.form["diabetes"]),
            'BMI': float(request.form["bmi"]),
            'Level': int(request.form["level"]),
            'Fitness Goal': int(request.form["goal"]),
            'Fitness Type': int(request.form["type"])
        }

        df = pd.DataFrame([user_input])

        # Scale numeric values
        df[['Age', 'Height', 'Weight', 'BMI']] = scaler.transform(
            df[['Age', 'Height', 'Weight', 'BMI']]
        )

        # Similarity calculation
        similarity = cosine_similarity(data[FEATURES], df).flatten()
        best = data.iloc[similarity.argmax()]

        return render_template(
            "result.html",
            workout=best["Exercises"],
            equipment=best["Equipment"],
            diet=best["Diet"]
        )

    return render_template("index.html")

# -------------------------------------------------
# RUN APP
# -------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
