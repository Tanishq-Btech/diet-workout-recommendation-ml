from flask import Flask, render_template, request, redirect, session
import sqlite3
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- LOAD MODEL & DATA ----------------

with open("model.pkl", "rb") as f:
    scaler = pickle.load(f)
    label_enc = pickle.load(f)

data = pd.read_csv("gym recommendation.csv")

features = [
    'Sex', 'Age', 'Height', 'Weight',
    'Hypertension', 'Diabetes', 'BMI',
    'Level', 'Fitness Goal', 'Fitness Type'
]

# ---------------- DATABASE ----------------

def get_db():
    return sqlite3.connect("users.db")

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

# ---------------- ROUTES ----------------

# ROOT FIX (IMPORTANT)
@app.route("/")
def home():
    return redirect("/login")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["username"]
        pwd = request.form["password"]

        con = get_db()
        cur = con.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (user, pwd))
        result = cur.fetchone()
        con.close()

        if result:
            session["user"] = user
            return redirect("/predict")

    return render_template("login.html")

# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        user = request.form["username"]
        pwd = request.form["password"]

        con = get_db()
        cur = con.cursor()
        try:
            cur.execute("INSERT INTO users VALUES (?,?)", (user, pwd))
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

# ---------------- PREDICT ----------------
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
        df[['Age', 'Height', 'Weight', 'BMI']] = scaler.transform(
            df[['Age', 'Height', 'Weight', 'BMI']]
        )

        similarity = cosine_similarity(data[features], df).flatten()
        best = data.iloc[similarity.argmax()]

        return render_template(
            "result.html",
            workout=best["Exercises"],
            equipment=best["Equipment"],
            diet=best["Diet"]
        )

    return render_template("index.html")

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
