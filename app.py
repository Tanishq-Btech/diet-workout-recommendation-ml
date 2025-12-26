from flask import Flask, render_template, request, redirect, session
import pandas as pd
import pickle
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "diet_workout_secret"

# ---------------- LOAD DATA ----------------
data = pd.read_csv("gym recommendation.csv")

# ✅ ENCODE DATASET (FIXES 'Male' ERROR)
label_cols = ['Sex', 'Hypertension', 'Diabetes', 'Level',
              'Fitness Goal', 'Fitness Type']

le = LabelEncoder()
for col in label_cols:
    data[col] = le.fit_transform(data[col])

# Load scaler & encoder from model.pkl
with open("model.pkl", "rb") as f:
    scaler = pickle.load(f)
    label_enc = pickle.load(f)

features = [
    'Sex','Age','Height','Weight',
    'Hypertension','Diabetes','BMI',
    'Level','Fitness Goal','Fitness Type'
]

# ---------------- DATABASE ----------------
def get_db():
    return sqlite3.connect("users.db")

def init_db():
    db = get_db()
    db.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
    db.execute("""
        CREATE TABLE IF NOT EXISTS history (
            username TEXT,
            exercises TEXT,
            diet TEXT,
            equipment TEXT
        )
    """)
    db.commit()
    db.close()

init_db()

# ---------------- LOGIN ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        user = cur.fetchone()
        db.close()

        if user:
            session["user"] = u
            return redirect("/predict")

    return render_template("login.html")

# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        db = get_db()
        db.execute(
            "INSERT INTO users VALUES (?, ?)",
            (request.form["username"], request.form["password"])
        )
        db.commit()
        db.close()
        return redirect("/")
    return render_template("signup.html")

# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect("/")

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

        # Scale numeric features
        df[['Age','Height','Weight','BMI']] = scaler.transform(
            df[['Age','Height','Weight','BMI']]
        )

        # Compute similarity
        similarity = cosine_similarity(data[features], df).flatten()
        best_match = data.iloc[similarity.argmax()]

        # 🔥 DEBUG PRINT (VERY IMPORTANT)
        print("Workout:", best_match["Exercises"])
        print("Equipment:", best_match["Equipment"])
        print("Diet:", best_match["Diet"])

        return render_template(
            "result.html",
            workout=best_match["Exercises"],
            equipment=best_match["Equipment"],
            diet=best_match["Diet"]
        )

    return render_template("index.html")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT exercises, diet, equipment FROM history WHERE username=?",
        (session["user"],)
    )
    records = cur.fetchall()
    db.close()

    return render_template("dashboard.html", records=records)

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

