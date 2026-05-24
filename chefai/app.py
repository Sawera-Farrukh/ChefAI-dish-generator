from flask import Flask, render_template, request, redirect, session, url_for
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from utils.recipe_generator import find_by_name, find_by_ingredients, clean_steps

app = Flask(__name__)
app.secret_key = "chefai-secret-key"

DB_PATH = "users.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, generate_password_hash(password))
            )
            db.commit()
            db.close()
            return redirect(url_for("login"))
        except:
            db.close()
            return render_template("signup.html", error="Username already exists!")

    return render_template("signup.html")


# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        db.close()

        if user and check_password_hash(user["password"], password):
            session["user"] = username
            return redirect(url_for("home"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ---------------- GENERATE RECIPE ----------------
@app.route("/generate", methods=["POST"])
def generate():
    dish_name = request.form.get("dish_name", "").strip()
    ingredients = request.form.get("ingredients", "").strip()

    if dish_name:
        recipe = find_by_name(dish_name)
        if recipe:
            recipe["ingredients"] = [i.strip() for i in recipe["ingredients"].split(",")]
            recipe["directions"] = clean_steps(recipe["directions"])
            return render_template("recipe_result.html", recipe=recipe)

    if ingredients:
        results = find_by_ingredients(ingredients)
        return render_template("ingredient_results.html", results=results, query=ingredients)

    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
