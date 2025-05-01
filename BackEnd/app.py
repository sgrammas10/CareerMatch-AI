from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
import os
from resume_scraper import process_resumes
from auth import auth, init_db

app = Flask(__name__)
app.register_blueprint(auth)
CORS(app)


# Home route
@app.route('/')
def home():
    return render_template('index.html')

# API route for testing CORS
@app.route('/api/data')
def get_data():
    return jsonify({"message": "CORS is working!"})

# Route for personal_info.html
@app.route('/personal_info')
def personal_info():
    return render_template('personal_info.html')

@app.route("/upload", methods=["POST"])
def upload_resume():
    form = request.form
    resume = request.files.get("resume")

    if not resume or resume.filename == "":
        return jsonify({"error": "No resume uploaded"}), 400

    email = form.get("email")
    fname = form.get("fname")
    lname = form.get("lname")
    birthdate = form.get("birthdate")

    if not all([email, fname, lname, birthdate]):
        return jsonify({"error": "Missing required fields"}), 400

    # Save or process the info + resume
    result = process_resumes(email, resume)

    # Optional: write user info to a CSV/db
    with open("BackEnd/personal_info.csv", "a", newline='', encoding='utf-8') as f:
        import csv
        csv.writer(f).writerow([fname, lname, birthdate, email])

    return jsonify(result)


if __name__ == "__main__":
    init_db()
    #app.run(port=8080, debug=True)
    app.run()