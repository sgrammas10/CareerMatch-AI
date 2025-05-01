from flask import Flask, request, jsonify, render_template, url_for, session
from flask_cors import CORS
import os
from resume_scraper import process_resumes
from auth import auth, init_db


#might need to run openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
# this will create a self-signed certificate for local testing, already in the gitignore 

app = Flask(__name__)
#app.secret_key = os.urandom(24)  # Required for sessions
app.secret_key = "your-super-secret-dev-key"


app.config.update({
    "SESSION_COOKIE_SAMESITE": "None",
    "SESSION_COOKIE_SECURE": True  # Set to True only if using HTTPS
})

app.register_blueprint(auth)
CORS(app,
     supports_credentials=True,
     resources={r"/*": {"origins": 
         "https://127.0.0.1:8080"
     }})

#helper functions:
import csv
def personal_info_already_exists(email, filename="BackEnd/personal_info.csv"):
    if not os.path.exists(filename):
        return False
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4 and row[3].strip().lower() == email.lower():
                return True
    return False


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
    print("SESSION:", dict(session))  # ADD THIS
    if "email" not in session:
        # User is not logged in
        return {"message": "Unauthorized. Please log in first.", "skipped": True}
    user_email = session["email"]
    form = request.form
    resume = request.files.get("resume")

    if not resume or resume.filename == "":
        return jsonify({"error": "No resume uploaded"}), 400

    fname = form.get("fname")
    lname = form.get("lname")
    birthdate = form.get("birthdate")

    if not all([user_email, fname, lname, birthdate]):
        return jsonify({"error": "Missing required fields"}), 400
    
    if personal_info_already_exists(user_email):
        return {"message": "Personal information already submitted for this email.", "skipped": True}


    # Save or process the info + resume
    result = process_resumes(user_email, resume)

    # Optional: write user info to a CSV/db
    with open("BackEnd/personal_info.csv", "a", newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([fname, lname, birthdate, user_email])

    return jsonify(result)


if __name__ == "__main__":
    init_db()
    app.run(ssl_context=("cert.pem", "key.pem"), port=5000, debug=True)