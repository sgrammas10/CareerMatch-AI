import sqlite3
from flask import Flask, request, jsonify, render_template, url_for, session, redirect, send_from_directory, send_file
from flask_cors import CORS
import os
#from resume_scraper import process_resumes
from auth import auth, init_db
from werkzeug.utils import secure_filename



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
    #print("SESSION:", dict(session))  # check for session data
    if "email" not in session:
        # User is not logged in
        return {"message": "Unauthorized. Please log in first.", "skipped": True}
    user_email = session["email"]
    form = request.form
    fname = form.get("fname")
    lname = form.get("lname")
    birthdate = form.get("birthdate")
    resume_file = request.files.get("resume")

    if not resume_file or resume_file.filename == "":
        return jsonify({"error": "No resume uploaded"}), 400

    if personal_info_already_exists(user_email):
        return {"message": "Personal information already submitted for this email.", "skipped": True}

    if not all([user_email, fname, lname, birthdate, resume_file]):
        return jsonify({"error": "Missing required fields"}), 400
    
    # Save or process the info + resume
    # Save resume to disk
    os.makedirs("resumes", exist_ok=True)
    filename = secure_filename(f"{user_email.replace('@', '_')}_resume.pdf")
    resume_path = os.path.join("resumes", filename)
    resume_file.save(resume_path)

    # write user info to a CSV/db
    with open("BackEnd/personal_info.csv", "a", newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([fname, lname, birthdate, user_email])

    # Save path to DB
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        UPDATE users
        SET birthday = ?, resume_path = ?, statement = ?
        WHERE email = ?
    """, (birthdate, resume_path, f"{fname} {lname}", user_email))
    conn.commit()
    conn.close()

    return jsonify({"message": "Resume uploaded successfully."})


@app.route("/profile", methods=["GET"])
def profile():
    if "email" not in session:
        return jsonify({"error": "Not logged in"}), 401

    email = session["email"]
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT birthday, resume_path, statement FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()

    if row:
        return jsonify({
            "email": email,
            "birthday": row[0],
            "resume_url": row[1],
            "statement": row[2],
        })
    else:
        return jsonify({"error": "User not found"}), 404

@app.route("/update_statement", methods=["POST"])
def update_statement():
    if "email" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    statement = data.get("statement", "").strip()

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("UPDATE users SET statement=? WHERE email=?", (statement, session["email"]))
    conn.commit()
    conn.close()

    return jsonify({"message": "Statement updated successfully."})

@app.route("/profile.html")
def protected_profile_page():
    if "email" not in session:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "FrontEnd"))
        return send_from_directory(base_path, "login.html")
    
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "FrontEnd-Protected"))
    return send_from_directory(base_path, "profile.html")




@app.route("/resume", methods=["GET"])
def get_resume():
    if "email" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    email = session["email"]
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT resume_path FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()

    if row and row[0] and os.path.exists(row[0]):
        return send_file(row[0], mimetype="application/pdf")
    else:
        return jsonify({"error": "Resume file not found."}), 404

    

@app.route("/resume_text", methods=["GET"])
def resume_text():
    if "email" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    email = session["email"]
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT resume_text FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()

    if row and row[0]:
        return jsonify({"text": row[0]})
    else:
        return jsonify({"error": "No resume found"}), 404



if __name__ == "__main__":
    init_db()
    app.run(ssl_context=("cert.pem", "key.pem"), port=5000, debug=True)