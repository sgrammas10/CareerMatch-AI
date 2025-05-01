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

# Upload route
@app.route("/upload", methods=["POST"])
def upload_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400

    resume = request.files["resume"]
    if resume.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        results = process_resumes(resume)
        return jsonify({"message": "Resume processed successfully", "results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_db()
    #app.run(port=8080, debug=True)
    app.run()