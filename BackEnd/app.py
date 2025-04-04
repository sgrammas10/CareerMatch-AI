from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from resume_scraper import process_resumes

from auth import auth, init_db

app = Flask(__name__)
app.register_blueprint(auth)
CORS(app)  

@app.route('/api/data')
def get_data():
    return jsonify({"message": "CORS is working!"})

#change index.html to whatever html file you want to test
@app.route('/')
def home():
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../FrontEnd')
    return send_from_directory(frontend_path, 'index.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../FrontEnd')
    return send_from_directory(frontend_path, filename)

@app.route('/personal_info')
def personal_info():
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../FrontEnd')
    return send_from_directory(frontend_path, 'personal_info.html')

@app.route("/upload", methods=["POST"])
@app.route("/upload/", methods=["POST"])
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
    app.run(port=8080, debug=True)
