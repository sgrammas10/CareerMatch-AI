from flask import Flask, request, jsonify
import os
from resume_scraper import process_resumes

from auth import auth

app = Flask(__name__)
app.register_blueprint(auth)


@app.route("/upload", methods=["POST"])
def upload_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400

    resume = request.files["resume"]
    if resume.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the resume temporarily
    resume_path = os.path.join("uploads", resume.filename)
    resume.save(resume_path)

    # Process the resume and get results
    try:
        results = process_resumes(resume_path)
        return jsonify({"message": "Resume processed successfully", "results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
