from flask import Flask, request, jsonify
import os
from resume_scraper import scrape_resume

def call_resume_scraper(filepath):
    return scrape_resume(filepath)


app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return jsonify({'message': f'Resume {file.filename} uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Invalid file type. Only PDF allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
