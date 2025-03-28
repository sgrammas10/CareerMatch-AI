#flask backend here
from flask import Flask, request, jsonify, render_template
###these three need to be implemented when the functions are finished and added into the folder
#from scraper import scrape_jobs
from summarization import summarize_jobs
#from recommender import recommend_jobs
import sqlite3

from auth import auth

app = Flask(__name__)
app.register_blueprint(auth)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/scrape", methods=["POST"])
def scrape():
    companies = request.json["companies"]
    jobs = scrape_jobs(companies)
    return jsonify({"jobs": jobs})

@app.route("/summarize", methods=["POST"])
def summarize():
    jobs = request.json["jobs"]
    summaries = summarize_jobs(jobs)
    return jsonify({"summaries": summaries})

@app.route("/recommend", methods=["POST"])
def recommend():
    user_profile = request.json["profile"]
    recommendations = recommend_jobs(user_profile)
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True)
