import os
from flask import Flask, render_template, request
from resume_ranker import rank_resumes

app = Flask(__name__)

UPLOAD_RESUMES = "uploads/resumes"
UPLOAD_JD = "uploads/jd"

os.makedirs(UPLOAD_RESUMES, exist_ok=True)
os.makedirs(UPLOAD_JD, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    results = None

    if request.method == "POST":
        # Save Job Description
        jd_file = request.files["job_description"]
        jd_path = os.path.join(UPLOAD_JD, jd_file.filename)
        jd_file.save(jd_path)

        # Save Resumes
        resume_files = request.files.getlist("resumes")
        for r in resume_files:
            r.save(os.path.join(UPLOAD_RESUMES, r.filename))

        # Rank resumes
        results = rank_resumes(UPLOAD_RESUMES, jd_path)

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)