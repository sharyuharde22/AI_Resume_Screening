import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skills import SKILLS

def read_file(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().lower()
    elif path.endswith(".pdf"):
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text().lower()
        return text
    return ""

def extract_skills(text):
    found = []
    for skill in SKILLS:
        if skill in text:
            found.append(skill)
    return set(found)

def rank_resumes(resume_folder, jd_path):
    job_desc_text = read_file(jd_path)
    jd_skills = extract_skills(job_desc_text)

    resumes_data = []

    resume_texts = []
    resume_names = []

    for file in os.listdir(resume_folder):
        file_path = os.path.join(resume_folder, file)
        text = read_file(file_path)
        if text.strip():
            resume_texts.append(text)
            resume_names.append(file)

            resume_skills = extract_skills(text)
            matched = resume_skills.intersection(jd_skills)
            missing = jd_skills - resume_skills

            resumes_data.append({
                "name": file,
                "resume_skills": resume_skills,
                "matched_skills": matched,
                "missing_skills": missing
            })

    documents = [job_desc_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(documents)

    similarity_scores = cosine_similarity(tfidf[0:1], tfidf[1:])[0]

    results = []
    for i, score in enumerate(similarity_scores):
        skill_match_percent = (
            len(resumes_data[i]["matched_skills"]) / len(jd_skills) * 100
        ) if jd_skills else 0

        results.append({
            "name": resume_names[i],
            "similarity": round(score * 100, 2),
            "skill_match": round(skill_match_percent, 2),
            "matched_skills": resumes_data[i]["matched_skills"],
            "missing_skills": resumes_data[i]["missing_skills"]
        })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)