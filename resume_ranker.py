import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

def read_file(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif path.endswith(".pdf"):
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text
    return ""

def rank_resumes(resume_folder, jd_path):
    resumes = []
    resume_names = []

    for file in os.listdir(resume_folder):
        file_path = os.path.join(resume_folder, file)
        text = read_file(file_path)
        if text.strip():
            resumes.append(text)
            resume_names.append(file)

    job_desc = read_file(jd_path)

    documents = [job_desc] + resumes

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(documents)

    similarity_scores = cosine_similarity(tfidf[0:1], tfidf[1:])[0]

    ranked = sorted(
        zip(resume_names, similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked