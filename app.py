import joblib
import numpy as np
from flask import Flask, request, render_template
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import string
import PyPDF2


# Load SpaCy model for NER (Named Entity Recognition)
nlp = spacy.load("en_core_web_lg")

# Load pre-trained models
tfidf_vectorizer = joblib.load("C:\\Users\\aashi\\OneDrive\\Desktop\\New folder\\tfidf_vectorizer.pkl")  # Load TF-IDF vectorizer
mlp_model = joblib.load("C:\\Users\\aashi\\OneDrive\\Desktop\\New folder\\mlp_model.pkl")  # Load trained MLP model


# Function to parse text from file using PyPDF2 (for both job description and resumes)
def parse_text(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print("Error parsing PDF:", str(e))
        return ""


# Function to clean and preprocess text using SpaCy
def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    doc = nlp(text)  # Tokenize and process text with SpaCy
    tokens = [token.lemma_ for token in doc if token.is_alpha]  # Lemmatize tokens
    return " ".join(tokens)


# Function to compute cosine similarity
def cosine_sim(v1, v2):
    return cosine_similarity([v1], [v2])[0][0]


# Function to classify resumes using MLP model
def classify_resume_with_mlp(job_desc, resume_text):
    try:
        job_vector = tfidf_vectorizer.transform([job_desc]).toarray()
        resume_vector = tfidf_vectorizer.transform([resume_text]).toarray()

        features = np.concatenate((job_vector, resume_vector), axis=1)
        print("Feature shape:", features.shape)
        fit_category = mlp_model.predict(features)[0]

        return 1.0 if fit_category == 0 else 0.0
    except Exception as e:
        print("Error in MLP prediction:", str(e))
        return 0.0


# Function to calculate the weighted overall score
def calculate_overall_score(cosine_score, mlp_score, ner_matches_score):
    cosine_weight = 0.3
    mlp_weight = 0.4
    ner_weight = 0.3
    return (cosine_score * cosine_weight) + (mlp_score * mlp_weight) + (ner_matches_score * ner_weight)


# Function to extract skills, education, and experience from resume using NER
def extract_ner(resume_text):
    doc = nlp(resume_text)
    skills, education, experience = [], [], []

    for ent in doc.ents:
        if ent.label_ == 'SKILL':  # Assuming 'SKILL' is a custom label
            skills.append(ent.text)
        elif ent.label_ == 'ORG':
            education.append(ent.text)
        elif ent.label_ == 'DATE':
            experience.append(ent.text)

    return skills, education, experience


# Function to rank resumes based on the overall score
def rank_resumes(resumes, job_desc):
    ranked_resumes = []
    job_desc_text = preprocess_text(job_desc)

    for resume in resumes:
        try:
            resume_text = parse_text(resume)
            if not resume_text:
                continue
            resume_text = preprocess_text(resume_text)

            job_vector = tfidf_vectorizer.transform([job_desc_text]).toarray()
            resume_vector = tfidf_vectorizer.transform([resume_text]).toarray()
            cosine_score = cosine_sim(job_vector[0], resume_vector[0])

            mlp_score = classify_resume_with_mlp(job_desc_text, resume_text)

            skills, education, experience = extract_ner(resume_text)
            ner_matches_score = len(skills) * 0.5 + len(education) * 0.3 + len(experience) * 0.2

            overall_score = calculate_overall_score(cosine_score, mlp_score, ner_matches_score)

            ranked_resumes.append({
                'resume_name': resume.filename,
                'overall_score': overall_score
            })
        except Exception as e:
            print("Error processing resume:", str(e))

    ranked_resumes.sort(key=lambda x: x['overall_score'], reverse=True)
    return ranked_resumes


# Flask app setup
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        if request.method == 'POST':
            job_desc_pdf = request.files['job_description']
            job_desc_text = parse_text(job_desc_pdf)

            resume_files = request.files.getlist('resumes')
            ranked_resumes = rank_resumes(resume_files, job_desc_text)

            return render_template('upload_form.html', ranked_resumes=ranked_resumes)

        return render_template('upload_form.html')
    except Exception as e:
        print("Error:", str(e))
        traceback.print_exc()
        return "Internal Server Error", 500


if __name__ == '__main__':
    app.run(debug=True)
