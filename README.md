# SwipeToHire

### NLP-Based Resume Screening, Matching & Job Role Recommendation System

SwipeToHire is an end-to-end **Data Science and NLP project** that analyzes resumes against job requirements, ranks candidates by relevance, evaluates skill alignment, and suggests suitable job roles based on resume profiles.

The system uses **NLP preprocessing, TF-IDF, cosine similarity, and skill matching** to provide explainable preliminary resume screening.

---
## Live Application

🚀 **Try SwipeToHire:**  
https://swipetohire.streamlit.app/

---
## Key Features
### Job Description Based Screening
- Paste an actual Job Description
- Upload multiple candidate resumes
- Rank candidates by Resume Relevance
- View Skill Coverage
- Identify Matched and Missing JD Skills
- Receive preliminary recommendations
- Download ranked results as CSV

### Job Role Based Screening
- Select from 25 predefined job roles
- Upload candidate resumes
- Compare resumes against the selected role profile
- View relevance, skill coverage, matched skills, and recommendations

### Job Role Suggestions
- Upload your resume
- Compare it against 25 job role profiles
- Get the **Top 5 suggested job roles**
- View Resume Relevance, Skill Coverage, and Matched Skills

---

## How It Works

```text
Resume / Job Description
        ↓
Text Extraction & Cleaning
        ↓
NLP Preprocessing
        ↓
TF-IDF Feature Extraction
        ↓
Cosine Similarity
        ↓
Skill Matching
        ↓
Resume Relevance + Skill Coverage
        ↓
Candidate Ranking / Job Role Suggestions
```

---

## NLP & Matching Approach

**TF-IDF** converts resume and job description text into numerical feature vectors.

**Cosine Similarity** measures the similarity between a resume and a Job Description or reference job role. This is displayed as **Resume Relevance**.

**Skill Coverage** measures the percentage of identified role-relevant technical and professional skills found in the resume.

The system also identifies **Matched Skills** and **Missing JD Skills**.

---

## Recommendation Levels

Candidates are given a preliminary recommendation based on Resume Relevance and Skill Coverage:

- 🟢 Strong Match
- 🔵 Good Match
- 🟡 Moderate Match
- 🔴 Low Match

These recommendations support initial screening and are not final hiring decisions.

---

## Dataset & Evaluation

The project uses the **Resume Dataset by Snehaan Bhawal**, containing approximately **2,484 resumes across 24 categories**.

The system was evaluated using strong-match, partial-match, and mismatch resume examples, along with score distribution, matched-skill analysis, and Top-K candidate evaluation.

---

## Technologies Used

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- TF-IDF & Cosine Similarity
- Regular Expressions
- Joblib
- PyPDF
- Streamlit
- Google Colab & VS Code

---

## Project Structure

```text
SwipeToHire/
│
├── app.py
├── SwipeToHire.ipynb
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

---

## Important Note

**Resume Relevance is not hiring accuracy or selection probability.** It represents textual similarity between resume content and job requirements.

**Skill Coverage does not prove candidate proficiency.** It represents the percentage of detected role-relevant skills found in the resume.

Final hiring decisions should also consider interviews, practical evaluations, experience, qualifications, and overall candidate suitability.

---

## Limitations

- TF-IDF does not provide deep semantic understanding
- Skill extraction uses a curated skill vocabulary
- Skill presence does not prove proficiency
- Experience quality and years are not independently verified
- Job role suggestions depend on predefined role profiles
- Image-only scanned PDFs require OCR before analysis

---

## Future Improvements

- Semantic embedding-based matching
- Advanced skill and experience extraction
- Larger and dynamic job role library
- Improved resume parsing

---

## Author

**Ayushman Pradhan**  
B.Tech Computer Science Engineering  
Artificial Intelligence & Machine Learning