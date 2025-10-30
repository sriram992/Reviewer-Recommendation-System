#Reviewer Recommendation System

An AI-powered system that helps match academic papers with the most relevant reviewers.  
It uses advanced NLP models and co-author network analysis to ensure accurate and meaningful reviewer recommendations.

---

#Key Features

  Two Matching Methods – Combines TF-IDF and SBERT models for precise reviewer matching.
  Author Extraction via NER + Regex Fallback** – Extracts author names using spaCy’s NER model, with a regex-based backup when NER is unavailable
  Co-Author Network Analysis – Boosts scores for reviewers connected through collaboration networks.
  PDF Upload Support – Automatically extracts paper text and author names.
  Author Exclusion – Automatically removes the paper’s own authors from the suggestions.
  Interactive Dashboard – Visualizes author statistics and co-authorship relationships.

---

#Installation

To set up and run the app locally, follow these steps:

```bash
git clone https://github.com/yourusername/reviewer-recommendation-system.git
cd reviewer-recommendation-system
python -m venv venv
venv\Scripts\activate  
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

#Usage

Once the installation is complete, launch the app using Streamlit:

```bash
streamlit run app.py
```

Then open your browser and go to:
  http://localhost:8501

---

#Project Structure

```
reviewer-recommendation-system/
├── project_ai.py
├── requirements.txt
├── folder/
│   ├── Author1_Paper1.txt
│   ├── Author2_Paper2.txt
│   └── ...
└── models/
```

---

#How It Works

1 Extracts text and author information from the uploaded paper.
2 Builds a co-author graph to analyze research collaboration patterns.
3 Computes similarity using TF-IDF and SBERT embeddings.
4 Combines both scores into a hybrid ranking for better recommendations.

```
Final Score = 0.8 × Content Similarity + 0.2 × Co-Author Boost
```

---

#Configuration

You can customize the model or weight values directly in `project_ai.py`:

```python
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
coauthor_weight = 0.2
```

---

#Requirements

```
streamlit>=1.28.0
torch>=2.0.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
pymupdf>=1.23.0
spacy>=3.7.0
```

---
