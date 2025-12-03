import streamlit as st
import os
import re
import pickle # Added pickle
import torch
import pymupdf
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ============================
# PAGE CONFIGURATION
# ============================
st.set_page_config(page_title="Reviewer Recommendation System", page_icon="📚", layout="wide")

# Custom CSS (Keeping your original styling)
st.markdown("""
<style>
[data-testid="stSidebar"] { display: none; }
.main-header { font-size: 2.8rem; font-weight: 700; color: #1f77b4; text-align: center; margin-bottom: 0.5rem; }
.sub-header { font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2.5rem; }
.metric-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center; }
.metric-value { font-size: 2.5rem; font-weight: 700; }
.metric-label { font-size: 0.9rem; opacity: 0.9; text-transform: uppercase; }
.section-header { font-size: 1.5rem; font-weight: 600; color: #333; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #e9ecef; }
</style>
""", unsafe_allow_html=True)

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================
# CLASSES (Network & Extractor needed for functionality)
# ============================
class CoAuthorExtractor:
    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
        self.author_patterns = [r'(?:authors?|by)[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)?)']

    def _normalize_name(self, name):
        return ' '.join(name.split()).title()

    def _extract_by_regex(self, text):
        header = text[:1500]
        authors = set()
        for pattern in self.author_patterns:
            matches = re.findall(pattern, header, re.MULTILINE)
            for match in matches:
                names = re.split(r',|\sand\s', match)
                for name in names:
                    name = name.strip()
                    if 2 <= len(name.split()) <= 4:
                        if all(word[0].isupper() for word in name.split() if word):
                            authors.add(self._normalize_name(name))
        return list(authors)
        
    def extract_authors(self, text):
        if self.nlp:
            try:
                doc = self.nlp(text[:2000]) 
                ner_authors = set()
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        name = self._normalize_name(ent.text)
                        if 2 <= len(name.split()) <= 4 and all(word[0].isupper() for word in name.split() if word):
                            ner_authors.add(name)
                if ner_authors: return list(ner_authors)
            except: pass
        return self._extract_by_regex(text)

class CoAuthorshipNetwork:
    def __init__(self):
        self.network = {} # Will be loaded from pickle
        self.paper_authors = {}
    
    # We add these methods back so the visualization still works
    def get_coauthors(self, author):
        return self.network.get(author, set())
    def has_collaborated(self, author1, author2):
        return author2 in self.network.get(author1, set())
    def collaboration_count(self, author1, author2):
        # Simplified for pre-calc version (approximate or requires saved paper_authors)
        return 1 if self.has_collaborated(author1, author2) else 0

# ============================
# RECOMMENDER SYSTEM (UPDATED)
# ============================
class MultiAuthorshipRecommender:
    def __init__(self, data_dict, sbert_model):
        self.unique_authors = data_dict['unique_authors']
        self.author_paper_map = data_dict['author_paper_map']
        self.network = data_dict['network']
        self.tfidf_vectorizer = data_dict['tfidf_vectorizer']
        self.tfidf_matrix = data_dict['tfidf_matrix']
        self.sbert_embeddings = data_dict['sbert_embeddings']
        self.sbert_model = sbert_model # Passed from main

        self.use_coauthor_boost = True
        self.coauthor_weight = 0.2
        self.content_weight = 0.8

    def _calculate_author_content_score(self, query_vec, author, is_sbert=False):
        paper_indices = self.author_paper_map.get(author, [])
        if not paper_indices: return 0.0
        
        if is_sbert:
            author_embeddings = self.sbert_embeddings[paper_indices]
            scores = cosine_similarity(query_vec, author_embeddings).flatten()
        else:
            author_vectors = self.tfidf_matrix[paper_indices]
            scores = cosine_similarity(query_vec, author_vectors).flatten()
        
        return float(np.max(scores))

    def _recommend_engine(self, text, is_sbert, paper_authors=None, top_k=5, exclude_authors=None):
        if not text.strip(): return None
        
        # Encoding Query
        if is_sbert:
            query_vec = self.sbert_model.encode([text], convert_to_numpy=True)
        else:
            query_vec = self.tfidf_vectorizer.transform([text])
        
        # Scoring
        content_scores = []
        for author in self.unique_authors:
            score = self._calculate_author_content_score(query_vec, author, is_sbert)
            content_scores.append(score)
        
        results_df = pd.DataFrame({'Author': self.unique_authors, 'Content': content_scores})
        
        # Co-author Boost
        if self.use_coauthor_boost and paper_authors:
            boost_scores = np.zeros(len(self.unique_authors))
            for idx, reviewer in enumerate(self.unique_authors):
                for paper_author in paper_authors:
                    if self.network.has_collaborated(reviewer, paper_author):
                        boost_scores[idx] = 0.5 # Simplified fixed boost
                        break
            results_df['CoAuthor'] = boost_scores
            results_df['Score'] = (self.content_weight * results_df['Content'] + self.coauthor_weight * results_df['CoAuthor'])
        else:
            results_df['Score'] = results_df['Content']
        
        if exclude_authors:
            results_df = results_df[~results_df['Author'].isin(exclude_authors)]
        
        return results_df.sort_values(by='Score', ascending=False).head(top_k)

    def recommend(self, text, paper_authors=None, top_k=5, exclude_authors=None):
        results = {}
        results['tfidf'] = self._recommend_engine(text, False, paper_authors, top_k, exclude_authors)
        results['sbert'] = self._recommend_engine(text, True, paper_authors, top_k, exclude_authors)
        return results

# ============================
# HELPER FUNCTIONS
# ============================
def extract_text_from_file(uploaded_file):
    text = ""
    try:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext == ".txt":
            text = uploaded_file.read().decode("utf-8")
        elif file_ext == ".pdf":
            doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
            text = "".join(page.get_text() for page in doc)
    except: return ""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', text.lower())).strip() if text else ""

def create_network_visualization(network, selected_authors):
    network_text = ""
    for author in selected_authors[:5]:
        coauthors = list(network.get_coauthors(author))[:3]
        if coauthors: network_text += f"**{author}** → {', '.join(coauthors)}\n\n"
    return network_text

# ============================
# MAIN APPLICATION
# ============================
def main():
    st.markdown('<div class="main-header">📚 Reviewer Recommendation System</div>', unsafe_allow_html=True)
    
    # --- LOAD PRE-CALCULATED DATA ---
    @st.cache_resource
    def load_system():
        # 1. Load the SBERT Model (needed for query encoding)
        model = SentenceTransformer(SBERT_MODEL_NAME)
        
        # 2. Load the Pickle File
        pkl_path = "reviewer_data.pkl"
        
        if not os.path.exists(pkl_path):
            return None, None
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        return data, model

    with st.spinner("🚀 Loading pre-computed database..."):
        data_dict, sbert_model = load_system()
    
    if data_dict is None:
        st.error("❌ 'reviewer_data.pkl' not found! Please run 'generate_data.py' locally and upload the result.")
        st.stop()
        
    # Initialize Recommender
    if 'recommender' not in st.session_state:
        st.session_state.recommender = MultiAuthorshipRecommender(data_dict, sbert_model)
        st.session_state.unique_authors = data_dict['unique_authors']
        st.session_state.papers = data_dict['papers']
        st.session_state.network = data_dict['network']

    # --- UI STATISTICS ---
    st.success(f"✅ System Online! Loaded {len(st.session_state.papers)} papers.")
    
    col1, col2 = st.columns(2)
    col1.metric("Total Papers", len(st.session_state.papers))
    col2.metric("Total Authors", len(st.session_state.unique_authors))
    
    # --- TABS ---
    tab1, tab2 = st.tabs(["📄 Recommend", "📊 Analytics"])
    
    with tab1:
        st.markdown("### Upload Paper")
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=['pdf', 'txt'])
        
        col1, col2 = st.columns(2)
        top_k = col1.number_input("Count", 1, 20, 5)
        exclude_authors = col2.checkbox("Exclude Authors", True)
        
        if uploaded_file and st.button("Generate Recommendations", type="primary"):
            text = extract_text_from_file(uploaded_file)
            if not text:
                st.error("No text found.")
            else:
                extractor = CoAuthorExtractor()
                paper_authors = extractor.extract_authors(text)
                
                if paper_authors: st.info(f"Detected Authors: {', '.join(paper_authors)}")
                
                exclude = paper_authors if exclude_authors else None
                results = st.session_state.recommender.recommend(text, paper_authors, top_k, exclude)
                
                c1, c2 = st.columns(2)
                c1.markdown("### TF-IDF Results")
                c1.dataframe(results['tfidf'], hide_index=True)
                c2.markdown("### SBERT Results")
                c2.dataframe(results['sbert'], hide_index=True)

    with tab2:
        st.markdown("### Database Overview")
        st.write(f"This system contains {len(st.session_state.papers)} research papers from {len(st.session_state.unique_authors)} unique authors.")

if __name__ == "__main__":
    main()
