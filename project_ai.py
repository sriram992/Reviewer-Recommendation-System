import streamlit as st
import os
import re
import torch
import pymupdf
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ============================
# PAGE CONFIGURATION
# ============================
st.set_page_config(
    page_title="Reviewer Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# CUSTOM CSS
# ============================
st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}
.main-header {
    font-size: 2.8rem; font-weight: 700; color: #1f77b4;
    text-align: center; margin-bottom: 0.5rem; letter-spacing: -0.5px;
}
.sub-header {
    font-size: 1.1rem; color: #666; text-align: center;
    margin-bottom: 2.5rem; font-weight: 400;
}
.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem; border-radius: 12px; color: white;
    text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.metric-value {font-size: 2.5rem; font-weight: 700; margin-bottom: 0.3rem;}
.metric-label {
    font-size: 0.9rem; opacity: 0.9; text-transform: uppercase;
    letter-spacing: 1px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem; background-color: #f8f9fa; padding: 0.5rem; border-radius: 10px;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #1f77b4; color: white;
}
.section-header {
    font-size: 1.5rem; font-weight: 600; color: #333;
    margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #e9ecef;
}
.network-info {
    background: #f8f9fa; padding: 1rem; border-radius: 8px;
    border-left: 4px solid #1f77b4; margin: 0.5rem 0;
}
.stButton button:hover {
    transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# ============================
# CONFIGURATION
# ============================
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================
# CO-AUTHOR EXTRACTION
# ============================
class CoAuthorExtractor:
    """Extract author names from paper text."""
    
    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None
        
        self.author_patterns = [
            r'(?:authors?|by)[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)?)'
        ]

    def _normalize_name(self, name):
        """Normalize author name."""
        return ' '.join(name.split()).title()

    def _extract_by_regex(self, text, max_chars=1500):
        """Fallback regex extraction."""
        header = text[:max_chars]
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
        """Extract author names using NER or regex."""
        if self.nlp:
            try:
                doc = self.nlp(text[:2000]) 
                ner_authors = set()
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        name = self._normalize_name(ent.text)
                        if 2 <= len(name.split()) <= 4 and all(word[0].isupper() for word in name.split() if word):
                            ner_authors.add(name)
                if ner_authors:
                    return list(ner_authors)
            except Exception:
                pass
        return self._extract_by_regex(text)


# ============================
# CO-AUTHORSHIP NETWORK
# ============================
class CoAuthorshipNetwork:
    """Build and query co-authorship network."""
    
    def __init__(self):
        self.network = defaultdict(set)
        self.paper_authors = {}
    
    def add_paper(self, paper_id, authors):
        self.paper_authors[paper_id] = authors
        for i, author1 in enumerate(authors):
            for author2 in authors[i+1:]:
                self.network[author1].add(author2)
                self.network[author2].add(author1)
    
    def get_coauthors(self, author):
        return self.network.get(author, set())
    
    def has_collaborated(self, author1, author2):
        return author2 in self.network.get(author1, set())
    
    def collaboration_count(self, author1, author2):
        return sum(author1 in authors and author2 in authors for authors in self.paper_authors.values())


# ============================
# DATA LOADING
# ============================
@st.cache_resource
def load_corpus_with_multi_authorship(base_dir):
    """Load all text files and build mappings."""
    extractor = CoAuthorExtractor()
    network = CoAuthorshipNetwork()

    papers = []
    author_paper_map = defaultdict(list)
    all_authors = set()
    paper_idx = 0

    for file in os.listdir(base_dir):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(base_dir, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                continue

            author_name = file.split("_")[0].strip()
            paper_authors = extractor.extract_authors(text)
            if author_name not in paper_authors:
                paper_authors.insert(0, author_name)

            paper_id = file.replace(".txt", "")
            papers.append({"text": text, "paper_id": paper_id, "authors": paper_authors})

            for author in paper_authors:
                author_paper_map[author].append(paper_idx)
                all_authors.add(author)

            network.add_paper(paper_id, paper_authors)
            paper_idx += 1
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    corpus_texts = [p["text"] for p in papers]
    unique_authors = sorted(list(all_authors))
    return unique_authors, corpus_texts, author_paper_map, papers, network


# ============================
# RECOMMENDER SYSTEM
# ============================
class MultiAuthorshipRecommender:
    """Recommender system using TF-IDF + SBERT + coauthor boost."""
    
    def __init__(self, unique_authors, corpus_texts, author_paper_map, papers, network,
                 use_coauthor_boost=True, coauthor_weight=0.2):
        
        self.unique_authors = unique_authors
        self.corpus_texts = corpus_texts
        self.author_paper_map = author_paper_map
        self.papers = papers
        self.network = network
        self.use_coauthor_boost = use_coauthor_boost
        self.coauthor_weight = coauthor_weight
        self.content_weight = 1.0 - coauthor_weight
        self._build_models()
    
    def _build_models(self):
        with st.spinner("🔧 Building TF-IDF model..."):
            self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus_texts)
        
        with st.spinner(f"🤖 Loading SBERT model ({SBERT_MODEL_NAME})..."):
            self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=DEVICE)
        
        with st.spinner("📊 Encoding corpus with SBERT..."):
            self.sbert_embeddings = self.sbert_model.encode(
                self.corpus_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32
            )
    
    def _calculate_author_content_score(self, query_vec, author, is_sbert=False):
        indices = self.author_paper_map.get(author, [])
        if not indices:
            return 0.0
        if is_sbert:
            scores = cosine_similarity(query_vec, self.sbert_embeddings[indices]).flatten()
        else:
            scores = cosine_similarity(query_vec, self.tfidf_matrix[indices]).flatten()
        return float(np.max(scores))
    
    def _calculate_coauthor_boost(self, paper_authors):
        boost = np.zeros(len(self.unique_authors))
        for idx, reviewer in enumerate(self.unique_authors):
            for pa in paper_authors:
                if self.network.has_collaborated(reviewer, pa):
                    collab_count = self.network.collaboration_count(reviewer, pa)
                    boost[idx] = min(1.0, collab_count / 5.0)
                    break
        return boost
    
    def _recommend_engine(self, text, is_sbert, paper_authors=None, top_k=5, exclude_authors=None):
        if not text.strip():
            return None
        
        query_vec = (
            self.sbert_model.encode([text], convert_to_numpy=True)
            if is_sbert else self.tfidf_vectorizer.transform([text])
        )
        
        content_scores = [
            self._calculate_author_content_score(query_vec, author, is_sbert)
            for author in self.unique_authors
        ]
        
        df = pd.DataFrame({'Author': self.unique_authors, 'Content': content_scores})
        if self.use_coauthor_boost and paper_authors:
            boost = self._calculate_coauthor_boost(paper_authors)
            df['CoAuthor'] = boost
            df['Score'] = self.content_weight * df['Content'] + self.coauthor_weight * df['CoAuthor']
        else:
            df['Score'] = df['Content']
        
        if exclude_authors:
            df = df[~df['Author'].isin(exclude_authors)]
        
        df = df.sort_values(by='Score', ascending=False).head(top_k).reset_index(drop=True)
        df.insert(0, 'Rank', range(1, len(df) + 1))
        df = df.round(4)
        return df
    
    def recommend(self, text, paper_authors=None, top_k=5, method='both', exclude_authors=None):
        results = {}
        if method in ['tfidf', 'both']:
            results['tfidf'] = self._recommend_engine(text, False, paper_authors, top_k, exclude_authors)
        if method in ['sbert', 'both']:
            results['sbert'] = self._recommend_engine(text, True, paper_authors, top_k, exclude_authors)
        return results


# ============================
# HELPER FUNCTIONS
# ============================
def extract_text_from_pdf(pdf_file):
    """Extract and clean text from PDF."""
    try:
        doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""


def create_network_visualization(network, selected_authors):
    """Simple textual network view."""
    if not selected_authors:
        return None
    info = ""
    for author in selected_authors[:5]:
        coauthors = list(network.get_coauthors(author))[:3]
        if coauthors:
            info += f"**{author}** → {', '.join(coauthors)}\n\n"
    return info or None


# ============================
# MAIN APPLICATION
# ============================
def main():
    st.markdown('<div class="main-header">📚 Reviewer Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Paper Review Matching with Co-Author Network Analysis</div>', unsafe_allow_html=True)

    base_dir = st.text_input("📁 Enter path to folder with text files:", value=r"folder/")

    if 'data_loaded' not in st.session_state and st.button("🚀 Load Corpus"):
        if os.path.exists(base_dir):
            with st.spinner(f"⏳ Loading data from: {base_dir} ..."):
                unique_authors, corpus_texts, author_paper_map, papers, network = load_corpus_with_multi_authorship(base_dir)
                if not papers:
                    st.error("❌ No valid papers found.")
                    return

                st.session_state.update({
                    'data_loaded': True,
                    'unique_authors': unique_authors,
                    'corpus_texts': corpus_texts,
                    'author_paper_map': author_paper_map,
                    'papers': papers,
                    'network': network,
                    'use_coauthor_boost': True,
                    'coauthor_weight': 0.2,
                    'recommender': MultiAuthorshipRecommender(
                        unique_authors, corpus_texts, author_paper_map, papers, network,
                        use_coauthor_boost=True, coauthor_weight=0.2)
                })
                st.success(f"✅ Loaded {len(papers)} papers from {len(unique_authors)} authors.")
                st.rerun()
        else:
            st.error(f"❌ Path does not exist: {base_dir}")
            return

    if 'data_loaded' not in st.session_state:
        st.info("💡 Please load the dataset first.")
        return

    # Show metrics
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='metric-container'><div class='metric-value'>{len(st.session_state.papers)}</div><div class='metric-label'>Total Papers</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-container'><div class='metric-value'>{len(st.session_state.unique_authors)}</div><div class='metric-label'>Total Authors</div></div>", unsafe_allow_html=True)
    avg_authors = np.mean([len(p['authors']) for p in st.session_state.papers])
    col3.markdown(f"<div class='metric-container'><div class='metric-value'>{avg_authors:.1f}</div><div class='metric-label'>Avg Authors/Paper</div></div>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📄 Recommend Reviewers", "👥 Author Pool", "📈 Analytics"])

    # ============ TAB 1 ============
    with tab1:
        st.markdown('<div class="section-header">📤 Upload Paper for Review</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop PDF here", type=['pdf'])
        col1, col2, col3 = st.columns([1, 1, 1])
        top_k = col1.number_input("Top K", min_value=1, max_value=20, value=5)
        exclude_authors = col2.checkbox("Exclude Paper Authors", value=True)
        use_boost = col3.checkbox("Enable Co-author Boost", value=True)

        if uploaded_file and st.button("🚀 Generate Recommendations", type="primary"):
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error("❌ Could not extract text.")
                return

            extractor = CoAuthorExtractor()
            paper_authors = extractor.extract_authors(text)
            st.session_state.recommender.use_coauthor_boost = use_boost

            if paper_authors:
                st.success(f"📝 Detected Authors: {', '.join(paper_authors[:5])}")

            exclude = paper_authors if exclude_authors else None
            results = st.session_state.recommender.recommend(text, paper_authors, top_k, 'both', exclude)

            st.markdown('<div class="section-header">🏆 Recommended Reviewers</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            if results.get('tfidf') is not None:
                col1.markdown("### 📊 TF-IDF Rankings")
                col1.dataframe(results['tfidf'], use_container_width=True)
            if results.get('sbert') is not None:
                col2.markdown("### 🤖 SBERT Rankings")
                col2.dataframe(results['sbert'], use_container_width=True)

            if use_boost and paper_authors:
                st.markdown('<div class="section-header">🌐 Collaboration Network</div>', unsafe_allow_html=True)
                viz = create_network_visualization(st.session_state.network, results['sbert']['Author'].tolist())
                if viz:
                    st.markdown(f'<div class="network-info">{viz}</div>', unsafe_allow_html=True)
                else:
                    st.info("No collaboration data available.")

    # ============ TAB 2 ============
    with tab2:
        st.markdown('<div class="section-header">👥 Reviewer Pool Overview</div>', unsafe_allow_html=True)
        search = st.text_input("🔍 Search Author", "")
        data = [
            {'Author': a,
             'Papers': len(st.session_state.author_paper_map.get(a, [])),
