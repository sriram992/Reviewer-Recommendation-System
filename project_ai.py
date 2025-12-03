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

# Custom CSS
st.markdown("""
<style>
[data-testid="stSidebar"] {
    display: none;
}
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}
.sub-header {
    font-size: 1.1rem;
    color: #666;
    text-align: center;
    margin-bottom: 2.5rem;
    font-weight: 400;
}
.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.metric-label {
    font-size: 0.9rem;
    opacity: 0.9;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background-color: #f8f9fa;
    padding: 0.5rem;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #1f77b4;
    color: white;
}
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #333;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e9ecef;
}
.network-info {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# ============================
# CONFIGURATION
# ============================
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 🚨 UPDATED PATH: CSV file path
CSV_PATH = "folder/corpus_index.csv"

# ============================
# CO-AUTHOR EXTRACTION
# ============================
class CoAuthorExtractor:
    """Extract author names from paper text."""
    
    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
        
        self.author_patterns = [
            r'(?:authors?|by)[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)?)',
        ]

    def _normalize_name(self, name):
        """Normalize author name for matching."""
        return ' '.join(name.split()).title()

    def _extract_by_regex(self, text, max_chars=1500):
        """Fallback method using simple REGEX heuristics."""
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
        """Extracts author names using NER or REGEX."""
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
            except:
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
        """Add a paper and its authors to the network."""
        self.paper_authors[paper_id] = authors
        for i, author1 in enumerate(authors):
            for author2 in authors[i+1:]:
                self.network[author1].add(author2)
                self.network[author2].add(author1)
    
    def get_coauthors(self, author):
        """Get all co-authors of a given author."""
        return self.network.get(author, set())
    
    def has_collaborated(self, author1, author2):
        """Check if two authors have collaborated."""
        return author2 in self.network.get(author1, set())
    
    def collaboration_count(self, author1, author2):
        """Count number of collaborations between two authors."""
        count = 0
        for paper_id, authors in self.paper_authors.items():
            if author1 in authors and author2 in authors:
                count += 1
        return count

# ============================
# DATA LOADING FROM CSV
# ============================
@st.cache_resource
def load_corpus_from_csv(csv_path):
    """Load all papers from CSV and create necessary data structures."""
    
    network = CoAuthorshipNetwork()
    papers = []
    author_paper_map = defaultdict(list)
    all_authors = set()
    
    if not os.path.exists(csv_path):
        st.error(f"❌ Error: Corpus file not found at: {csv_path}")
        return None, None, None, None, None
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {e}")
        return None, None, None, None, None

    # Ensure required columns exist
    required_cols = ['text_content', 'paper_id', 'folder_owner']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ CSV file must contain columns: {required_cols}")
        st.info(f"Found columns: {list(df.columns)}")
        return None, None, None, None, None

    # Initialize CoAuthorExtractor for extracting authors from text
    extractor = CoAuthorExtractor()
    
    for paper_idx, row in df.iterrows():
        try:
            text = row['text_content']
            paper_id = str(row['paper_id'])
            folder_owner = str(row['folder_owner']).strip()
            
            # Skip if text is missing or empty
            if pd.isna(text) or not str(text).strip():
                continue
            
            text = str(text)
            
            # Extract authors from the text content
            # This will attempt to find author names in the paper text
            extracted_authors = extractor.extract_authors(text)
            
            # Use folder_owner as primary author if no authors extracted
            if not extracted_authors:
                authors_list = [folder_owner] if folder_owner and folder_owner != 'nan' else []
            else:
                # Include folder_owner in the authors list if not already present
                authors_list = extracted_authors.copy()
                if folder_owner and folder_owner != 'nan' and folder_owner not in authors_list:
                    authors_list.insert(0, folder_owner)
            
            # Skip papers with no authors
            if not authors_list:
                continue

            papers.append({
                'text': text,
                'paper_id': paper_id,
                'authors': authors_list,
                'folder_owner': folder_owner
            })
            
            # Add authors to mapping
            for author in authors_list:
                author_paper_map[author].append(paper_idx)
                all_authors.add(author)
            
            # Add to network
            network.add_paper(paper_id, authors_list)

        except Exception as e:
            st.warning(f"⚠️ Skipped row {paper_idx} due to data error: {e}")
            continue

    if not papers:
        st.error("❌ No valid papers found in CSV file!")
        return None, None, None, None, None

    corpus_texts = [p['text'] for p in papers]
    unique_authors = sorted(list(all_authors))
    
    st.info(f"📊 Loaded {len(papers)} papers with {len(unique_authors)} unique authors")
    
    return unique_authors, corpus_texts, author_paper_map, papers, network


# ============================
# RECOMMENDER SYSTEM
# ============================
class MultiAuthorshipRecommender:
    """Recommender where ALL authors on a paper get full content credit."""
    
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
        
        self.dynamic_authors = []
        self._build_models()
    
    def _build_models(self):
        """Build TF-IDF and SBERT models."""
        with st.spinner("🔧 Building TF-IDF model..."):
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=1,
                max_df=1.0
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus_texts)
        
        with st.spinner(f"🤖 Loading SBERT model ({SBERT_MODEL_NAME})..."):
            self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=DEVICE)
        
        with st.spinner("📊 Encoding corpus with SBERT..."):
            self.sbert_embeddings = self.sbert_model.encode(
                self.corpus_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32
            )
    
    def _calculate_author_content_score(self, query_vec, author, is_sbert=False):
        """Calculate content score for an author based on ALL papers they authored."""
        paper_indices = self.author_paper_map.get(author, [])
        
        if not paper_indices:
            return 0.0
        
        if is_sbert:
            author_embeddings = self.sbert_embeddings[paper_indices]
            scores = cosine_similarity(query_vec, author_embeddings).flatten()
        else:
            author_vectors = self.tfidf_matrix[paper_indices]
            scores = cosine_similarity(query_vec, author_vectors).flatten()
        
        return float(np.max(scores))
    
    def _calculate_coauthor_boost(self, paper_authors):
        """Calculate co-authorship boost for each reviewer."""
        boost_scores = np.zeros(len(self.unique_authors))
        
        for idx, reviewer in enumerate(self.unique_authors):
            for paper_author in paper_authors:
                if self.network.has_collaborated(reviewer, paper_author):
                    collab_count = self.network.collaboration_count(reviewer, paper_author)
                    boost_scores[idx] = min(1.0, collab_count / 5.0)
                    break
        
        return boost_scores
    
    def _recommend_engine(self, text, is_sbert, paper_authors=None, top_k=5, exclude_authors=None):
        """Core recommendation engine."""
        if not text.strip():
            return None
        
        if is_sbert:
            query_vec = self.sbert_model.encode([text], convert_to_numpy=True)
        else:
            query_vec = self.tfidf_vectorizer.transform([text])
        
        content_scores = []
        for author in self.unique_authors:
            score = self._calculate_author_content_score(query_vec, author, is_sbert)
            content_scores.append(score)
        
        results_df = pd.DataFrame({
            'Author': self.unique_authors,
            'Content': content_scores
        })
        
        if self.use_coauthor_boost and paper_authors:
            boost_scores = self._calculate_coauthor_boost(paper_authors)
            results_df['CoAuthor'] = boost_scores
            results_df['Score'] = (
                self.content_weight * results_df['Content'] +
                self.coauthor_weight * results_df['CoAuthor']
            )
        else:
            results_df['Score'] = results_df['Content']
        
        if exclude_authors:
            results_df = results_df[~results_df['Author'].isin(exclude_authors)]
        
        results_df = results_df.sort_values(by='Score', ascending=False)
        top_results = results_df.head(top_k).copy()
        
        top_results.insert(0, 'Rank', range(1, len(top_results) + 1))
        top_results['Score'] = top_results['Score'].round(4)
        top_results['Content'] = top_results['Content'].round(4)
        
        if 'CoAuthor' in top_results.columns:
            top_results['CoAuthor'] = top_results['CoAuthor'].round(4)
        
        return top_results
    
    def recommend(self, text, paper_authors=None, top_k=5, method='both', exclude_authors=None):
        """Get reviewer recommendations."""
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
        
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""

def create_network_visualization(network, selected_authors):
    """Create a simple network visualization of collaborations."""
    if not selected_authors:
        return None
    
    network_text = ""
    for author in selected_authors[:5]:
        coauthors = list(network.get_coauthors(author))[:3]
        if coauthors:
            network_text += f"**{author}** → {', '.join(coauthors)}\n\n"
    
    return network_text if network_text else None

# ============================
# MAIN APPLICATION
# ============================
def main():
    # Header
    st.markdown('<div class="main-header">📚 Reviewer Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Paper Review Matching with Co-Author Network Analysis</div>', unsafe_allow_html=True)
    
    # Initialize data on first load
    if 'data_loaded' not in st.session_state:
        if os.path.exists(CSV_PATH):
            with st.spinner(f"⏳ Loading data from {CSV_PATH} and initializing models..."):
                try:
                    unique_authors, corpus_texts, author_paper_map, papers, network = \
                        load_corpus_from_csv(CSV_PATH)
                    
                    if unique_authors is None:
                        st.stop()
                        
                    st.session_state.data_loaded = True
                    st.session_state.unique_authors = unique_authors
                    st.session_state.corpus_texts = corpus_texts
                    st.session_state.author_paper_map = author_paper_map
                    st.session_state.papers = papers
                    st.session_state.network = network
                    st.session_state.use_coauthor_boost = True
                    st.session_state.coauthor_weight = 0.2
                    
                    # Initialize recommender
                    recommender = MultiAuthorshipRecommender(
                        unique_authors, corpus_texts, author_paper_map, papers, network,
                        use_coauthor_boost=True, coauthor_weight=0.2
                    )
                    st.session_state.recommender = recommender
                    
                    st.success(f"✅ System ready! Loaded {len(papers)} papers from {len(unique_authors)} authors.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error processing CSV data or initializing models: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
        else:
            st.error(f"❌ Corpus CSV file not found at: {CSV_PATH}")
            st.info("💡 Please ensure the CSV file is present and update the CSV_PATH variable if necessary.")
            st.stop()
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{len(st.session_state.papers)}</div>
            <div class="metric-label">Total Papers</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{len(st.session_state.unique_authors)}</div>
            <div class="metric-label">Total Authors</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_authors = np.mean([len(p['authors']) for p in st.session_state.papers])
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{avg_authors:.1f}</div>
            <div class="metric-label">Avg Authors/Paper</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📄 Recommend Reviewers", "👥 Author Pool", "📈 Analytics"])
    
    # Tab 1: Recommend Reviewers
    with tab1:
        st.markdown('<div class="section-header">📤 Upload Paper for Review</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Drop your PDF file here or click to browse", type=['pdf'], label_visibility="collapsed")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            top_k = st.number_input("Number of Recommendations", min_value=1, max_value=20, value=5)
        with col2:
            exclude_authors = st.checkbox("Exclude Paper Authors", value=True)
        with col3:
            use_boost = st.checkbox("Enable Co-author Boost", value=True)
        
        if uploaded_file is not None:
            if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing paper and generating recommendations..."):
                    # Extract text
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if not text:
                        st.error("❌ Could not extract text from PDF! Please ensure the PDF is readable.")
                        return
                    
                    # Extract authors
                    extractor = CoAuthorExtractor()
                    paper_authors = extractor.extract_authors(text)
                    
                    # Update boost setting
                    st.session_state.recommender.use_coauthor_boost = use_boost
                    
                    # Show detected authors
                    if paper_authors:
                        st.success(f"**📝 Detected Authors:** {', '.join(paper_authors[:5])}")
                        
                        known = [a for a in paper_authors if a in st.session_state.unique_authors]
                        new = [a for a in paper_authors if a not in st.session_state.unique_authors]
                        
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            if known:
                                st.info(f"✅ {len(known)} author(s) found in corpus")
                        with col_info2:
                            if new:
                                st.warning(f"🆕 {len(new)} new author(s) detected")
                    
                    # Get recommendations
                    authors_to_exclude = paper_authors if exclude_authors else None
                    results = st.session_state.recommender.recommend(
                        text,
                        paper_authors=paper_authors,
                        top_k=top_k,
                        method='both',
                        exclude_authors=authors_to_exclude
                    )
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-header">🏆 Recommended Reviewers</div>', unsafe_allow_html=True)
                    
                    # Display results
                    if results:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📊 TF-IDF Based Rankings")
                            if 'tfidf' in results:
                                st.dataframe(
                                    results['tfidf'], 
                                    use_container_width=True, 
                                    hide_index=True,
                                    height=400
                                )
                        
                        with col2:
                            st.markdown("### 🤖 SBERT Based Rankings")
                            if 'sbert' in results:
                                st.dataframe(
                                    results['sbert'], 
                                    use_container_width=True, 
                                    hide_index=True,
                                    height=400
                                )
                        
                        # Network visualization
                        if use_boost and paper_authors:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown('<div class="section-header">🌐 Collaboration Network</div>', unsafe_allow_html=True)
                            top_reviewers = results['sbert']['Author'].tolist()
                            network_viz = create_network_visualization(st.session_state.network, top_reviewers)
                            if network_viz:
                                st.markdown(f'<div class="network-info">{network_viz}</div>', unsafe_allow_html=True)
                            else:
                                st.info("ℹ️ No collaboration data available for top recommended reviewers.")
    
    # Tab 2: Author Pool
    with tab2:
        st.markdown('<div class="section-header">👥 Reviewer Pool Overview</div>', unsafe_allow_html=True)
        
        # Search
        search_query = st.text_input("🔍 Search authors by name", "", placeholder="Enter author name...")
        
        # Create DataFrame
        author_data = []
        for author in st.session_state.unique_authors:
            paper_count = len(st.session_state.author_paper_map.get(author, []))
            coauthor_count = len(st.session_state.network.get_coauthors(author))
            author_data.append({
                'Author': author,
                'Papers': paper_count,
                'Collaborators': coauthor_count
            })
        
        df = pd.DataFrame(author_data)
        
        # Filter
        if search_query:
            df = df[df['Author'].str.contains(search_query, case=False)]
        
        # Display
        st.dataframe(df, use_container_width=True, hide_index=True, height=500)
        
        # Statistics
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Authors", len(df))
        with col2:
            st.metric("📄 Avg Papers/Author", f"{df['Papers'].mean():.1f}")
        with col3:
            st.metric("🤝 Avg Collaborators", f"{df['Collaborators'].mean():.1f}")
    
    # Tab 3: Analytics
    with tab3:
        st.markdown('<div class="section-header">📈 Dataset Analytics</div>', unsafe_allow_html=True)
        
        # Create analytics
        author_data = []
        for author in st.session_state.unique_authors:
            paper_count = len(st.session_state.author_paper_map.get(author, []))
            coauthor_count = len(st.session_state.network.get_coauthors(author))
            author_data.append({
                'Author': author,
                'Papers': paper_count,
                'Collaborators': coauthor_count
            })
        
        df = pd.DataFrame(author_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Top 15 Most Prolific Authors")
            top_authors = df.nlargest(15, 'Papers')[['Author', 'Papers']]
            st.dataframe(top_authors, use_container_width=True, hide_index=True, height=500)
        
        with col2:
            st.markdown("### 🌐 Top 15 Most Connected Authors")
            top_collab = df.nlargest(15, 'Collaborators')[['Author', 'Collaborators']]
            st.dataframe(top_collab, use_container_width=True, hide_index=True, height=500)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Total Papers", len(st.session_state.papers))
        with col2:
            st.metric("👥 Total Authors", len(df))
        with col3:
            st.metric("📊 Max Papers by Author", df['Papers'].max())
        with col4:
            st.metric("🤝 Max Collaborators", df['Collaborators'].max())

if __name__ == "__main__":
    main()

