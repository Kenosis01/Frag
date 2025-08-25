import hashlib, re, sqlite3
from collections import Counter
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from rank_bm25 import BM25Okapi
import sys

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class Frag:
    """Fingerprint Retrieval Augmented Generation system."""
    
    def __init__(self, ngram_size=3, top_features=12, db_path="frag.db"):
        self.ngram_size, self.top_features = ngram_size, top_features
        self.db_path = db_path
        
        self.init_db()
        
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
        except OSError:
            print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        self.stop_words = set(stopwords.words('english'))
        # Remove YAKE as per specification for minimal approach
        
        self.bm25 = None
        self.corpus_features = []
    
    def init_db(self):
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                features TEXT,
                metadata TEXT,
                source_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_fingerprint ON chunks(fingerprint)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_source ON chunks(source_file)')
        self.conn.commit()
    
    def build_bm25_corpus(self):
        """Build BM25 corpus from stored chunks."""
        cursor = self.conn.execute('SELECT features FROM chunks')
        self.corpus_features = []
        
        for row in cursor:
            features_str = row[0]
            if features_str:
                features = features_str.split('|')
                self.corpus_features.append(features)
        
        if self.corpus_features:
            self.bm25 = BM25Okapi(self.corpus_features)
        else:
            self.bm25 = None
    
    def extract_features(self, text, is_query=False):
        """Extract sparse features from text - fully dynamic, no hardcoded patterns."""
        if not text.strip():
            return []
            
        features = []
        
        # Basic text preprocessing
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        
        if self.nlp:
            doc = self.nlp(text_clean)
            # Extract all meaningful content words dynamically
            tokens = [token.lemma_.lower() for token in doc 
                     if (not token.is_stop and token.is_alpha and len(token.lemma_) > 2)]
        else:
            # Fallback without spaCy - extract all words longer than 2 chars
            tokens = [t.lower() for t in re.findall(r'\b\w{3,}\b', text_clean) 
                     if t.lower() not in self.stop_words]
        
        if not tokens:
            return []
        
        # Dynamic word frequency analysis
        word_freq = Counter(tokens)
        total_words = len(tokens)
        
        # Select top words based on frequency distribution
        feature_limit = self.top_features * 2 if is_query else self.top_features
        for word, count in word_freq.most_common(feature_limit):
            # Dynamic threshold based on document characteristics
            min_freq = max(1, total_words // 100)  # At least 1% frequency or minimum 1
            if count >= min_freq:
                features.append(word)
        
        # Dynamic n-gram extraction
        if len(tokens) >= 2:
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
            bigram_freq = Counter(bigrams)
            
            # Add most frequent bigrams
            for bg, count in bigram_freq.most_common(feature_limit // 2):
                if count >= max(1, len(bigrams) // 50):  # Dynamic threshold
                    features.append(bg)
        
        # Dynamic trigram extraction for longer texts
        if len(tokens) >= 3 and total_words > 50:
            trigrams = [f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}" for i in range(len(tokens) - 2)]
            trigram_freq = Counter(trigrams)
            
            for tg, count in trigram_freq.most_common(feature_limit // 4):
                if count >= 2:  # Only include repeated trigrams
                    features.append(tg)
        
        # Dynamic named entity extraction (all entity types)
        if self.nlp:
            doc = self.nlp(text)
            entities_found = set()
            for ent in doc.ents:
                if len(ent.text.strip()) > 2:
                    entity_clean = ent.text.lower().replace(' ', '_').replace('-', '_')
                    if entity_clean not in entities_found and len(entity_clean) > 2:
                        features.append(f"ent_{entity_clean}")
                        entities_found.add(entity_clean)
        
        # Remove duplicates and return
        return list(set(features))
    
    def add_chunk(self, text, meta=None):
        """Add chunk to database."""
        if not text.strip():
            return None
            
        features = self.extract_features(text)
        fp = hashlib.md5('|'.join(features).encode()).hexdigest()[:12]
        
        cursor = self.conn.execute('''
            INSERT INTO chunks (content, fingerprint, features, metadata, source_file)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            text, 
            fp, 
            '|'.join(features),
            str(meta or {}),
            meta.get('source', '') if meta else ''
        ))
        self.conn.commit()
        
        self.build_bm25_corpus()
        
        return cursor.lastrowid
    
    def chunk_document(self, file_path):
        """Chunk document into smaller pieces."""
        elements = partition(filename=file_path)
        chunks = chunk_by_title(
            elements, 
            max_characters=500,
            new_after_n_chars=400,
            overlap=50
        )
        return [str(chunk).strip() for chunk in chunks if len(str(chunk).strip()) > 30]
    
    def add_document(self, file_path):
        """Process and add entire document."""
        chunks = self.chunk_document(file_path)
        chunk_ids = []
        
        for i, chunk_text in enumerate(chunks):
            meta = {'source': file_path, 'chunk_index': i, 'total_chunks': len(chunks)}
            chunk_id = self.add_chunk(chunk_text, meta)
            if chunk_id:
                chunk_ids.append(chunk_id)
        
        return chunk_ids
    
    def retrieve(self, query, top_k=3):
        """Retrieve similar chunks using BM25 - fully dynamic matching."""
        if not self.bm25:
            self.build_bm25_corpus()
            if not self.bm25:
                return []
        
        # Extract query features dynamically
        query_features = self.extract_features(query, is_query=True)
        if not query_features:
            return []
        
        try:
            scores = self.bm25.get_scores(query_features)
        except:
            return []
        
        cursor = self.conn.execute(
            'SELECT id, content, features, metadata FROM chunks ORDER BY id'
        )
        
        results = []
        for i, row in enumerate(cursor):
            if i < len(scores):
                chunk_id, content, features, metadata = row
                score = scores[i]
                
                # Dynamic content matching boost
                query_words = [w.lower() for w in query.split() if len(w) > 2]
                content_lower = content.lower()
                
                # Count overlapping words between query and content
                word_matches = 0
                for word in query_words:
                    if word in content_lower:
                        word_matches += 1
                
                # Apply dynamic boost based on word overlap ratio
                if word_matches > 0:
                    overlap_ratio = word_matches / len(query_words)
                    boost_factor = 1 + (overlap_ratio * 0.5)  # Up to 50% boost
                    score *= boost_factor
                
                # Include all non-zero scores
                if score > 0:
                    results.append((chunk_id, score, content, metadata))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_stats(self):
        """Get database statistics."""
        cursor = self.conn.execute('SELECT COUNT(*) FROM chunks')
        total_chunks = cursor.fetchone()[0]
        
        cursor = self.conn.execute('SELECT COUNT(DISTINCT source_file) FROM chunks')
        unique_sources = cursor.fetchone()[0]
        
        return {
            'total_chunks': total_chunks,
            'unique_sources': unique_sources,
            'db_path': self.db_path
        }
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()