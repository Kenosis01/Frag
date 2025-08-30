import hashlib, re, json
from collections import Counter
import spacy
import nltk
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch
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
    """Fingerprint Retrieval Augmented Generation system - Elasticsearch powered."""
    
    def __init__(self, ngram_size=3, top_features=12, es_host="localhost:9200", index_name="frag_chunks"):
        self.ngram_size, self.top_features = ngram_size, top_features
        self.index_name = index_name
        
        # Initialize Elasticsearch client
        try:
            self.es = Elasticsearch([es_host])
            # Test connection
            if not self.es.ping():
                raise ConnectionError("Cannot connect to Elasticsearch")
        except Exception as e:
            print(f"Elasticsearch connection failed: {e}")
            print("Make sure Elasticsearch is running on {es_host}")
            self.es = None
            
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
        except OSError:
            print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize index
        self.init_index()
    

    
    
    def init_index(self):
        """Initialize Elasticsearch index with optimized mapping for text search."""
        if not self.es:
            return
            
        # Index mapping optimized for BM25 and feature search
        mapping = {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "standard",
                        "similarity": "BM25"
                    },
                    "features": {
                        "type": "text",
                        "analyzer": "keyword",
                        "similarity": "BM25"
                    },
                    "fingerprint": {"type": "keyword"},
                    "metadata": {"type": "object"},
                    "created_at": {"type": "date"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "feature_analyzer": {
                            "tokenizer": "keyword",
                            "filter": ["lowercase"]
                        }
                    }
                }
            }
        }
        
        # Create index if it doesn't exist
        try:
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(index=self.index_name, body=mapping)
                print(f"✅ Created Elasticsearch index: {self.index_name}")
        except Exception as e:
            print(f"Error creating index: {e}")
    
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
        """Add text chunk to Elasticsearch index."""
        if not text.strip() or not self.es:
            return None
            
        features = self.extract_features(text)
        fp = hashlib.md5('|'.join(features).encode()).hexdigest()[:12]
        
        # Prepare document for Elasticsearch
        doc = {
            'content': text,
            'fingerprint': fp,
            'features': ' '.join(features),  # Space-separated for text analysis
            'metadata': meta or {},
            'created_at': 'now'
        }
        
        try:
            # Index document in Elasticsearch
            result = self.es.index(index=self.index_name, body=doc)
            return result['_id']
        except Exception as e:
            print(f"Error indexing document: {e}")
            return None
    

    
    def retrieve(self, query, top_k=3):
        """Retrieve similar chunks using Elasticsearch BM25 search."""
        if not self.es:
            return []
        
        # Extract query features dynamically
        query_features = self.extract_features(query, is_query=True)
        if not query_features:
            return []
        
        # Build Elasticsearch query using multi_match for BM25 scoring
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content^2", "features"],  # Boost content field
                                "type": "best_fields",
                                "tie_breaker": 0.3
                            }
                        },
                        {
                            "multi_match": {
                                "query": ' '.join(query_features),
                                "fields": ["features^1.5", "content"],
                                "type": "cross_fields",
                                "tie_breaker": 0.5
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": top_k,
            "_source": ["content", "metadata", "fingerprint"]
        }
        
        try:
            # Execute search
            response = self.es.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                doc_id = hit['_id']
                score = hit['_score']
                source = hit['_source']
                
                # Dynamic content matching boost
                query_words = [w.lower() for w in query.split() if len(w) > 2]
                content_lower = source['content'].lower()
                
                # Count overlapping words between query and content
                word_matches = sum(1 for word in query_words if word in content_lower)
                
                # Apply dynamic boost based on word overlap ratio
                if word_matches > 0:
                    overlap_ratio = word_matches / len(query_words)
                    boost_factor = 1 + (overlap_ratio * 0.3)  # Up to 30% boost
                    score *= boost_factor
                
                results.append((doc_id, score, source['content'], source.get('metadata', {})))
            
            # Sort by adjusted score
            results.sort(key=lambda x: x[1], reverse=True)
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_stats(self):
        """Get Elasticsearch index statistics."""
        if not self.es:
            return {'error': 'Elasticsearch not connected'}
            
        try:
            # Get index stats
            stats = self.es.indices.stats(index=self.index_name)
            count_result = self.es.count(index=self.index_name)
            
            return {
                'total_chunks': count_result['count'],
                'index_name': self.index_name,
                'storage_type': 'elasticsearch',
                'index_size': stats['indices'][self.index_name]['total']['store']['size_in_bytes']
            }
        except Exception as e:
            return {'error': f'Stats error: {e}'}
    
    def clear_index(self):
        """Clear all documents from the index."""
        if not self.es:
            return False
            
        try:
            # Delete all documents
            self.es.delete_by_query(
                index=self.index_name,
                body={"query": {"match_all": {}}}
            )
            return True
        except Exception as e:
            print(f"Error clearing index: {e}")
            return False
    
    def delete_index(self):
        """Delete the entire index."""
        if not self.es:
            return False
            
        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                print(f"✅ Deleted index: {self.index_name}")
                return True
        except Exception as e:
            print(f"Error deleting index: {e}")
            return False