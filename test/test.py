import json
import time
from frag import Frag
from datetime import datetime

class FragTester:
    """Real-time testing framework for FRAG system."""
    
    def __init__(self):
        self.frag = Frag(db_path="test_frag.db")
        self.results = []
        
    def load_corpus(self, corpus_file="corpus.jsonl"):
        """Load corpus documents into FRAG system."""
        print("🔄 Loading corpus documents...")
        count = 0
        
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        doc = json.loads(line)
                        if 'text' not in doc or '_id' not in doc:
                            print(f"  ⚠️ Skipping line {line_num}: Missing required fields")
                            continue
                            
                        doc_id = self.frag.add_chunk(doc['text'], {'doc_id': doc['_id']})
                        count += 1
                        print(f"  ✅ Loaded doc {doc['_id']} -> chunk_id {doc_id}")
                        
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️ Skipping line {line_num}: Invalid JSON - {e}")
                        continue
                    except Exception as e:
                        print(f"  ⚠️ Error processing line {line_num}: {e}")
                        continue
                    
            print(f"\n📊 Total documents loaded: {count}")
            return True
            
        except FileNotFoundError:
            print(f"❌ Corpus file {corpus_file} not found")
            return False
        except Exception as e:
            print(f"❌ Error loading corpus: {e}")
            return False
    
    def load_queries(self, queries_file="queries.jsonl"):
        """Load test queries."""
        queries = []
        
        try:
            with open(queries_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        query = json.loads(line)
                        # Convert to expected format
                        formatted_query = {
                            'query_id': query['_id'],
                            'query': query['text']
                        }
                        queries.append(formatted_query)
                        
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️ Skipping query line {line_num}: Invalid JSON - {e}")
                        continue
                    except KeyError as e:
                        print(f"  ⚠️ Skipping query line {line_num}: Missing field {e}")
                        continue
                    
            print(f"📝 Loaded {len(queries)} test queries")
            return queries
            
        except FileNotFoundError:
            print(f"❌ Queries file {queries_file} not found")
            return []
        except Exception as e:
            print(f"❌ Error loading queries: {e}")
            return []
    
    def load_qrels(self, qrels_file="qrels.jsonl"):
        """Load query relevance judgments."""
        qrels = {}
        
        try:
            with open(qrels_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        qrel = json.loads(line)
                        # Convert to expected format
                        query_id = qrel['query-id']
                        formatted_qrel = {
                            'query_id': query_id,
                            'doc_id': qrel['corpus-id'],
                            'relevance': qrel.get('score', 1),  # Use score as relevance
                            'label': 'relevant' if qrel.get('score', 1) > 0 else 'not_relevant'
                        }
                        
                        if query_id not in qrels:
                            qrels[query_id] = []
                        qrels[query_id].append(formatted_qrel)
                        
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️ Skipping qrel line {line_num}: Invalid JSON - {e}")
                        continue
                    except KeyError as e:
                        print(f"  ⚠️ Skipping qrel line {line_num}: Missing field {e}")
                        continue
                    
            print(f"📋 Loaded relevance judgments for {len(qrels)} queries")
            return qrels
            
        except FileNotFoundError:
            print(f"⚠️ Qrels file {qrels_file} not found - will skip evaluation")
            return {}
        except Exception as e:
            print(f"❌ Error loading qrels: {e}")
            return {}
    
    def run_query(self, query_data, qrels=None):
        """Run a single query and display real-time results."""
        query_id = query_data['query_id']
        query_text = query_data['query']
        
        print(f"\n🔍 QUERY {query_id}: {query_text}")
        print("=" * 60)
        
        # Measure retrieval time
        start_time = time.time()
        results = self.frag.retrieve(query_text, top_k=5)
        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms
        
        print(f"⚡ Retrieval time: {retrieval_time:.2f}ms")
        print(f"📊 Found {len(results)} results\\n")
        
        if not results:
            print("❌ No results found")
            return
        
        # Display results with relevance checking
        for i, (chunk_id, score, content, metadata) in enumerate(results, 1):
            doc_id = metadata.get('doc_id', 'unknown')
            
            # Check relevance if qrels available
            relevance_info = ""
            if qrels and query_id in qrels:
                for qrel in qrels[query_id]:
                    if qrel['doc_id'] == doc_id:
                        rel_score = qrel['relevance']
                        rel_label = qrel['label']
                        if rel_score == 2:
                            relevance_info = f" 🎯 HIGHLY_RELEVANT"
                        elif rel_score == 1:
                            relevance_info = f" ✅ RELEVANT"
                        else:
                            relevance_info = f" ❌ NOT_RELEVANT"
                        break
            
            print(f"📄 RESULT {i} (Score: {score:.4f}, Doc: {doc_id}){relevance_info}")
            
            # Show content preview
            content_preview = content[:200] + "..." if len(content) > 200 else content
            print(f"   {content_preview}")
            print()
        
        # Store result for analysis
        self.results.append({
            'query_id': query_id,
            'query': query_text,
            'retrieval_time_ms': retrieval_time,
            'num_results': len(results),
            'results': [(chunk_id, score, metadata.get('doc_id', 'unknown')) 
                       for chunk_id, score, content, metadata in results]
        })
    
    def calculate_metrics(self, qrels):
        """Calculate retrieval metrics."""
        if not qrels or not self.results:
            print("⚠️ Cannot calculate metrics - missing qrels or results")
            return
        
        print("\\n📊 EVALUATION METRICS")
        print("=" * 40)
        
        total_precision = 0
        total_recall = 0
        total_queries = 0
        
        for result in self.results:
            query_id = result['query_id']
            if query_id not in qrels:
                continue
                
            retrieved_docs = [doc_id for _, _, doc_id in result['results']]
            relevant_docs = [qrel['doc_id'] for qrel in qrels[query_id] if qrel['relevance'] > 0]
            
            if not retrieved_docs:
                continue
                
            # Calculate precision and recall
            relevant_retrieved = set(retrieved_docs) & set(relevant_docs)
            precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
            recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
            
            total_precision += precision
            total_recall += recall
            total_queries += 1
            
            print(f"Query {query_id}: P={precision:.3f}, R={recall:.3f}")
        
        if total_queries > 0:
            avg_precision = total_precision / total_queries
            avg_recall = total_recall / total_queries
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            print(f"\\n🎯 AVERAGE METRICS:")
            print(f"   Precision: {avg_precision:.3f}")
            print(f"   Recall: {avg_recall:.3f}")
            print(f"   F1-Score: {f1_score:.3f}")
    
    def run_test_suite(self):
        """Run complete test suite with real-time display."""
        print("🧪 FRAG REAL-TIME TEST SUITE")
        print("🕐 Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 60)
        
        # Clear previous data
        self.frag.clear_db()
        
        # Load data
        if not self.load_corpus():
            return
        
        queries = self.load_queries()
        if not queries:
            return
            
        qrels = self.load_qrels()
        
        # Show system stats
        stats = self.frag.get_stats()
        print(f"\\n📈 System Stats: {stats['total_chunks']} chunks in {stats['storage_type']}")
        
        # Run queries with real-time display
        print("\\n🚀 RUNNING QUERIES...")
        for query_data in queries:
            self.run_query(query_data, qrels)
            time.sleep(0.5)  # Brief pause for readability
        
        # Calculate and display metrics
        self.calculate_metrics(qrels)
        
        # Final summary
        avg_time = sum(r['retrieval_time_ms'] for r in self.results) / len(self.results) if self.results else 0
        print(f"\\n⏱️  Average retrieval time: {avg_time:.2f}ms")
        print("🏁 Test suite completed!")

def main():
    """Main test runner."""
    tester = FragTester()
    
    try:
        tester.run_test_suite()
    except KeyboardInterrupt:
        print("\\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\\n❌ Test error: {e}")
    finally:
        tester.frag.close()

if __name__ == "__main__":
    main()