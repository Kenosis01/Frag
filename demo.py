from frag import Frag
import sys

def main():
    print("🔥 FRAG: Fingerprint Retrieval Augmented Generation")
    print("🔍 Elasticsearch-Powered Scalable Version")
    print("=" * 55)
    
    # Initialize FRAG system
    print("🔄 Connecting to Elasticsearch...")
    frag = Frag(top_features=8)
    
    if not frag.es:
        print("❌ Elasticsearch connection failed!")
        print("Please ensure Elasticsearch is running on localhost:9200")
        print("Installation: https://www.elastic.co/downloads/elasticsearch")
        return
    
    print("✅ Connected to Elasticsearch successfully!")
    
    print("📄 Add your text chunks (enter 'done' when finished):")
    print("-" * 55)
    
    chunk_count = 0
    while True:
        text = input(f"\n📝 Enter text chunk {chunk_count + 1} (or 'done'): ").strip()
        
        if text.lower() == 'done':
            break
            
        if text:
            chunk_id = frag.add_chunk(text, {'chunk_index': chunk_count})
            if chunk_id:
                chunk_count += 1
                print(f"✅ Added chunk {chunk_count} (ID: {chunk_id})")
    
    if chunk_count == 0:
        print("❌ No text chunks added. Exiting.")
        return
    
    
    # Show stats
    stats = frag.get_stats()
    if 'error' in stats:
        print(f"⚠️ Warning: {stats['error']}")
    else:
        print(f"\n📊 Index: {stats['total_chunks']} chunks in '{stats['index_name']}'")
        if 'index_size' in stats:
            size_mb = stats['index_size'] / (1024 * 1024)
            print(f"📏 Storage: {size_mb:.2f} MB ({stats['storage_type']})")
    print(f"🎯 Algorithm: Elasticsearch BM25 (Multi-field matching)")
    print("-" * 55)
    
    # Interactive query loop
    while True:
        query = input("\n🔍 Enter query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # Retrieve results
        results = frag.retrieve(query, top_k=3)
        
        if not results:
            print("❌ No results found")
            continue
        
        print(f"\n📋 Found {len(results)} results:")
        print("=" * 60)
        
        for i, (doc_id, score, content, metadata) in enumerate(results, 1):
            print(f"\n📄 RESULT {i} (ES Score: {score:.4f}, Doc ID: {doc_id})")
            
            # Show content
            display_text = content.strip()[:400]
            print(f"\n{display_text}")
            
            if len(content) > 400:
                print(f"... [showing first 400 of {len(content)} chars]")
            
            # Show metadata if available
            if metadata:
                print(f"Metadata: {metadata}")
            
            print("-" * 40)
    
    # Cleanup
    print("\n✅ Session ended. Data persisted in Elasticsearch.")

if __name__ == "__main__":
    main()