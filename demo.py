from frag import Frag
import os
import sys

def main():
    print("🔥 FRAG: Fingerprint Retrieval Augmented Generation")
    print("=" * 55)
    
    # Get PDF file from user
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        pdf_file = input("Enter PDF file path (default: diffusion.pdf): ").strip()
        if not pdf_file:
            pdf_file = "diffusion.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"❌ File not found: {pdf_file}")
        return
    
    # Initialize FRAG system
    frag = Frag(top_features=8)
    
    # Process document
    print(f"📄 Processing {pdf_file}...")
    try:
        chunk_ids = frag.add_document(pdf_file)
        print(f"✅ Stored {len(chunk_ids)} chunks in database")
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return
    
    # Show stats
    stats = frag.get_stats()
    print(f"📊 Database: {stats['total_chunks']} chunks, {stats['unique_sources']} sources")
    print(f"🎯 Algorithm: BM25 Okapi")
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
        
        for i, (chunk_id, score, content, metadata) in enumerate(results, 1):
            print(f"\n📄 RESULT {i} (BM25 Score: {score:.4f}, ID: {chunk_id})")
            
            # Show content
            display_text = content.strip()[:500]
            print(f"\n{display_text}")
            
            if len(content) > 500:
                print(f"... [showing first 500 of {len(content)} chars]")
            
            print("-" * 40)
    
    # Cleanup
    frag.close()
    print("\n✅ Session ended. Database saved.")

if __name__ == "__main__":
    main()