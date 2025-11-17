import os
import ssl
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil

# Global variables
vector_db = None
retriever = None
chunks = []
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embedding_model():
    """
    Get embeddings model with automatic SSL handling
    Tries normal download first, then disables SSL verification if needed
    """
    print(" Loading embeddings model...")
    
    try:
        # Try normal download first
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(" Model loaded successfully")
        return embeddings
        
    except Exception as e:
        error_msg = str(e)
        
        # Check if it's an SSL error
        if "SSL" in error_msg or "certificate" in error_msg.lower():
            print(" SSL certificate error detected")
            print(" Temporarily disabling SSL verification for model download...")
            
            # Disable SSL verification temporarily
            try:
                # Save original context
                original_context = ssl._create_default_https_context
                
                # Disable SSL verification
                ssl._create_default_https_context = ssl._create_unverified_context
                
                # Set environment variables as backup
                os.environ['CURL_CA_BUNDLE'] = ''
                os.environ['REQUESTS_CA_BUNDLE'] = ''
                
                # Try loading model again
                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                
                # Restore original SSL context
                ssl._create_default_https_context = original_context
                
                # Clean up environment variables
                os.environ.pop('CURL_CA_BUNDLE', None)
                os.environ.pop('REQUESTS_CA_BUNDLE', None)
                
                print(" Model loaded successfully (SSL verification bypassed)")
                return embeddings
                
            except Exception as e2:
                print(f" ERROR: Failed to load model even with SSL disabled: {e2}")
                return None
        else:
            print(f" ERROR: Failed to load embeddings model: {e}")
            return None


def init_rag(pdf_path):
    """Initialize RAG system with proper persistence handling"""
    global vector_db, retriever, chunks
    
    print("\n" + "="*60)
    print(" RAG INITIALIZATION DEBUG")
    print("="*60)
    
    pdf_path = Path(pdf_path)
    
    # Step 1: Verify PDF file exists
    print(f"\n File Path: {pdf_path}")
    print(f" Absolute Path: {pdf_path.absolute()}")
    print(f" File Exists: {pdf_path.exists()}")
    
    if not pdf_path.exists():
        print(f" ERROR: PDF not found at {pdf_path.absolute()}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Files in current directory:")
        for f in os.listdir('.'):
            print(f"     - {f}")
        return False
    
    # Step 2: Check file size
    file_size = pdf_path.stat().st_size
    print(f" File Size: {file_size / 1024:.2f} KB")
    
    if file_size == 0:
        print(" ERROR: PDF file is empty (0 bytes)")
        return False
    
    # Step 3: Load embeddings model with SSL handling
    embeddings = get_embedding_model()
    
    if embeddings is None:
        print(" ERROR: Could not load embeddings model")
        return False
    
    # Step 4: Check if vector store already exists
    print(f"\n Checking for existing vector store...")
    if os.path.exists(PERSIST_DIR):
        print(f" Vector store already exists at {PERSIST_DIR}")
        print(f"   Loading from disk (faster than rebuilding)...")
        
        try:
            vector_db = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings,
                collection_name="handbook"
            )
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            
            # Test if it actually works
            test_results = vector_db.similarity_search("test", k=1)
            
            if len(test_results) > 0:
                chunks = [1] * 190  # Dummy list to indicate chunks exist
                
                print(f" Vector store loaded successfully")
                print(f" Contains indexed documents")
                print("\n" + "="*60)
                print(f" RAG INITIALIZED (FROM CACHE)")
                print("="*60 + "\n")
                return True
            else:
                print(f" Vector store exists but appears empty")
                print(f"   Will rebuild from PDF...")
                shutil.rmtree(PERSIST_DIR)
            
        except Exception as e:
            print(f" Could not load cached vector store: {e}")
            print(f" Will rebuild from PDF...")
            if os.path.exists(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR)
    
    # Step 5: Load PDF from scratch
    print(f"\n Step 1: Loading PDF...")
    try:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        print(f" Loaded {len(documents)} pages")
        
        if len(documents) == 0:
            print(" ERROR: PDF loaded but contains 0 pages")
            print("   This might be an image-only PDF or encrypted PDF")
            return False
        
        first_page = documents[0].page_content[:200]
        print(f" First page preview: {first_page}...")
        
    except Exception as e:
        print(f" ERROR loading PDF: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Split into chunks
    print(f"\n Step 2: Splitting into chunks...")
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f" Created {len(chunks)} chunks")
        
        if len(chunks) == 0:
            print(" ERROR: No chunks created after splitting")
            return False
        
        chunk_sizes = [len(c.page_content) for c in chunks]
        print(f"   Min chunk size: {min(chunk_sizes)} chars")
        print(f"   Max chunk size: {max(chunk_sizes)} chars")
        print(f"   Avg chunk size: {sum(chunk_sizes)/len(chunk_sizes):.0f} chars")
        
    except Exception as e:
        print(f" ERROR splitting chunks: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 7: Create vector store
    print(f"\n Step 3: Creating vector store...")
    try:
        print(f"   Creating Chroma vector store...")
        print(f"   This may take a few minutes...")
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name="handbook"
        )
        print(f" Vector store created and persisted")
        
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        print(f" Retriever initialized")
        
    except Exception as e:
        print(f" ERROR creating vector store: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print(f" RAG INITIALIZED SUCCESSFULLY")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Persisted to: {PERSIST_DIR}")
    print("="*60 + "\n")
    
    return True


def get_rag_context(query):
    """
    Get context from handbook RAG
    Returns: (context_text, confidence_score, page_numbers)
    
    Chroma returns DISTANCE scores (lower is better)
    We convert to similarity scores (higher is better)
    """
    global vector_db, chunks
    
    print(f"\n RAG Query: '{query}'")
    
    if vector_db is None:
        print(" RAG not initialized - vector_db is None")
        return None, 0.0, []
    
    if len(chunks) == 0:
        print(" RAG has 0 chunks")
        return None, 0.0, []
    
    try:
        # Get results with scores
        results = vector_db.similarity_search_with_score(query, k=3)
        
        print(f" Found {len(results)} results")
        
        if not results:
            print(" No results returned from vector search")
            return None, 0.0, []
        
        context_parts = []
        pages = set()
        scores = []
        
        print(f"\n Results breakdown:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"   Result {i}:")
            print(f"     - Distance score: {score:.4f} (lower is better)")
            print(f"     - Content length: {len(doc.page_content)} chars")
            print(f"     - Preview: {doc.page_content[:100]}...")
            
            context_parts.append(doc.page_content)
            scores.append(score)
            
            # Extract page number
            if "page" in doc.metadata:
                page_num = doc.metadata["page"]
                pages.add(page_num)
                print(f"     - Page: {page_num}")
        
        # Combine all context
        context = "\n\n".join(context_parts)
        
        # Convert Chroma distance to similarity score
        # Chroma uses L2 distance, lower scores = more similar
        # Typical range: 0.3 (very similar) to 2.0+ (not similar)
        
        avg_distance = sum(scores) / len(scores)
        
        # Convert distance to similarity (0 to 1 scale, higher is better)
        avg_similarity = 1.0 / (1.0 + avg_distance)
        
        print(f"\n Score Analysis:")
        print(f"   Raw scores: {[f'{s:.4f}' for s in scores]}")
        print(f"   Avg distance: {avg_distance:.4f}")
        print(f"   Converted similarity: {avg_similarity:.4f} ({avg_similarity:.2%})")
        
        # Quality check
        if avg_distance > 1.5:
            print(f" WARNING: High distance score suggests poor match")
        elif avg_distance < 0.5:
            print(f" Excellent match")
        else:
            print(f" Good match")
        
        return context, avg_similarity, sorted(list(pages))
        
    except Exception as e:
        print(f" RAG retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, []


def is_handbook_question(question):
    """Check if question is suitable for handbook lookup"""
    handbook_keywords = [
        'handbook', 'policy', 'procedure', 'rule', 'regulation',
        'staff', 'employee', 'uniform', 'dress code', 'conduct',
        'working hours', 'leave', 'attire', 'behavior', 'guidelines'
    ]
    
    question_lower = question.lower()
    matches = [kw for kw in handbook_keywords if kw in question_lower]
    
    if matches:
        print(f" Handbook keywords detected: {matches}")
        return True
    
    return False


def test_rag_search(query):
    """Test function to check RAG performance"""
    print(f"\n{'='*70}")
    print(f" TESTING RAG SEARCH")
    print(f"{'='*70}")
    print(f"Query: {query}")
    
    context, confidence, pages = get_rag_context(query)
    
    print(f"\n Results:")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Pages: {pages}")
    print(f"   Context length: {len(context) if context else 0} chars")
    
    if context:
        print(f"\n Context preview:")
        print(context[:500])
        print("...")
    else:
        print("\n No context returned")
    
    print(f"\n{'='*70}\n")
    
    return context, confidence, pages


def reset_vector_store():
    """Delete existing vector store to force rebuild"""
    if os.path.exists(PERSIST_DIR):
        print(f"Deleting vector store at {PERSIST_DIR}...")
        shutil.rmtree(PERSIST_DIR)
        print("Vector store deleted. Will rebuild on next init_rag() call.")
    else:
        print("No vector store found to delete.")