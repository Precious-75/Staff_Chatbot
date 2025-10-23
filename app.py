from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response
import requests
import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv
from handbook_rag import init_rag, get_rag_context

load_dotenv('.env')
API_KEY = os.getenv('API_KEY')
API_URL = "https://api.groq.com/openai/v1/chat/completions"

app = Flask(__name__)
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'

from db import db, init_db
init_db(app)

# Initialize RAG system at startup
HANDBOOK_PDF_PATH = "HandbookQA.pdf"
if os.path.exists(HANDBOOK_PDF_PATH):
    print("📚 Initializing RAG handbook system...")
    init_rag(HANDBOOK_PDF_PATH)
    print("✅ RAG system ready!")
else:
    print(f"⚠️  Handbook not found at: {HANDBOOK_PDF_PATH}")

# CONFIDENCE THRESHOLDS
CSV_CONFIDENCE_THRESHOLD = 0.30
RAG_CONFIDENCE_THRESHOLD = 0.30

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_greeting(message):
    """Check if message is a greeting"""
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 
                 'good evening', 'greetings', 'howdy', 'sup', 'what\'s up']
    msg_lower = message.lower().strip()
    return any(greeting in msg_lower for greeting in greetings) and len(msg_lower.split()) <= 3

def get_groq_response(message, context=None, is_greeting=False):
    """Get response from Groq API as Greeny G - SHORT AND DIRECT"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    current_date = datetime.now().strftime("%B %Y")
    
    # Build system prompt based on context type
    if is_greeting:
        system_content = """You are Greeny G, a friendly support assistant for Greensprings School.
Respond to greetings warmly but briefly (1 sentence max).
Example: "Hi! I'm Greeny G. How can I help you today?" """
        
    elif context:
        system_content = f"""You are Greeny G, a support assistant for Greensprings School.
Current date: {current_date}

HANDBOOK CONTEXT:
{context}

CRITICAL INSTRUCTIONS:
- Answer ONLY what was asked - nothing extra
- Keep responses under 50 words
- Be direct and concise
- Use simple, natural language
- Do NOT add extra information not asked for
- Do NOT mention the handbook or sources
- Just answer the specific question asked

Example:
Question: "What time does school start?"
Good: "School starts at 8:00 AM."
Bad: "According to the handbook, school starts at 8:00 AM. This ensures students have adequate time to prepare for their classes and participate in morning activities."

ANSWER ONLY WHAT IS ASKED."""
    else:
        system_content = f"""You are Greeny G, a support assistant for Greensprings School.
Current date: {current_date}

Answer questions in 1-2 sentences maximum (under 50 words).
Be direct and helpful. Answer ONLY what was asked."""
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": message}
        ],
        "temperature": 0.3,  # Lower temperature for more focused responses
        "max_tokens": 100,  # Reduced from 250/300
        "top_p": 0.8
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            return answer
        else:
            print(f"🦙 Groq error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"🦙 Groq error: {e}")
        return None

def is_weak_response(response):
    """Check if response is weak/unhelpful"""
    if not response or len(response.strip()) < 10:
        return True
    
    weak_phrases = [
        'i do not understand', 'i don\'t understand', 'i don\'t know',
        'not sure', 'i can\'t help', 'unclear', 'i\'m not trained'
    ]
    
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in weak_phrases)

# ============================================================================
# SMART RESPONSE ROUTING
# ============================================================================

def get_smart_response(user_message):
    """
    Flow: Greetings → CSV → Handbook → Groq AI → Default
    """
    
    print(f"\n{'='*70}")
    print(f"👤 USER: {user_message}")
    print(f"{'='*70}")
    
    # ========================================================================
    # STEP 1: Handle Greetings
    # ========================================================================
    if is_greeting(user_message):
        print("\n👋 Detected greeting - using Greeny G")
        greeting_response = get_groq_response(user_message, is_greeting=True)
        if greeting_response:
            print("✅ Greeting handled by Greeny G")
            return greeting_response
    
    # ========================================================================
    # STEP 2: Check CSV Database (School QA)
    # ========================================================================
    print("\n🎯 STEP 2: Checking CSV database (School QA)...")
    try:
        result = get_response(user_message)
        
        if isinstance(result, tuple):
            csv_response, csv_confidence = result
            print(f"   Confidence: {csv_confidence:.2%}")
            
            if csv_response and csv_confidence >= CSV_CONFIDENCE_THRESHOLD and not is_weak_response(csv_response):
                print(f"✅ CSV HIGH CONFIDENCE ({csv_confidence:.2%}) - USING")
                return csv_response
            else:
                print(f"⚠️  CSV confidence too low or weak response: {csv_confidence:.2%}")
    
    except Exception as e:
        print(f"❌ CSV Error: {e}")
    
    # ========================================================================
    # STEP 3: Check Handbook (RAG) - ALWAYS CHECK!
    # ========================================================================
    print("\n📚 STEP 3: Checking Handbook (RAG)...")
    handbook_context = None
    handbook_pages = []
    
    try:
        handbook_context, rag_confidence, handbook_pages = get_rag_context(user_message)
        print(f"   Context length: {len(handbook_context) if handbook_context else 0} chars")
        print(f"   Confidence: {rag_confidence:.2%}")
        print(f"   Pages: {handbook_pages}")
        print(f"   Threshold: {RAG_CONFIDENCE_THRESHOLD:.2%}")
        
        # If we have ANY handbook context, use it with Greeny G
        if handbook_context and len(handbook_context.strip()) > 10:
            print(f"✅ Found handbook content - using Greeny G (concise mode)")
            groq_response = get_groq_response(user_message, context=handbook_context)
            
            if groq_response and not is_weak_response(groq_response):
                print("✅ SUCCESS - Greeny G with Handbook (concise)")
                return groq_response
        else:
            print(f"⚠️  No relevant handbook content found")
    
    except Exception as e:
        print(f"❌ RAG Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # STEP 4: Groq AI Fallback (General Knowledge)
    # ========================================================================
    print("\n🦙 STEP 4: Using Greeny G fallback (General Knowledge)...")
    groq_response = get_groq_response(user_message)
    
    if groq_response and not is_weak_response(groq_response):
        print("✅ SUCCESS - Greeny G (General Knowledge)")
        return groq_response
    
    # ========================================================================
    # STEP 5: Default Response
    # ========================================================================
    print("❌ All systems failed - returning default response")
    return ("I'm sorry, I couldn't find a specific answer to your question. "
            "Please try rephrasing or contact Greensprings School support for assistance.")

# ============================================================================
# ROUTES
# ============================================================================

@app.route("/", methods=["GET"])
def index_get():
    return render_template("base.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id", "guest")
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"reply": "Please enter a message."})
        
        print(f"\n📨 Received from {user_id}: {user_message}")
        
        # Get smart response
        bot_response = get_smart_response(user_message)
        
        # Save to database
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_id, message, response)
            VALUES (?, ?, ?)
        ''', (user_id, user_message, bot_response))
        conn.commit()
        conn.close()
        print(f"💾 Saved for {user_id}")
        
        return jsonify({"reply": bot_response})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"reply": "Sorry, I'm experiencing technical difficulties. Please try again."})

@app.route("/chat/<user_id>", methods=["GET"])
def get_history(user_id):
    """Get chat history for a user"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT message, response, timestamp 
            FROM conversations 
            WHERE user_id = ?
            ORDER BY timestamp DESC
        ''', (user_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = [
            {
                "message": row[0],
                "response": row[1],
                "timestamp": row[2]
            }
            for row in rows
        ]
        
        return jsonify({"history": history})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": "Could not fetch history"})

@app.route("/predict", methods=["POST"])
def predict():
    """Alternative chat endpoint"""
    try:
        data = request.get_json()
        text = data.get("message", "").strip()
        
        if not text:
            return jsonify({"answer": "Please enter a message."})
        
        response = get_smart_response(text)
        return jsonify({"answer": response})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"answer": "Sorry, I'm experiencing technical difficulties."})

# ============================================================================
# TEST ENDPOINTS
# ============================================================================

@app.route("/test-groq", methods=["GET"])
def test_groq():
    """Test Groq API"""
    test_response = get_groq_response("What year is it now?")
    if test_response:
        return f"✅ Greeny G working! Response: {test_response}"
    else:
        return "❌ Greeny G not working"

@app.route("/test-csv")
def test_csv():
    """Test CSV"""
    test_response = get_response("student login")
    return f"CSV: {test_response}"

@app.route("/test-rag")
def test_rag():
    """Test RAG with concise responses"""
    test_question = "Can PE staff wear their sportswear throughout the day?"
    context, confidence, pages = get_rag_context(test_question)
    
    if context:
        concise_response = get_groq_response(test_question, context=context)
        return f"""✅ RAG with Concise Responses working!
Test: {test_question}
Confidence: {confidence:.2%}
Pages: {pages}

Original Context (first 300 chars):
{context[:300]}...

Concise Response:
{concise_response}"""
    else:
        return f"⚠️  RAG found nothing for: {test_question}"

@app.route("/test-all")
def test_all():
    """Test all systems"""
    results = []
    
    # Test CSV
    try:
        csv_test = get_response("test")
        results.append("✅ CSV: Working")
    except Exception as e:
        results.append(f"❌ CSV: Failed - {e}")
    
    # Test RAG
    try:
        context, conf, pages = get_rag_context("test")
        results.append(f"✅ RAG: Working (confidence: {conf:.2%})")
    except Exception as e:
        results.append(f"❌ RAG: Failed - {e}")
    
    # Test Greeny G (Groq)
    try:
        groq_test = get_groq_response("test")
        if groq_test:
            results.append("✅ Greeny G (Groq): Working with Concise Responses")
        else:
            results.append("❌ Greeny G: No response")
    except Exception as e:
        results.append(f"❌ Greeny G: Failed - {e}")
    
    return "<br>".join(results)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 STARTING GREENY G CHATBOT - CONCISE MODE")
    print("="*70)
    print("\n📋 Response Flow:")
    print("   1️⃣  Greetings → Greeny G")
    print("   2️⃣  CSV Database (School QA)")
    print("   3️⃣  Handbook (RAG) → Concise answers by Greeny G")
    print("   4️⃣  Greeny G (Groq AI Fallback)")
    print("   5️⃣  Default Response")
    print("\n✨ NEW: All responses are now short and direct!")
    print("="*70 + "\n")
    
    app.run(port=5000, debug=True)