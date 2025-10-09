from weakref import ref
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response, load_csv_qa
import requests
import json
import re
import datetime
import os
from dotenv import load_dotenv

load_dotenv('.env')
API_KEY = os.getenv('API_KEY')
API_URL = "https://api.groq.com/openai/v1/chat/completions"

app = Flask(__name__)

# Configure database on the actual Flask app instance
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chatbot.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Apply CORS to the same app instance used for routes
CORS(app)

# SQLite Database (bind SQLAlchemy to this app)
from db import db, ChatHistory, init_db
from datetime import datetime

init_db(app)

# CONFIDENCE THRESHOLD
CONFIDENCE_THRESHOLD = 0.90

def get_groq_response(message):
    """Get response from Groq API"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide clear, concise, and accurate responses. Keep responses under 150 words and be professional."
            },
            {
                "role": "user", 
                "content": message
            }
        ],
        "temperature": 0.7,
        "max_tokens": 200,
        "top_p": 0.9
    }
    
    try:
        print(f"🦙 Asking Groq: {message}")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            print(f"🦙 Groq responded: {answer[:100]}...")
            return answer
        else:
            print(f"🦙 Groq error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("🦙 Connection error to Groq API")
        return None
    except Exception as e:
        print(f"🦙 Groq error: {e}")
        return None

def is_weak_response(response):
    """Check if the response from your existing system is weak/unhelpful"""
    if not response or len(response.strip()) < 5:
        return True
    
    weak_phrases = [
        'i do not understand',
        'i don\'t understand',
        'i don\'t know',
        'not sure',
        'i can\'t help',
        'i don\'t have',
        'unclear',
        'please explain',
        'can you please explain',
        'try again',
        'contact support',
        'i\'m not trained',
        'out of scope'
    ]
    
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in weak_phrases)

def get_smart_response(user_message):
    """
    Simple routing:
    1. Try your system first (JSON intents + CSV)
    2. If weak response -> Try Groq as fallback
    3. Return whichever worked
    """
    
    print(f"\n👤 User: {user_message}")
    
    # ALWAYS try your existing system first
    print("🎯 Trying your existing system (JSON intents + CSV)...")
    try:
        result = get_response(user_message)
        
        # Check if result is a tuple (response, confidence) or just a string
        if isinstance(result, tuple):
            existing_response, confidence = result
            print(f"📋 Your system: {existing_response}")
            print(f"🧠 Confidence: {confidence}")
            
            # If confidence is good, use your system's response
            if confidence >= CONFIDENCE_THRESHOLD:
                print("✅ Using your system's response (good confidence)")
                return existing_response
        else:
            # Old format - just a string response
            existing_response = result
            print(f"📋 Your system: {existing_response}")
        
        # Check if the response is strong/helpful
        if existing_response and not is_weak_response(existing_response):
            print("✅ Using your system's response")
            return existing_response
        
        # If weak response, try Groq as fallback
        print("⚠️ Your system gave weak response - trying Groq as fallback")
        groq_response = get_groq_response(user_message)
        
        if groq_response:
            print("✅ Using Groq's response")
            return groq_response
        else:
            print("⚠️ Groq also failed, using original response")
            return existing_response or "I'm not sure I understand. Could you please rephrase your question?"
            
    except Exception as e:
        print(f"❌ Error with your system: {e}")
        # Try Groq as fallback if system crashes
        groq_response = get_groq_response(user_message)
        if groq_response:
            return groq_response
        return "I'm having technical difficulties. Please try again."

# THE ROUTES

@app.route("/", methods=["GET"])
def index_get():
    return render_template("base.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint - handles user messages"""
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id", "guest")
        user_message = data.get("message", "")
        
        if not user_message:
            return jsonify({"reply": "Please enter a message."})
        
        print(f"\n Received message from user {user_id}: {user_message}")
        
        # Get smart response using routing logic
        bot_response = get_smart_response(user_message)
        
        # Save to database
        with app.app_context():
            new_chat = ChatHistory(
                user_id=user_id,
                user_message=user_message,
                bot_response=bot_response,
                timestamp=datetime.utcnow()
            )
            db.session.add(new_chat)
            db.session.commit()
            print(f"Saved to database: ID {new_chat.id}")
        
        return jsonify({"reply": bot_response})
        
    except Exception as e:
        print(f" Error in /chat endpoint: {e}")
        return jsonify({"reply": "Sorry, I'm having technical difficulties. Please try again."})

@app.route("/history", methods=["GET"])
def history():
    """Get chat history from database"""
    try:
        chats = ChatHistory.query.order_by(ChatHistory.timestamp.asc()).all()
        history_data = [
            {
                "id": chat.id,
                "user_id": chat.user_id,
                "user": chat.user_message,
                "bot": chat.bot_response,
                "timestamp": chat.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            for chat in chats
        ]
        print(f"Retrieved {len(history_data)} chat messages from database")
        return jsonify(history_data)
    except Exception as e:
        print(f" Error in /history endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Alternative chat endpoint (for compatibility)"""
    try:
        data = request.get_json(silent=True) or {}
        # Accept message from JSON body, form, or query as fallback
        text = (data.get("message")
                or request.form.get("message")
                or request.args.get("message", "")).strip()

        if not text:
            return jsonify({"answer": "Please enter a message."}), 200

        # Get smart response
        response = get_smart_response(text)

        return jsonify({"answer": response}), 200

    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        return jsonify({"answer": "Sorry, I'm having technical difficulties."}), 200

# THE ENDPOINTS FOR TESTING

@app.route("/test-groq", methods=["GET"])
def test_groq():
    """Test Groq API connection"""
    test_response = get_groq_response("What is 2+20?")
    if test_response:
        return f"Groq is working! Response: {test_response}"
    else:
        return "Groq is not working. Check your API key and connection."

@app.route("/test-csv")
def test_csv():
    """Test CSV/intents system"""
    test_response = get_response("student login")
    return f"Your system response: {test_response}"

@app.route("/test-db")
def test_db():
    """Test database connection"""
    try:
        # Add a test message
        test_chat = ChatHistory(
            user_id="test_user",
            user_message="Test message",
            bot_response="Test response",
            timestamp=datetime.utcnow()
        )
        db.session.add(test_chat)
        db.session.commit()
        
        # Count total messages
        count = ChatHistory.query.count()
        return f"Database is working! Total messages: {count}"
    except Exception as e:
        return f"Database error: {e}"

if __name__ == "__main__":
    print("Starting chatbot server...")
    print("Your system (JSON intents + CSV) tries first")
    print("Groq as fallback for weak responses")
    print("SQLite database for chat history")
    print("\nEndpoints:")
    print("   - POST http://localhost:5000/chat(main endpoint)")
    print("   - GET  http://localhost:5000/history (get chat history)")
    print("   - http://localhost:5000/test-groq")
    print("   - http://localhost:5000/test-db")
    app.run(port=5000, debug=True)