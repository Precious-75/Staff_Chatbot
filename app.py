from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response  # Your existing chat system
import requests
import json
import re

app = Flask(__name__)
CORS(app)

def get_llama_response(message):
    """Get response from Llama for general knowledge questions"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama2:7b-chat",
        "prompt": f"""You are a helpful assistant. Provide clear, concise, and accurate responses. 
        Keep responses under 150 words and be professional.

        User: {message}
        Assistant:""",
        "stream": False,
        "options": {
            "temperature": 0.7,
            "max_tokens": 200,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
    }
    
    try:
        print(f"🦙 Asking Llama: {message}")
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '').strip()
            print(f"🦙 Llama responded: {answer[:100]}...")
            return answer
        else:
            print(f"❌ Llama error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("⚠️ Ollama is not running. Start it with: ollama serve")
        return None
    except Exception as e:
        print(f"❌ Llama error: {e}")
        return None

def is_specific_question(message):
    """
    Detect if this might be a specific question that your CSV should handle
    Add keywords related to your specific domain here
    """
    # Add keywords that indicate domain-specific questions
    specific_keywords = [
        'school', 'student', 'teacher', 'class', 'grade', 'homework',
        'assignment', 'exam', 'test', 'course', 'subject', 'schedule',
        'login', 'password', 'account', 'system', 'portal', 'access',
        'it support', 'computer', 'network', 'wifi', 'printer',
        'registration', 'enrollment', 'fee', 'payment', 'library',
        'cafeteria', 'campus', 'dormitory', 'parking'
        # Add more keywords specific to your domain
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in specific_keywords)

def is_general_question(message):
    """
    Detect if this is a general knowledge question that Llama should handle
    """
    general_patterns = [
        r'\bwhat is\b', r'\bwho is\b', r'\bwhen did\b', r'\bwhere is\b',
        r'\bhow to\b', r'\bwhy does\b', r'\bexplain\b', r'\btell me about\b',
        r'\bdefine\b', r'\bdefinition\b', r'\bhistory of\b', r'\bfacts about\b'
    ]
    
    general_keywords = [
        'science', 'math', 'history', 'geography', 'biology', 'chemistry',
        'physics', 'literature', 'art', 'music', 'sports', 'news',
        'weather', 'recipe', 'cooking', 'health', 'medicine', 'technology',
        'programming', 'computer science', 'philosophy', 'psychology'
    ]
    
    message_lower = message.lower()
    
    # Check patterns
    if any(re.search(pattern, message_lower) for pattern in general_patterns):
        return True
    
    # Check keywords
    if any(keyword in message_lower for keyword in general_keywords):
        return True
    
    return False

def is_weak_response(response):
    """Check if the response from your existing system is weak/unhelpful"""
    if not response or len(response.strip()) < 5:
        return True
    
    # Common phrases that indicate your system doesn't know the answer
    weak_phrases = [
        'i do not understand',
        'i don\'t understand',
        'i don\'t know',
        'sorry',
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
    Enhanced smart routing:
    1. Check if it's clearly a specific question -> use your system
    2. Check if it's clearly a general question -> use Llama first
    3. For unclear questions, try your system first, then Llama if weak
    """
    
    print(f"\n👤 User: {user_message}")
    
    # Route 1: Clearly specific questions - use your system first
    if is_specific_question(user_message):
        print("🎯 Detected specific question - trying your system first")
        try:
            existing_response = get_response(user_message)
            print(f"📋 Your system: {existing_response}")
            
            if existing_response and not is_weak_response(existing_response):
                print("✅ Using your system's response")
                return existing_response
            else:
                print("⚠️ Your system gave weak response, trying Llama backup")
                llama_response = get_llama_response(user_message)
                if llama_response:
                    return llama_response
                else:
                    return existing_response or "I'm having trouble with that question."
                    
        except Exception as e:
            print(f"❌ Error with your system: {e}")
    
    # Route 2: Clearly general questions - use Llama first
    elif is_general_question(user_message):
        print("🌍 Detected general question - trying Llama first")
        llama_response = get_llama_response(user_message)
        
        if llama_response:
            print("✅ Using Llama's response")
            return llama_response
        else:
            print("⚠️ Llama failed, trying your system as backup")
            try:
                existing_response = get_response(user_message)
                return existing_response or "I'm having trouble with that question."
            except Exception as e:
                print(f"❌ Both systems failed: {e}")
                return "I'm having trouble with that question right now."
    
    # Route 3: Unclear questions - try your system first (it has CSV + neural net)
    else:
        print("❓ Unclear question type - trying your system first")
        try:
            existing_response = get_response(user_message)
            print(f"📋 Your system: {existing_response}")
            
            if existing_response and not is_weak_response(existing_response):
                print("✅ Using your system's response")
                return existing_response
            else:
                print("⚠️ Your system gave weak response, trying Llama")
                llama_response = get_llama_response(user_message)
                if llama_response:
                    print("✅ Using Llama's response")
                    return llama_response
                else:
                    print("⚠️ Both systems struggled")
                    return existing_response or "I'm having trouble understanding your question. Could you please rephrase it?"
                    
        except Exception as e:
            print(f"❌ Error with your system: {e}")
            # Try Llama as last resort
            llama_response = get_llama_response(user_message)
            return llama_response or "I'm having technical difficulties. Please try again."

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    try:
        data = request.get_json()
        text = data.get("message", "").strip()
        
        if not text:
            return jsonify({"answer": "Please enter a message."})
        
        # Get smart response with enhanced routing
        response = get_smart_response(text)
        
        return jsonify({"answer": response})
        
    except Exception as e:
        print(f"❌ Error in predict: {e}")
        return jsonify({"answer": "Sorry, I'm having technical difficulties."})

@app.route("/test-llama")
def test_llama():
    """Test endpoint to check if Llama is working"""
    test_response = get_llama_response("What is 2+2?")
    if test_response:
        return f"✅ Llama is working! Response: {test_response}"
    else:
        return "❌ Llama is not working. Make sure 'ollama serve' is running and llama2:7b-chat is downloaded."

@app.route("/test-csv")
def test_csv():
    """Test endpoint to check if your CSV system is working"""
    test_response = get_response("student login")  # Use a question likely in your CSV
    return f"📋 Your system response: {test_response}"

if __name__ == "__main__":
    print("🚀 Starting enhanced chatbot...")
    print("📋 CSV + Neural Network system for specific questions")
    print("🦙 Llama integration for general knowledge")
    print("🎯 Smart routing based on question type")
    print("🧪 Test endpoints:")
    print("   - http://localhost:5000/test-llama")
    print("   - http://localhost:5000/test-csv")
    app.run(port=5000, debug=True)