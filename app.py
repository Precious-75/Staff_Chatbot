from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response
import requests
import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv
from handbook_rag import init_rag, get_rag_context
from urllib.parse import quote

load_dotenv('.env')
API_KEY = os.getenv('API_KEY')
API_URL = "https://api.groq.com/openai/v1/chat/completions"

app = Flask(__name__, template_folder='C:/Staff_Chatbot/static') 
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'

HANDBOOK_PDF_PATH = "HandbookQA.pdf"
if os.path.exists(HANDBOOK_PDF_PATH):
    init_rag(HANDBOOK_PDF_PATH)

CSV_CONFIDENCE_THRESHOLD = 0.30
RAG_CONFIDENCE_THRESHOLD = 0.30

# future refernce-Add emails here with keywords that will trigger them
CONTACT_EMAILS = {
    'hr': {
        'email': 'request.lekki@greenspringsschool.com',
        'subject': 'HR Request',
        'keywords': ['hr', 'human resources', 'request', 'staff request', 'personnel']
    }
}

def should_add_contact_link(message, response):
    message_lower = message.lower()
    response_lower = response.lower() if response else ""
    
    triggers = ['more information', 'contact', 'email', 'request', 'how do i', 'reach out']
    
    for trigger in triggers:
        if trigger in message_lower or trigger in response_lower:
            return True
    return False

def get_relevant_contact(message):
    message_lower = message.lower()
    
    for contact_type, contact_info in CONTACT_EMAILS.items():
        for keyword in contact_info['keywords']:
            if keyword in message_lower:
                return contact_type, contact_info
    
    return 'hr', CONTACT_EMAILS['hr']

def add_contact_link(response, message):
    if not should_add_contact_link(message, response):
        return response
    
    contact_type, contact_info = get_relevant_contact(message)
    email = contact_info['email']
    subject = contact_info['subject']
    
    # Create Gmail web link only
    gmail_link = f"https://mail.google.com/mail/?view=cm&fs=1&to={email}&su={quote(subject)}"
    
    # Simple contact section without border
    contact_section = f'''

<div style="margin-top: 10px;">
    <a href="{gmail_link}" target="_blank" style="color: #1a73e8; text-decoration: none; font-weight: 500;">📧 Open in Gmail</a>
</div>'''
    
    return response + contact_section

def is_greeting(message):
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 
                 'good evening', 'greetings', 'howdy', 'sup', 'what\'s up']
    msg_lower = message.lower().strip()
    return any(greeting in msg_lower for greeting in greetings) and len(msg_lower.split()) <= 3

def get_groq_response(message, context=None, is_greeting=False):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    current_date = datetime.now().strftime("%B %Y")
    
    if is_greeting:
        system_content = """You are Greeny G, a friendly support assistant for Greensprings School.
Respond to greetings warmly and briefly introduce yourself as Greeny G. Keep it to 1-2 sentences.
Example: "Hello! I'm Greeny G, your Greensprings School support assistant. How can I help you today?" """
        
    elif context:
        system_content = f"""You are Greeny G, a support assistant for Greensprings School.
Current date: {current_date}

HANDBOOK CONTEXT:
{context}

Answer ONLY what was asked in under 50 words. Be direct and concise."""
    else:
        system_content = f"""You are Greeny G, a support assistant for Greensprings School.
Current date: {current_date}

Answer in 1-2 sentences maximum (under 50 words). Be direct and helpful."""
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": message}
        ],
        "temperature": 0.3,
        "max_tokens": 100,
        "top_p": 0.8
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        return None
    except:
        return None

def is_weak_response(response):
    if not response or len(response.strip()) < 10:
        return True
    
    weak_phrases = ['i do not understand', 'i don\'t understand', 'i don\'t know',
                    'not sure', 'i can\'t help', 'unclear']
    
    return any(phrase in response.lower() for phrase in weak_phrases)

def get_smart_response(user_message):
    base_response = None
    
    if is_greeting(user_message):
        base_response = get_groq_response(user_message, is_greeting=True)
    
    if not base_response:
        try:
            result = get_response(user_message)
            if isinstance(result, tuple):
                csv_response, csv_confidence = result
                if csv_response and csv_confidence >= CSV_CONFIDENCE_THRESHOLD and not is_weak_response(csv_response):
                    base_response = csv_response
        except:
            pass
    
    if not base_response:
        try:
            handbook_context, rag_confidence, handbook_pages = get_rag_context(user_message)
            if handbook_context and len(handbook_context.strip()) > 10:
                groq_response = get_groq_response(user_message, context=handbook_context)
                if groq_response and not is_weak_response(groq_response):
                    base_response = groq_response
        except:
            pass
    
    if not base_response:
        groq_response = get_groq_response(user_message)
        if groq_response and not is_weak_response(groq_response):
            base_response = groq_response
    
    if not base_response:
        base_response = "I'm sorry, I couldn't find a specific answer. Please contact Greensprings School support."
    
    return add_contact_link(base_response, user_message)

@app.route("/", methods=["GET"])
def index_get():
    return render_template("base.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id", "guest")
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"reply": "Please enter a message."})
        
        bot_response = get_smart_response(user_message)
        
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_id, message, response)
            VALUES (?, ?, ?)
        ''', (user_id, user_message, bot_response))
        conn.commit()
        conn.close()
        
        return jsonify({"reply": bot_response})
        
    except Exception as e:
        return jsonify({"reply": "Sorry, I'm experiencing technical difficulties."})

@app.route("/chat/<user_id>", methods=["GET"])
def get_history(user_id):
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
        
        history = [{"message": row[0], "response": row[1], "timestamp": row[2]} for row in rows]
        return jsonify({"history": history})
        
    except:
        return jsonify({"error": "Could not fetch history"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("message", "").strip()
        
        if not text:
            return jsonify({"answer": "Please enter a message."})
        
        response = get_smart_response(text)
        return jsonify({"answer": response})
        
    except:
        return jsonify({"answer": "Sorry, I'm experiencing technical difficulties."})

@app.route("/test-groq", methods=["GET"])
def test_groq():
    test_response = get_groq_response("What year is it now?")
    return f"Groq: {test_response if test_response else 'Not working'}"

@app.route("/test-csv")
def test_csv():
    test_response = get_response("student login")
    return f"CSV: {test_response}"

@app.route("/test-rag")
def test_rag():
    test_question = "Can PE staff wear their sportswear throughout the day?"
    context, confidence, pages = get_rag_context(test_question)
    
    if context:
        concise_response = get_groq_response(test_question, context=context)
        return f"RAG working. Confidence: {confidence:.2%}, Response: {concise_response}"
    return "RAG found nothing"

@app.route("/test-all")
def test_all():
    results = []
    
    try:
        get_response("test")
        results.append("CSV: Working")
    except:
        results.append("CSV: Failed")
    
    try:
        context, conf, pages = get_rag_context("test")
        results.append(f"RAG: Working ({conf:.2%})")
    except:
        results.append("RAG: Failed")
    
    try:
        groq_test = get_groq_response("test")
        results.append("Groq: Working" if groq_test else "Groq: No response")
    except:
        results.append("Groq: Failed")
    
    try:
        test_response = get_smart_response("How do I contact HR?")
        results.append("Contact Links: Working" if 'mailto:' in test_response else "Contact Links: Not triggered")
    except:
        results.append("Contact Links: Failed")
    
    return "<br>".join(results)

if __name__ == "__main__":
    app.run(port=5000, debug=True)