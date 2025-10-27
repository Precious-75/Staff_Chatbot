import os
import csv
from difflib import SequenceMatcher
import re
import requests 
from dotenv import load_dotenv

load_dotenv('.env')
API_KEY = os.getenv('API_KEY')
API_URL = "https://api.groq.com/openai/v1/chat/completions"

bot_name = "Greeny G"


def load_csv_qa(csv_file_path):
    """
    Load Q&A pairs from CSV file
    Expected CSV format: Question,Answer,Category,Access
    """
    qa_pairs = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            for row_num, row in enumerate(csv_reader, 1):
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    question = str(row[0]).strip()
                    answer = str(row[1]).strip()
                    
                    # Skip header row
                    if row_num == 1 and question.lower() in ['question', 'q', 'questions']:
                        continue
                    
                    qa_pairs.append({
                        'question': question.lower(),
                        'answer': answer,
                        'keywords': extract_keywords(question)
                    })
                    
        print(f" Loaded {len(qa_pairs)} Q&A pairs from CSV")
        
    except FileNotFoundError:
        print(f" CSV file not found: {csv_file_path}")
    except Exception as e:
        print(f" Error loading CSV: {e}")
        
    return qa_pairs

def extract_keywords(text):
    """Extract important keywords from a question for better matching"""
    stop_words = {'what', 'how', 'where', 'when', 'why', 'who', 'is', 'are', 'can', 'do', 'does', 
                  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i', 'my'}
    
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]
    
    return keywords

# Load your CSV file
csv_qa_pairs = load_csv_qa('school_it_qa.csv')

def find_csv_answer(user_input, threshold=0.5):
    """
    Enhanced CSV matching with multiple strategies
    Returns: (answer, confidence_score) or (None, 0)
    """
    user_input_clean = user_input.lower().strip()
    user_keywords = extract_keywords(user_input)
    
    best_match = None
    best_score = 0
    
    for qa_pair in csv_qa_pairs:
        # Strategy 1: Direct string similarity
        similarity = SequenceMatcher(None, user_input_clean, qa_pair['question']).ratio()
        
        # Strategy 2: Keyword matching score
        keyword_score = 0
        if user_keywords and qa_pair['keywords']:
            matching_keywords = set(user_keywords) & set(qa_pair['keywords'])
            keyword_score = len(matching_keywords) / max(len(user_keywords), len(qa_pair['keywords']))
        
        # Strategy 3: Partial phrase matching
        phrase_score = 0
        user_phrases = user_input_clean.split()
        qa_phrases = qa_pair['question'].split()
        
        for user_phrase in user_phrases:
            if len(user_phrase) > 3:
                for qa_phrase in qa_phrases:
                    if user_phrase in qa_phrase or qa_phrase in user_phrase:
                        phrase_score += 0.1
        
        # Combine scores with weights
        combined_score = (similarity * 0.4) + (keyword_score * 0.4) + (min(phrase_score, 1.0) * 0.2)
        
        if combined_score > 0.3:
            print(f"   Matching '{user_input[:50]}' with '{qa_pair['question'][:50]}'")
            print(f"   Combined score: {combined_score:.2f}")
        
        if combined_score > best_score and combined_score > threshold:
            best_score = combined_score
            best_match = qa_pair
    
    if best_match:
        print(f" Best CSV match (score: {best_score:.2f}): {best_match['question'][:80]}")
        return best_match['answer'], best_score
    else:
        print(f"  No CSV match found above threshold {threshold}")
        return None, 0

#groq Api call

def get_groq_response(message, context=None, is_greeting=False):
    """Get response from Groq API as Greeny G"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build system prompt based on context type
    if is_greeting:
        system_content = """You are Greeny G, a friendly support assistant for Greensprings School.
Respond to greetings warmly but briefly (1-2 sentences max).
Introduce yourself as Greeny G and offer to help.
Example: "Hi! I'm Greeny G, your Greensprings School assistant. How can I help you today?" """
        
    elif context:
        system_content = f"""You are Greeny G, a support assistant for Greensprings School.

Use the handbook information below to answer accurately and concisely.

HANDBOOK CONTEXT:
{context}

Instructions:
- Answer directly, no greetings
- Cite the handbook when relevant
- Be clear and professional
- Keep under 200 words
- If handbook doesn't fully answer, supplement with knowledge"""
    else:
        system_content = """You are Greeny G, a helpful support assistant for Greensprings School.
Answer questions clearly and concisely. Keep responses under 150 words.
Be helpful, professional, and accurate. If you don't know something, say so and suggest contacting support."""
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": message}
        ],
        "temperature": 0.7,
        "max_tokens": 250 if is_greeting else 300,
        "top_p": 0.9
    }
    
    try:
        print(f" Asking Greeny G: {message[:60]}...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            print(f" Greeny G responded ({len(answer)} chars)")
            return answer
        else:
            print(f" Groq error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f" Groq error: {e}")
        return None


def get_response(msg):
    """
    Simplified response function - only handles CSV matching
    Flow is managed by app.py:
    1. CSV matching (this function)
    2. Handbook RAG (app.py)
    3. Groq AI fallback (app.py)
    
    Returns: (response, confidence_score) or (None, 0)
    """
    print(f"\n Processing: '{msg}'")
    
    # Try to find answer in CSV
    csv_answer, csv_confidence = find_csv_answer(msg)
    
    if csv_answer and csv_confidence > 0:
        print(f" Using CSV response (confidence: {csv_confidence:.2%})")
        return csv_answer, csv_confidence
    
    print("  No CSV match found")
    return None, 0.0


def test_csv_matching():
    """Test the CSV matching with sample questions"""
    test_questions = [
        "How do I reset my wifi password?",
        "What is the student portal login?", 
        "Where can I find my grades?",
        "How to access email?",
        "printer problems"
    ]
    
    print(" Testing CSV matching:")
    print("="*70)
    for question in test_questions:
        answer, confidence = find_csv_answer(question)
        print(f"\nQ: {question}")
        print(f"A: {answer[:100] + '...' if answer else 'No match found'}")
        print(f"Confidence: {confidence:.2%}")
        print("-" * 70)

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit, 'test' to run CSV tests)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        elif sentence == "test":
            test_csv_matching()
            continue

        resp, confidence = get_response(sentence)
        if resp:
            print(f"{bot_name}: {resp} (confidence: {confidence:.2%})")
        else:
            print(f"{bot_name}: No match found in CSV")