import os
import random
import json
import csv
from difflib import SequenceMatcher
import re
import requests 
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from dotenv import load_dotenv

load_dotenv('.env')
API_KEY = os.getenv('API_KEY')
API_URL = "https://api.groq.com/openai/v1/chat/completions"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Support Desk"

# Enhanced CSV loading with better preprocessing
def load_csv_qa(csv_file_path):
    """
    Load Q&A pairs from CSV file with enhanced preprocessing
    Expected CSV format: Question,Answer,Category,Access
    """
    qa_pairs = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            for row_num, row in enumerate(csv_reader, 1):
                if len(row) >= 2 and row[0].strip() and row[1].strip():  # Skip empty rows
                    question = str(row[0]).strip()
                    answer = str(row[1]).strip()
                    
            
                    if row_num == 1 and question.lower() in ['question', 'q', 'questions']:
                        continue
                    
                    qa_pairs.append({
                        'question': question.lower(),
                        'answer': answer,
                        'keywords': extract_keywords(question)  # Extract key terms for better matching
                    })
                    
        print(f" Loaded {len(qa_pairs)} Q&A pairs from CSV")
        
    except FileNotFoundError:
        print(f" CSV file not found: {csv_file_path}")
    except Exception as e:
        print(f" Error loading CSV: {e}")
        
    return qa_pairs

def extract_keywords(text):
    """Extract important keywords from a question for better matching"""
    # Remove common words and extract meaningful terms
    stop_words = {'what', 'how', 'where', 'when', 'why', 'who', 'is', 'are', 'can', 'do', 'does', 
                  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i', 'my'}
    
    # Clean and split text
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
            if len(user_phrase) > 3:  # Only check meaningful phrases
                for qa_phrase in qa_phrases:
                    if user_phrase in qa_phrase or qa_phrase in user_phrase:
                        phrase_score += 0.1
        
        # Combine scores with weights
        combined_score = (similarity * 0.4) + (keyword_score * 0.4) + (min(phrase_score, 1.0) * 0.2)
        
        # Debug output for top matches
        if combined_score > 0.3:
            print(f" Matching '{user_input[:50]}' with '{qa_pair['question'][:50]}'")
            print(f"   Similarity: {similarity:.2f}, Keywords: {keyword_score:.2f}, Phrases: {phrase_score:.2f}")
            print(f"   Combined score: {combined_score:.2f}")
        
        if combined_score > best_score and combined_score > threshold:
            best_score = combined_score
            best_match = qa_pair
    
    if best_match:
        print(f"Best CSV match (score: {best_score:.2f}): {best_match['question'][:100]}")
        return best_match['answer'], best_score
    else:
        print(f" No CSV match found above threshold {threshold}")
        return None, 0

def get_neural_response(msg):
    """
    Get response from the trained neural network model
    Returns: (response, confidence) or (None, 0)
    """
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    confidence = prob.item()
    print(f" Neural network confidence: {confidence:.2f} for tag '{tag}'")
    
    # LOWERED THRESHOLD - Only use neural network if confident
    if confidence > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f" Neural network response: {response[:100]}...")
                return response, confidence
    
    print(" Neural network confidence too low")
    return None, confidence

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

def get_response(msg):
    """
    Main response function with enhanced logic:
    1. Try CSV matching first (for specific domain questions)
    2. Fall back to neural network if no CSV match
    3. Try Groq AI for general questions
    4. Return default message if all fail
    
    Returns: (response, confidence_score)
    """
    print(f"\n Processing: '{msg}'")
    
    # First, try to find answer in CSV with enhanced matching
    csv_answer, csv_confidence = find_csv_answer(msg)
    if csv_answer:
        # Add contextual intro messages
        intro_messages = [
            "Based on our knowledge base: ",
            "Here's the information I found: ",
            "According to our records: ",
            "I found this in our database: ",
        ]
        intro = random.choice(intro_messages)
        full_response = f"{intro}{csv_answer}"
        print(f" Using CSV response: {full_response[:100]}...")
        return full_response, csv_confidence
    
    # If no CSV match, try the neural network model
    print(" No CSV match, trying neural network...")
    neural_response, neural_confidence = get_neural_response(msg)
    if neural_response:
        return neural_response, neural_confidence
    
    # If both fail, try Groq for general questions
    print(" No neural network match, trying Groq AI...")
    try:
        groq_response = get_groq_response(msg)
        if groq_response and len(groq_response.strip()) > 0:
            print(f" Using Groq response: {groq_response[:100]}...")
            return groq_response, 0.9  # High confidence for Groq
    except Exception as e:
        print(f" Groq error: {e}")
    
    # If everything fails, return a helpful default message with low confidence
    default_responses = [
        "I don't have specific information about that. Could you rephrase your question or provide more details?",
        "I'm not sure about that particular topic. Can you try asking in a different way?",
        "I don't understand that question. Could you please be more specific or try rephrasing it?",
        "I need more information to help you with that. Can you provide additional details?",
    ]
    
    default_response = random.choice(default_responses)
    print(f" Using default response: {default_response}")
    # Return low confidence for default responses so they can be overridden
    return default_response, 0.0

# Test function to help debug CSV matching
def test_csv_matching():
    """Test the CSV matching with sample questions"""
    test_questions = [
        "How do I reset my wifi password?",
        "What is the student portal login?", 
        "Where can I find my grades?",
        "How to access email?",
        "printer problems"
    ]
    
    print("🧪 Testing CSV matching:")
    for question in test_questions:
        answer, confidence = find_csv_answer(question)
        print(f"Q: {question}")
        print(f"A: {answer[:100] + '...' if answer else 'No match found'} (confidence: {confidence:.2f})")
        print("-" * 50)

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
        print(f"{bot_name}: {resp} (confidence: {confidence:.2f})")