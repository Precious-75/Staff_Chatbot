import random
import json
import csv
from difflib import SequenceMatcher
import re

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

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
                    
                    # Skip header row if detected
                    if row_num == 1 and question.lower() in ['question', 'q', 'questions']:
                        continue
                    
                    qa_pairs.append({
                        'question': question.lower(),
                        'answer': answer,
                        'keywords': extract_keywords(question)  # Extract key terms for better matching
                    })
                    
        print(f"✅ Loaded {len(qa_pairs)} Q&A pairs from CSV")
        
    except FileNotFoundError:
        print(f"❌ CSV file not found: {csv_file_path}")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        
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
            print(f"🔍 Matching '{user_input[:50]}' with '{qa_pair['question'][:50]}'")
            print(f"   Similarity: {similarity:.2f}, Keywords: {keyword_score:.2f}, Phrases: {phrase_score:.2f}")
            print(f"   Combined score: {combined_score:.2f}")
        
        if combined_score > best_score and combined_score > threshold:
            best_score = combined_score
            best_match = qa_pair
    
    if best_match:
        print(f"✅ Best CSV match (score: {best_score:.2f}): {best_match['question'][:100]}")
        return best_match['answer']
    else:
        print(f"❌ No CSV match found above threshold {threshold}")
        return None

def get_neural_response(msg):
    """Get response from the trained neural network model"""
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    print(f"🧠 Neural network confidence: {prob.item():.2f} for tag '{tag}'")
    
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f"✅ Neural network response: {response[:100]}...")
                return response
    
    print("❌ Neural network confidence too low")
    return None

def get_response(msg):
    """
    Main response function with enhanced logic:
    1. Try CSV matching first (for specific domain questions)
    2. Fall back to neural network if no CSV match
    3. Return default message if both fail
    """
    print(f"\n📝 Processing: '{msg}'")
    
    # First, try to find answer in CSV with enhanced matching
    csv_answer = find_csv_answer(msg)
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
        print(f"✅ Using CSV response: {full_response[:100]}...")
        return full_response
    
    # If no CSV match, try the neural network model
    print("🔄 No CSV match, trying neural network...")
    neural_response = get_neural_response(msg)
    if neural_response:
        return neural_response
    
    # If both fail, return a helpful default message
    default_responses = [
        "I don't have specific information about that. Could you rephrase your question or provide more details?",
        "I'm not sure about that particular topic. Can you try asking in a different way?",
        "I don't understand that question. Could you please be more specific or try rephrasing it?",
        "I need more information to help you with that. Can you provide additional details?",
    ]
    
    default_response = random.choice(default_responses)
    print(f"❌ Using default response: {default_response}")
    return default_response

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
        answer = find_csv_answer(question)
        print(f"Q: {question}")
        print(f"A: {answer[:100] + '...' if answer else 'No match found'}")
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

        resp = get_response(sentence)
        print(f"{bot_name}: {resp}")