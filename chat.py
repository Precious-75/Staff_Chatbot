import random
import json
import csv
from difflib import SequenceMatcher

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

# Load CSV data
def load_csv_qa(csv_file_path):
    """
    Load Q&A pairs from CSV file
    Expected CSV format: Question,Answer,Category,Access
    """
    qa_pairs = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)  # Use csv.reader for your format
            
            for row in csv_reader:
                if len(row) >= 2:  # Make sure we have at least question and answer
                    qa_pairs.append({
                        'question': str(row[0]).lower().strip(),
                        'answer': str(row[1]).strip()
                    })
        print(f"Loaded {len(qa_pairs)} Q&A pairs from CSV")
    except FileNotFoundError:
        print(f"CSV file not found: {csv_file_path}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
    return qa_pairs

# Load your CSV file
csv_qa_pairs = load_csv_qa('school_it_qa.csv')

def find_csv_answer(user_input, threshold=0.6):
    """
    Find matching answer from CSV using string similarity
    """
    user_input = user_input.lower().strip()
    best_match = None
    best_score = 0
    
    for qa_pair in csv_qa_pairs:
        # Calculate similarity between user input and CSV question
        similarity = SequenceMatcher(None, user_input, qa_pair['question']).ratio()
        
        # Also check if user input contains key words from the question
        words_match = any(word in user_input for word in qa_pair['question'].split() if len(word) > 3)
        
        if similarity > best_score and similarity > threshold:
            best_score = similarity
            best_match = qa_pair
        elif words_match and similarity > threshold * 0.7:
            best_score = similarity
            best_match = qa_pair
    
    return best_match['answer'] if best_match else None

def get_response(msg):
    # First, try to find answer in CSV
    csv_answer = find_csv_answer(msg)
    if csv_answer:
        # Add a greeting/intro before the CSV answer
        intro_messages = [
            "Here's what I found for you: ",
            "I'm accessing our database to find the most current information. ",
            "Allow me to compile the relevant data for your inquiry.",
            "I appreciate your patience while I prepare a thorough response. ",
        ]
        intro = random.choice(intro_messages)
        return f"{intro}{csv_answer}"
    
    # If no CSV match, use the neural network model
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand your question. Can you please explain "    

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(f"{bot_name}: {resp}")