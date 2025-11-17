class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const {openButton, chatBox, sendButton} = this.args;

        // Add initial greeting message
        this.addInitialGreeting(chatBox);

        openButton.addEventListener('click', () => this.toggleState(chatBox))

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }


    // Add this new method to display initial greeting
    addInitialGreeting(chatbox) {
        let greetingMsg = { 
            name: "Support Desk", 
            message: "Hello! How can I be of Assistance today?" 
        };
        this.messages.push(greetingMsg);
        this.updateChatText(chatbox);
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hides the box
        if(this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    // Create typing indicator
    createTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'messages__item messages__item--visitor typing-indicator';
        typingDiv.id = 'typingIndicator';
        
        const dotsDiv = document.createElement('div');
        dotsDiv.className = 'typing-dots';
        dotsDiv.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        
        typingDiv.appendChild(dotsDiv);
        return typingDiv;
    }

    // Show typing indicator
    showTypingIndicator(chatbox) {
        // Remove any existing typing indicator first
        this.hideTypingIndicator();
        
        const typingIndicator = this.createTypingIndicator();
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.appendChild(typingIndicator);
        
        // Scroll to bottom
        chatmessage.scrollTop = chatmessage.scrollHeight;
        
        return typingIndicator;
    }

    // Hide typing indicator
    hideTypingIndicator() {
        const existingIndicator = document.getElementById('typingIndicator');
        if (existingIndicator && existingIndicator.parentNode) {
            existingIndicator.parentNode.removeChild(existingIndicator);
        }
    }

    // Disable/enable input during processing
    setInputState(chatbox, enabled) {
        const textField = chatbox.querySelector('input');
        const sendButton = chatbox.querySelector('.send__button');
        
        textField.disabled = !enabled;
        sendButton.disabled = !enabled;
        
        if (enabled) {
            textField.focus();
        }
    }

    // Updated onSendButton with 3-second typing indicator
    async onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        // Disable input while processing
        this.setInputState(chatbox, false);

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);

        // Clear input and update chat to show user message
        textField.value = '';
        this.updateChatText(chatbox);

        // Show typing indicator
        const typingIndicator = this.showTypingIndicator(chatbox);

        try {
            // Start API call but don't await it yet
            const apiPromise = fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: JSON.stringify({ message: text1 }),
                mode: 'cors',
                headers: {
                  'Content-Type': 'application/json'
                },
            });

            // Ensure typing indicator shows for at least 3 seconds
            const minDelayPromise = new Promise(resolve => setTimeout(resolve, 3000));

            // Wait for both API response and minimum delay
            const [response] = await Promise.all([apiPromise, minDelayPromise]);
            const result = await response.json();
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            // Add bot response
            let msg2 = { name: "Support Desk", message: result.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox);

        } catch (error) {
            console.error('Error:', error);
            
            // Ensure minimum delay even on error
            await new Promise(resolve => setTimeout(resolve, 3000));
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            // Add error message
            let errorMsg = { name: "Support Desk", message: "Sorry, I encountered an error. Please try again." };
            this.messages.push(errorMsg);
            this.updateChatText(chatbox);
            
        } finally {
            // Re-enable input
            this.setInputState(chatbox, true);
        }
    }

    updateChatText(chatbox) {
        var html = '';
        // Remove .reverse() to show messages in chronological order
        this.messages.forEach(function(item, index) {
            if (item.name === "Support Desk") 
            {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            }
            else 
            {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
          });
          
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        
        // Save typing indicator if it exists
        const existingTypingIndicator = chatmessage.querySelector('#typingIndicator');
        
        chatmessage.innerHTML = html;
        
        // Re-append typing indicator if it was there
        if (existingTypingIndicator) {
            chatmessage.appendChild(existingTypingIndicator);
        }
        
        // Scroll to bottom after updating
        chatmessage.scrollTop = chatmessage.scrollHeight;
    }
}

// Fixed sendMessage function - now uses correct URL
async function sendMessage(message) {
    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {  // Fixed: now matches the main fetch
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // Add the response to chat
        addMessageToChat(data.answer, 'bot');  // Keep your existing structure
        
    } catch (error) {
        console.error('Error:', error);
        addMessageToChat('Sorry, I\'m having trouble connecting.', 'bot');
    }
}

// Floating text management
class FloatingTextManager {
    constructor() {
        this.floatingText = document.getElementById('floating-text');
        this.chatbotGreeting = document.getElementById('chatbot-greeting');
        this.chatboxButton = document.querySelector('.chatbox__button button');
        this.intentsData = null;
        
        this.init();
    }

    async init() {
        // Load intents data
        await this.loadIntents();
        
        // Set initial floating text
        this.updateFloatingText();
        
        // Add click listener to chatbot button
        this.chatboxButton.addEventListener('click', () => {
            this.hideFloatingText();
        });
    }

    async loadIntents() {
        try {
            // Replace 'intents.json' with your actual file path
            const response = await fetch('./intents.json');
            this.intentsData = await response.json();
            console.log('Intents loaded successfully');
        } catch (error) {
            console.error('Error loading intents:', error);
        }
    }

    updateFloatingText() {
        if (!this.intentsData) return;

        // Look for floating text configuration in intents
        const floatingTextIntent = this.intentsData.intents?.find(
            intent => intent.tag === 'floating_text' || intent.tag === 'greeting'
        );

        let textToShow = "Hello! How may I assist you today?"; // Default

        if (floatingTextIntent && floatingTextIntent.responses) {
            // Use the first response as floating text
            textToShow = floatingTextIntent.responses[0];
        } else if (this.intentsData.floating_text) {
            // If there's a direct floating_text property
            textToShow = this.intentsData.floating_text;
        } else if (this.intentsData.default_greeting) {
            // If there's a default greeting
            textToShow = this.intentsData.default_greeting;
        }

        // Update both floating text and chatbot greeting
        this.floatingText.textContent = textToShow;
        if (this.chatbotGreeting) {
            this.chatbotGreeting.textContent = textToShow;
        }
    }

    hideFloatingText() {
        this.floatingText.classList.add('hidden');
    }

    showFloatingText() {
        this.floatingText.classList.remove('hidden');
    }

    // Method to update floating text from intents dynamically
    setFloatingTextFromIntent(intentTag) {
        if (!this.intentsData) return;

        const intent = this.intentsData.intents?.find(i => i.tag === intentTag);
        if (intent && intent.responses && intent.responses.length > 0) {
            const newText = intent.responses[0];
            this.floatingText.textContent = newText;
            if (this.chatbotGreeting) {
                this.chatbotGreeting.textContent = newText;
            }
        }
    }
}

// Initialize floating text manager when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.floatingTextManager = new FloatingTextManager();
});

const chatbox = new Chatbox();
chatbox.display();