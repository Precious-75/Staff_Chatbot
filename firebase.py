# import firebase_admin
# from firebase_admin import credentials, db
# import datetime
# import json

# print(" Initializing Firebase...")
# cred = credentials.Certificate(r"C:\Staff_Chatbot\chatbot-44c81-firebase-adminsdk-fbsvc-91ba707575.json")
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://chatbot-44c81-default-rtdb.firebaseio.com'
# })

# database_ref = db.reference()
# print("Firebase initialized successfully!")

# def save_message(message, sender, conversation_id=None):
#     print(f" Attempting to save message: '{message}' from {sender}")
#     try:
#         message_data = {
#             'user_message': message,
#             'sender': sender,  # 'user' or 'bot'
#             'timestamp': datetime.datetime.now().isoformat(),
#             'conversation_id': conversation_id or 'default'
#         }
        
#         print(f"Message data prepared: {message_data}")
        
#         messages_ref = database_ref.child('chat_messages')
#         result = messages_ref.push(message_data)
#         print(f" Message saved successfully! Key: {result.key}")
        
#     except Exception as e:
#         print(f"Error saving message: {e}")
#         import traceback
#         traceback.print_exc()