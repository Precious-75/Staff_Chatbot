from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class ChatHistory(db.Model):
    __tablename__ = 'conversations'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ChatHistory {self.id}>'

def init_db(app):
    """Initialize the database"""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully!")