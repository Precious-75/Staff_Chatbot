from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Create SQLAlchemy instance without binding to a Flask app yet.
# The binding happens in init_db(app) to avoid creating multiple Flask apps.
db = SQLAlchemy()


class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100))  # optional unique user id
    user_message = db.Column(db.Text)
    bot_response = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


def init_db(flask_app):
    """Initialize SQLAlchemy with the provided Flask app and create tables."""
    db.init_app(flask_app)
    with flask_app.app_context():
        db.create_all()