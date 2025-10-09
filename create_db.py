from db import app, db
from db import ChatHistory

with app.app_context():
	db.create_all()
	print("Database and tables created successfully.")
