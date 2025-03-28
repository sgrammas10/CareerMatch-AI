from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import re
import smtplib
from email.mime.text import MIMEText

auth = Blueprint("auth", __name__)

# Helper function to send a welcome email
def send_welcome_email(email, username):
    sender_email = "careermatchainoreply@gmail.com"
    sender_password = "your-email-password"  # Replace with your email password or app-specific password
    subject = "Welcome to CareerMatch AI"
    body = f"Hi {username},\n\nWelcome to CareerMatch AI! We're excited to have you on board.\n\nBest regards,\nThe CareerMatch AI Team"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())
    except Exception as e:
        print(f"Failed to send email: {e}")

# Sign-Up Route
@auth.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username = data["username"]
    email = data["email"]
    password = data["password"]

    # Validate password
    if not re.match(r"^(?=.*[!@#$%^&*(),.?\":{}|<>])[A-Za-z\d!@#$%^&*(),.?\":{}|<>]{8,}$", password):
        return jsonify({"error": "Password must be at least 8 characters long and include at least one symbol."}), 400

    # Hash the password
    hashed_password = generate_password_hash(password)

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    # Check if username or email already exists
    cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
    existing_user = cursor.fetchone()
    if existing_user:
        conn.close()
        return jsonify({"error": "Username or email already exists."}), 400

    # Insert new user into the database
    try:
        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed_password))
        conn.commit()
        send_welcome_email(email, username)  # Send welcome email
        return jsonify({"message": "Account created successfully"}), 201
    except Exception as e:
        return jsonify({"error": "An error occurred during sign-up."}), 500
    finally:
        conn.close()