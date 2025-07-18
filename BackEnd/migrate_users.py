import sqlite3

#This is a one time script - do NOT run it automatically


conn = sqlite3.connect("users.db")
c = conn.cursor()

# Add columns if they don't exist
try:
    c.execute("ALTER TABLE users ADD COLUMN birthday TEXT")
except sqlite3.OperationalError:
    pass  # Column already exists

try:
    c.execute("ALTER TABLE users ADD COLUMN resume_path TEXT")
except sqlite3.OperationalError:
    pass

try:
    c.execute("ALTER TABLE users ADD COLUMN statement TEXT")
except sqlite3.OperationalError:
    pass

conn.commit()
conn.close()

print("Migration complete.")