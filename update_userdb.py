import sqlite3

conn = sqlite3.connect("users.db")
c = conn.cursor()

# Add columns if they do not exist already
try:
    c.execute("ALTER TABLE users ADD COLUMN birthday TEXT")
    print("✅ Added column: birthday")
except sqlite3.OperationalError:
    print("ℹ️ Column 'birthday' already exists")

try:
    c.execute("ALTER TABLE users ADD COLUMN resume_blob BLOB")
    print("✅ Added column: resume_blob")
except sqlite3.OperationalError:
    print("ℹ️ Column 'resume_blob' already exists")

try:
    c.execute("ALTER TABLE users ADD COLUMN statement TEXT")
    print("✅ Added column: statement")
except sqlite3.OperationalError:
    print("ℹ️ Column 'statement' already exists")

try:
    c.execute("ALTER TABLE users ADD COLUMN resume_path TEXT")
    print("✅ Added column: resume_path")
except sqlite3.OperationalError:
    print("ℹ️ Column 'resume_path' already exists")


conn.commit()
conn.close()
print("✅ Schema update complete.")