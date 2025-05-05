import sqlite3

conn = sqlite3.connect("users.db")
c = conn.cursor()

# Add the new column to store user statements
c.execute("ALTER TABLE users ADD COLUMN statement TEXT")

conn.commit()
conn.close()

print("✅ 'statement' column added successfully.")
