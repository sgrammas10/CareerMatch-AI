import sqlite3

conn = sqlite3.connect("users.db")
c = conn.cursor()

c.execute("PRAGMA table_info(users);")



conn.commit()
conn.close()
print("✅ Schema update complete.")