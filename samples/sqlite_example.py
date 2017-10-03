import sqlite3

conn = sqlite3.connect('preprocessing.db')
c = conn.cursor()

# Create a table
c.execute('''CREATE TABLE repos (date text, trans text)''')

# Insert row of data
c.execute("INSERT INTO repos VALUES('2006-01-06', 'BUY')")

conn.commit()
conn.close()
