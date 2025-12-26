import mysql.connector
import pandas as pd 

# Connect to MySQL
mydb = mysql.connector.connect(
    host="localhost",       # or 127.0.0.1
    user="root",            # your MySQL username
    password="Rijin@1234", # your MySQL password
    database="ecommerce"  # your created DB name
)

# Check connection
if mydb.is_connected():
    print("Connected to MySQL Database")

# Create a cursor to interact with the DB
cursor = mydb.cursor()

# Example: Fetch all data from a table
cursor.execute("SELECT * FROM issue;")
columns = [i[0] for i in cursor.description]
rows = cursor.fetchall()

print(" | ".join(columns))
print("-" * 50)

# Print rows
for row in rows:
    print(" | ".join(str(x) for x in row))

# Close connection
cursor.close()
mydb.close()


