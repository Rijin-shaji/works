import mysql.connector

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Rijin@1234",
    database="ecommerce"
)

cursor = conn.cursor()


# Step 2: Update parent
cursor.execute("""
UPDATE issue
SET product_id = 301
WHERE issue_id = 1
""")

# Save changes
conn.commit()

# Close connection
cursor.close()
conn.close()
