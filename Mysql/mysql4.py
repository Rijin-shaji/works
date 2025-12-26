import mysql.connector

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Rijin@1234",
    database="ecommerce"
)

cursor = conn.cursor()

# Insert one row into 'issue' table
sql = "INSERT INTO issue (issue_id,user_id,product_id,seller_id,issue_type,issue_describe,issue_status,raised_date) VALUES (%s, %s,%s, %s,%s, %s,%s, %s);"
values = (16, 117,41,55,58,77,85,"2025-10-11")
cursor.execute(sql, values)

cursor.executemany(sql)

# Commit changes
conn.commit()

# Close connection
cursor.close()
conn.close()
