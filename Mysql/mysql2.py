import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Rijin@1234",
    database="ecommerce"
)

query = "select * from user; "  
df = pd.read_sql(query, conn)

conn.close()
print(df)
