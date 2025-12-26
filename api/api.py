import mysql.connector
from flask import Flask, jsonify

app = Flask(__name__)

# MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Rijin@1234",
    database="ecommerce"
)
cursor = db.cursor(dictionary=True)  # returns rows as dictionaries

@app.route('/api/all-data')
def get_all_data():
    try:
        # Get all table names
        cursor.execute("SHOW TABLES")
        tables = [list(row.values())[0] for row in cursor.fetchall()]

        # Get data from each table
        all_data = {}
        for table in tables:
            cursor.execute(f"SELECT * FROM `{table}`")
            all_data[table] = cursor.fetchall()

        return jsonify({
            'success': True,
            'tables_count': len(all_data),
            'data': all_data
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)



