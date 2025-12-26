import mysql.connector
from flask import Flask, Response, jsonify
from collections import OrderedDict
import json
import traceback

app = Flask(__name__)

# MySQL connection configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Rijin@1234",
    "database": "ecommerce"
}

# Optional: desired column order per table
table_columns_order = {
    "delivery_hub": ["hub_id", "hub_name", "address", "pin", "phone", "user_no", "product_no"],
    "issue": ["issue_id", "user_id", "product_id", "seller_id", "issue_type", "issue_describe", "issue_status", "raised_date"],
    "payment": ["payment_id", "user_id", "delivery_id", "amount", "payment_method", "payment_status", "seller_id"],
    "platforms": ["platform_id", "platform_name", "seller_id", "product_id"],
    "product": ["name", "brand", "product_id", "real_price", "discount_price", "stock_quantity", "product_type", "payment_id", "product_status", "Rev_id", "iss_id", "user_id", "seller_id"],
    "review": ["review_id", "user_id", "product_id", "seller_id", "rating", "review_text", "review_date"],
    "seller": ["name", "dob", "number", "sid", "email", "password", "gender", "nationality", "address", "shop_name", "gst_no", "p_id", "pay_id"],
    "user": ["name", "dob", "number", "usid", "email", "password", "gender", "nationality", "address"]
}

def get_db_connection():
    """Create a new database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        raise

@app.route("/", methods=["GET"])
def home():
    """Root endpoint"""
    return jsonify({
        "message": "E-commerce API",
        "endpoints": [
            "/api/all-tables",
            "/api/test"
        ]
    })

@app.route("/api/test", methods=["GET"])
def test():
    """Simple test endpoint"""
    return jsonify({"status": "success", "message": "API is working!"})

@app.route("/api/all-tables", methods=["GET"])
def get_all_tables():
    conn = None
    cursor = None
    
    try:
        print("Attempting to connect to database...")
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        print("Database connected successfully!")
        
        all_data = OrderedDict()

        # Step 1: Get all table names from the database
        print("Fetching table names...")
        cursor.execute("SHOW TABLES")
        tables = [list(row.values())[0] for row in cursor.fetchall()]
        print(f"Found tables: {tables}")

        for table in tables:
            print(f"Processing table: {table}")
            
            # Step 2: Get all columns in this table
            cursor.execute(f"SHOW COLUMNS FROM `{table}`")
            columns_in_db = [col['Field'] for col in cursor.fetchall()]

            # Step 3: Use custom order if provided, otherwise keep DB order
            if table in table_columns_order:
                columns_ordered = [col for col in table_columns_order[table] if col in columns_in_db]
                # Add any remaining columns that were not in the custom order
                remaining_cols = [col for col in columns_in_db if col not in columns_ordered]
                columns_ordered.extend(remaining_cols)
            else:
                columns_ordered = columns_in_db

            # Step 4: Fetch data
            cols_str = ", ".join(f"`{col}`" for col in columns_ordered)
            cursor.execute(f"SELECT {cols_str} FROM `{table}`")
            rows = cursor.fetchall()
            print(f"  -> Found {len(rows)} rows")

            # Step 5: Rebuild rows as OrderedDict to preserve column order
            ordered_rows = []
            for row in rows:
                ordered_row = OrderedDict()
                for col in columns_ordered:
                    if col in row:
                        ordered_row[col] = row[col]
                ordered_rows.append(ordered_row)

            all_data[table] = ordered_rows

        print("Data fetched successfully!")
        
        # Step 6: Return JSON preserving order
        return Response(
            json.dumps({"success": True, "data": all_data}, ensure_ascii=False, indent=2),
            mimetype="application/json",
            status=200
        )

    except mysql.connector.Error as db_err:
        error_msg = f"Database error: {str(db_err)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({
            "success": False, 
            "error": error_msg,
            "type": "database_error"
        }), 500
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({
            "success": False, 
            "error": error_msg,
            "type": "general_error"
        }), 500
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("Database connection closed")

if __name__ == "__main__":
    print("Starting Flask server...")
    print(f"Access the API at: http://192.168.1.67:5000/api/all-tables")
    # Removed the space before IP
    app.run(host="192.168.1.67", port=5000, debug=True)

