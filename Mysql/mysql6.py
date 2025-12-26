import mysql.connector
from graphviz import Digraph

# MySQL connection
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Rijin@1234",
    database="ecommerce"
)
cursor = mydb.cursor()

# Get all tables
cursor.execute("SHOW TABLES")
tables = [t[0] for t in cursor.fetchall()]

# Create Graphviz Digraph
dot = Digraph(comment='ER Diagram', format='png')
dot.attr(rankdir='LR')  # Left-to-right layout
dot.attr(dpi='300')     # High resolution

# Pre-fetch all foreign keys to identify FK columns
cursor.execute(f"""
    SELECT TABLE_NAME, COLUMN_NAME
    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
    WHERE TABLE_SCHEMA = '{mydb.database}'
      AND REFERENCED_TABLE_NAME IS NOT NULL;
""")
fk_columns = cursor.fetchall()  # list of tuples (table, column)

# Add table nodes with columns
for table in tables:
    cursor.execute(f"""
        SELECT COLUMN_NAME, COLUMN_TYPE, COLUMN_KEY
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{mydb.database}'
          AND TABLE_NAME = '{table}';
    """)
    columns = cursor.fetchall()
    
    col_strings = []
    for col_name, col_type, col_key in columns:
        # Determine column type for diagram
        if col_key == 'PRI':
            # Primary key: dark color and bold
            col_strings.append(f"<FONT COLOR='Red'><B>{col_name} : {col_type} (PK)</B></FONT>")
        elif (table, col_name) in fk_columns:
            # Foreign key: different color
            col_strings.append(f"<FONT COLOR='blue'>{col_name} : {col_type} (FK)</FONT>")
        else:
            # Normal column
            col_strings.append(f"{col_name} : {col_type}")
    
    # Build node label as HTML table
    label = f"<<TABLE BORDER='0' CELLBORDER='1' CELLSPACING='0' COLOR='darkgreen' BGCOLOR='lightyellow'><TR><TD><B>{table}</B></TD></TR>"
    for col in col_strings:
        label += f"<TR><TD ALIGN='LEFT'>{col}</TD></TR>"
    label += "</TABLE>>"
    
    dot.node(table, label=label, shape='Box') # To create the Tables

# Add foreign key relationships as edges
for table in tables:
    cursor.execute(f"""
        SELECT COLUMN_NAME, REFERENCED_TABLE_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = '{mydb.database}'
          AND TABLE_NAME = '{table}'
          AND REFERENCED_TABLE_NAME IS NOT NULL;
    """)
    for col, ref_table in cursor.fetchall():
        dot.edge(table, ref_table, label=col,color='darkgreen', fontcolor='darkblue')  #To set the arrows

# Save and render diagram
dot.render('erdiagram_columns_fk', view=True)

cursor.close()
mydb.close()
print("✅ ER diagram with PK and FK columns saved as 'erdiagram_columns_fk.png'")




