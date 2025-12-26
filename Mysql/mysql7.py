import pandas as pd
from sqlalchemy import create_engine

# Load CSV
csv_file = r"C:/Users/shaji/Dropbox/PC/Downloads/sql(in).csv"
df_csv = pd.read_csv(csv_file)

print("CSV Preview:")
print(df_csv.head())

# Create SQLAlchemy engine (pymysql driver)
engine = create_engine("mysql+pymysql://root:Rijin%401234@localhost/ecommerce")

# Load MySQL table into Pandas
df_sql = pd.read_sql("SELECT user.*,seller.*,product.*,review.*,platforms.*,payment.*,issue.*,delivery_hub", con=engine)

print("MySQL Table Preview:")
print(df_sql.head())

df_csv.rename(columns={'Product_id': 'product_id'}, inplace=True)


# Merge CSV and MySQL table on 'product_id'
df_linked = pd.merge(df_csv, df_sql, on='product_id', how='inner')  # inner join
print("Linked Table:")
print(df_linked.head())