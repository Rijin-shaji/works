# CSV =Common separated Values
#A plain text file used to store tabular data .
#Each line = a Row 
#Each column = a field (attribute).
#Values are usually separated by a comma (,), but can also use other delimiters (;, |, \t)
#Name, Age, Country

#Reading a CSV file
import csv

with open("data.csv", "r") as file:  # r= read mode .
    reader = csv.reader(file)        # with ensures the file will be automatically closed after use.
    for row in reader:               #  CSV file usually have multiple rows
        print(row)                   # We want to process them one by one 
                                     # A for loop is the natural way to go through each row.
#without loop?
#If you want all rows at once , you can convert it into a list
# using list

rows=list(reader)                    # reader = it is an Iterator.
print(rows)                          # It reads the CSV Files row by Row

# using next
import csv
with open("students.csv", "r") as file: # file variable name.
    reader = csv.reader(file)
    header = next(reader)   # read the first row
    print("Header:", header)

    first_row = next(reader)  # read the next row
    print("First data row:", first_row)

#using Dictreader
import csv
with open("students.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)  # each row is a dictionary
#Dictreader = Instead of giving rows as lists, It give them as Dictionaries.  

# Pandas
import pandas as pd #data manipulation, filtering, grouping.
df = pd.read_csv("students.csv")
print(df)
#it is the most powerful & easiest way. 
#reasons
#Simple sintax
#Data is stored in a DataFrame
#Powerfull Data Analysis Tools
#Handles large files better
#Handles Missing Data
#Support many Formats

#Use CSV if you only need to read/write small CSV files Quickly.
# Use pandas if you need fast,clean and powerful data manipulation.

#Writing
#simple row-by-row writing.
import csv

data = [
    ["Name", "Age", "Grade"],
    ["Alice", 20, "A"],
    ["Bob", 22, "B"]
]

with open("students.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)  # write multiple rows 

#when working with dictionaries.
import csv

with open("students.csv", "w", newline="") as file:
    fieldnames = ["Name", "Age", "Grade"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()   # writes: Name,Age,Grade
    writer.writerow({"Name": "Alice", "Age": 20, "Grade": "A"})
    writer.writerow({"Name": "Bob", "Age": 22, "Grade": "B"})

#Pandas
#best for structured, tabular datasets.
import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [20, 22, 19],
    "Grade": ["A", "B", "A"]
}

df = pd.DataFrame(data)
df.to_csv("students.csv", index=False) 

#numpy
#for numeric/scientific data.
import num as np #numeric computations, arrays, matrices

data = np.array([[1, 2, 3], [4, 5, 6]])
np.savetxt("numbers.csv", data, delimiter=",", fmt="%d")

#how to append new rows to an existing CSV
import csv

new_rows = [
    ["David", 23, "C"],
    ["Emma", 21, "B"]
]

with open("students.csv", "a", newline="") as file:  # "a" = append mode
    writer = csv.writer(file)
    writer.writerows(new_rows)   # add multiple rows

#instead of " w " use " a "

#Visualizing
import pandas as pd

# Read CSV
df = pd.read_csv("students.csv")

# Show first 5 rows
print(df.head())

#using prettytable >> It It will arrange everything in a table.
import csv
from prettytable import PrettyTable

table = PrettyTable()

with open("students.csv", "r") as file:
    reader = csv.reader(file)
    headers = next(reader)
    table.field_names = headers
    for row in reader:
        table.add_row(row)

print(table)

#using graph
import pandas as pd
import matplotlib.pyplot as plt  #matplotlib is a Python library for creating graphs and visualizations.

df = pd.read_csv("students.csv")

# Bar chart of student ages
plt.bar(df["Name"], df["Age"])  #bar for bar graph
plt.xlabel("Name")
plt.ylabel("Age")
plt.title("Student Ages")
plt.show()

#seaborn graph
import pandas as pd
import seaborn as sns           #Seaborn is a Python library built on top of matplotlib.
import matplotlib.pyplot as plt #easier to create attractive and informative graphs with less code than plain matplotlib.

df = pd.read_csv("students.csv")

sns.barplot(x="Name", y="Age", data=df)
plt.show()

#plotly
import pandas as pd
import plotly.express as px #Plotly Express is a high-level Python library for creating interactive plots quickly.

df = pd.read_csv("students.csv")

fig = px.bar(df, x="Name", y="Age", color="Grade", title="Student Ages by Grade")
fig.show()

#difference
#---Matplotlib
#A basic, static plotting library.
#Very flexible but requires more code to make plots look good.
#Good for custom, detailed control over plots.
#---Seaborn
#Built on Matplotlib, focused on statistical plots.
#Provides beautiful default styles and works directly with pandas DataFrames.
#Easier than Matplotlib for common charts like boxplots, violin plots, pairplots.
#---Plotly Express
#High-level library for interactive plots (zoom, hover, pan).
#Very simple to use, often one line of code for complex charts.
#Best for interactive dashboards or web visualizations.

#delimiter="," --- Defines the character that separates values in your CSV file.
#quotechar='"' --- enclose fields that contain the delimiter
#eg
Name,Comment
Alice,"Hello, world!"
Bob,"Python, CSV"
#With quotechar='"', it treats "Hello, world!" as one field.
#skipinitialspace=True -- Ignores spaces immediately following the delimiter.
#lineterminator="\n" -- Each row will end with a newline character

#graphs -- lineplot,pie,histplot,boxplot etc.