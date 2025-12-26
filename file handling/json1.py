#JSON = lightweight format for storing and exchanging data.
#Human-readable (like a dictionary), but language-independent.
#Looks like a Python dictionary (key-value pairs).
#Commonly used in:
#Web APIs
#Config files (.json)
#Data exchange between programs

#--Read
#Reading JSON from a String
import json

# JSON string
data = '{"name": "Roger", "age": 24, "skills": ["Python", "AI"]}'

# Parse JSON string → Python dict
python_obj = json.loads(data)

print(python_obj)
print(python_obj["skills"][0])  # Access the skills list,then the element at index 0(Python)

#From a JSON File
import json

with open("data.json", "r") as f:
    data = json.load(f)

print(data)

#using request library
import requests #to take datas from the link

response = requests.get("https://jsonplaceholder.typicode.com/todos/1")
data = response.json()  # direct JSON parsing
print(data["title"])

#Using Pandas
import pandas as pd

df = pd.read_json("data.json")
print(df.head())

#--write
import json

python_obj = {
    "name": "Roger",
    "age": 24,
    "skills": ["Python", "AI"]
}

# Convert Python dict to JSON string
json_str = json.dumps(python_obj) 
print(json_str)
#dump means converting python objects to Jsonstring.
#load() means convert Json strings into python objects

#Write JSON to a File
import json

python_obj = {
    "name": "Rijin",
    "age": 25,
    "skills": ["Python", "AI"]
}

with open("data.json", "w") as f:
    json.dump(python_obj, f)  # writes compact JSON

# Pretty print in file
with open("data_pretty.json", "w") as f:
    json.dump(python_obj, f, indent=4, sort_keys=True)
#indent=if set to an integers, it enables printing by adding indentation,and newlines.
#f-string format
#sorted_key=true means sorts the output dictionary alphabetically.

#using Pandas
import pandas as pd

df = pd.DataFrame([
    {"id": 1, "name": "A"},
    {"id": 2, "name": "B"}
])

# Save to file
df.to_json("data.json", orient="records", indent=4)#orient convertong to Json strings

#visualizing
#Pretty Printing (Formatted JSON)
import json

data = {
    "name": "Roger",
    "age": 24,
    "skills": ["Python", "AI", "Data Science"],
    "details": {"city": "Kochi", "hobbies": ["Reading", "Music"]}
}

# Pretty print
print(json.dumps(data, indent=4))

#using pandas
import pandas as pd

json_data = [
    {"name": "Rijin", "score": 90},
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 95}
]

df = pd.DataFrame(json_data)

# Visualize using matplotlib
import matplotlib.pyplot as plt

plt.bar(df["name"], df["score"])
plt.show()

