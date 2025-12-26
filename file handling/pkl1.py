#Pickle is a Python module used for serializing and deserializing Python objects.
#Serialization → converting a Python object into a byte stream.
#Deserialization → converting a byte stream back into a Python object.
#Pickle files usually use the extension .pkl, but it’s not mandatory.

#reading
#using pandas
import pandas as pd

# Example DataFrame
df = pd.DataFrame({
    'Name': ['Roger', 'Alice', 'Bob'],
    'Age': [22, 25, 30]
})

# Save to Pickle file
df.to_pickle('dataframe.pkl')#pickle dumps

#Basic way
import pickle

# Read a single object from a Pickle file
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)

#using request
import pickle
import requests

url = 'https://example.com/data.pkl'
response = requests.get(url)
data = pickle.loads(response.content)

print(data)

#from compressed files
import pickle
import gzip

# Reading gzip-compressed pickle
with gzip.open('data.pkl.gz', 'rb') as f:
    data = pickle.load(f)

print(data)

#--writing
import pickle

data = {'name': 'Roger', 'age': 22}

# Write object to Pickle file
with open('data.pkl', 'wb') as f:  # 'wb' = write binary
    pickle.dump(data, f)

#using gzip
# import pickle
import gzip

data = {'scores': [90, 80, 70]}

with gzip.open('data.pkl.gz', 'wb') as f:
    pickle.dump(data, f)

#using pandas
import pandas as pd

df = pd.DataFrame({
    'Name': ['Rijin', 'Alice', 'Bob'],
    'Age': [22, 25, 30]
})

df.to_pickle('df.pkl')

#visualising    
#pandas
import pandas as pd
import matplotlib.pyplot as plt

# Load DataFrame
df = pd.read_pickle('df.pkl')
print(df.head())  # Check the data

# Example 1: Line plot
df.plot(x='Name', y='Age', kind='line', marker='o', title='Age Line Plot')
plt.show()

#pickle
import pickle
import pprint  # pretty print for better readability

# Load Pickle file
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

# Pretty print
pprint.pprint(data)

#numerical
import matplotlib.pyplot as plt

with open('numbers.pkl', 'rb') as f:
    numbers = pickle.load(f)  # e.g., [10, 20, 15, 30]

plt.plot(numbers, marker='o')
plt.title("Line Plot of Numbers from Pickle")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()



