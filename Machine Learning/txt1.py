#read
#using pandas
import pandas as pd 

# Tab-separated
df = pd.read_csv("data.txt", sep="\t")

# Fixed-width columns
df = pd.read_fwf("data.txt")

#read entire file
with open("example.txt", "r") as f:
    content = f.read()  # reads whole file into a string
    print(content)

#write
#Writing to a New File
with open("example.txt", "w") as f:
    
    f.write("Hello World!\n")
    f.write("This will overwrite existing content.\n")

#pandas
import pandas as pd

df = pd.DataFrame({
    "Name": ["Roger", "Alin", "Jose"],
    "Age": [20, 21, 22]
})

# Save as text (tab-separated)
df.to_csv("data.txt", sep="\t", index=False)

# Save as comma-separated
df.to_csv("data.txt", index=False)

#visualising
import matplotlib.pyplot as plt

plt.plot(df['Age'], df['Score'], marker='o')
plt.title("Score vs Age")
plt.xlabel("Age")
plt.ylabel("Score")
plt.grid(True)
plt.show()

    

