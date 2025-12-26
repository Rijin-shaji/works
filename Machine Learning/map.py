a=input("Enter the numbers").split(" ")
b=list(map(int,a)) 
cube=list(map(lambda b : b**3,b))  
print(cube)