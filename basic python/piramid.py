#user input 
#check the range
#print it like a piramid


n=int(input("Enter the number : "))
for i in range(1,n+1): 
     for s in range(n - i):
        print(" ", end=" ")
     for j in range(1, i + 1):
        print(j, end=" ")
     for k in range(i - 1, 0, -1):
        print(k, end=" ")
     print()