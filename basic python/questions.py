def prime(n):
 l=[]
 for i in range(2,n+1):
     count=0
     m=i**0.5   
     if i<1:
      print(" Enter a positive number ")
     elif int(i)!=i:
      print(" Invalid number ") 
     else:
      for j in range(1,int(m)+1):
        if i%j==0 :
            count+=1
      if count<2:
        l.append(i)
 return l


n=int(input("Enter the number : "))
prime(n)
x=prime(n)
for i in range(1,n+1):
    if i in x:   
      for s in range(n - i):
        print(" ", end="")
      for j in range(1, i +1):
        print("*", end=" ")
    else:
      for p in range(n - i):
        print(" ", end="")
      for q in range(1, i + 1):     
        print(str(i),end=" ")
       
    print()
     