#user input , any number
#check wether it's a prime number or not
#print
"""try:
 num=float(input("Enter any number : "))
 count=0
 m=num**0.5   
 if num<1:
    print(" Enter a positive number ")
 elif int(num)!=num:
    print(" Invalid number ") 
 else:
    for i in range(1,int(m)+1):
        if num%i==0 :
            count+=1
    if count>=2:
        print("it's not a prime number")
    else:
        print("it's a prime number ")      
except:
   print("Invalid")"""


l=[]
num=int(input("Enter the number : "))
for i in range(2,num+1):
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
print(l)