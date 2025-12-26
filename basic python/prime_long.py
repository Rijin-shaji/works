#user input , any number
#check wether it's a prime number or not
#print
try:
 num=float(input("Enter any number : "))
 count=0   
 if num<1:
    print(" Enter a positive number ")
 elif int(num)!=num:
    print(" Invalid number ") 
 else:
    for i in range(1,int(num+1)):
        if num%i==0:
            count+=1
    if count>2:
        print("it's not a prime number")
    else:
        print("it's a prime number ")                   
except:
   print("Invalid")        