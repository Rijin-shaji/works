#calculator
#input the 2 values
#selection of the operations 
#def all the operations 
#printing the value

def add(n,m):
    add=n+m    
    return add

def sub(n,m):
    sub=n-m
    return sub

def mul(n,m) :
    mul=n*m
    return mul

def div(n,m):
     div=n/m
     return div
     

def mod(n,m):
    mod=n%m
    return mod

def fld(n,m):
    fld=n//m
    return fld

def sq(n,m) :
    sq= n**m
    return sq


try:
 a=float(input("Enter the first number : "))
 b=float(input("Enter the second number : "))
 print(" 1. Addition\n 2.Substraction\n 3.Mutiplication\n 4.Division \n 5.modulus \n 6.floor division \n 7.square ")    
 n=int(input("Choose the option : "))
 if n==1:
    print(add(a,b))
    p=add(a,b)
    print(f"Answer is {p} ")
 elif n==2:
    print(sub(a,b))
    p=sub(a,b)
    print(f"Answer is {p} ")    
 elif n==3:
    print(mul(a,b))
    p=mul(a,b)
    print(f"Answer is {p} ")
 elif n==4:
    try:
     b!=0
     print(div(a,b))
     p=div(a,b)
     print(f"Answer is {p} ")
    except:
     print("Not possible")    
 elif n==5:
    try: 
     b!=0
     print(mod(a,b))
     p=mod(a,b)
     print(f"Answer is {p} ")
    except:
     print("Not possible ")   
 elif n==6:
    try: 
     b!=0
     print(fld(a,b))
     p=fld(a,b)
     print(f"Answer is {p} ")
    except:
      print("Not possible")
 elif n==7:
    print(sq(a,b))
    p=sq(a,b)
    print(f"Answer is {p} ")
 else:
    print("Enter the number from the option ")   
except:
    print("Enter the number in a numerical way ")
else:
   print("Calculation completed")    
finally:
   print("Thank you for using our Calculator")
      


