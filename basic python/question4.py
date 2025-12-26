import random

number=random.randint(1,100)
print(number)
n=0
while n!=number:
    n=float(input("Guess the number : "))
    if n<number:
        print("Number is smaller . Try again ")
    elif n>number:
        print("Number is bigger . Try again ")
    else:
        print("You gussed the correct number . Well Done ")        
        break