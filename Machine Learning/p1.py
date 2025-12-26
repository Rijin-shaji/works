# user   input
# to check wether the number is even or odd
# if the number is Even add
# if odd skip
#last print the sum value of Even numbers

add=0
n=0
num=int(input("Enter the Number : "))
for i in range(num+1):
    if i%2==0:
       add+=i
       n+=1
print(f"Sum of Even numbers of {num} : {add} ")
print(f"Count of the Even numbers : {n}")
