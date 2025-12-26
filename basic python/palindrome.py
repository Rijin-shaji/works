#User input
#Check each characters
#check the word is a palindrome or not
#if it is a palindrome show "It is a palindrome"
#if not show "It is not"

n=input("Enter any word : ")
m=n[::-1]
if n==m:
     print(f"It's a palindrome .\n{n} = {m}")
else:
     print(f"It's not a palindrome.\n{n} != {m}")  
     
   


