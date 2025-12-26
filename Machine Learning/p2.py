#first store a values for 
#tell user to enter the password 
#3 tries for the password
#if the password is correct print " Correct "
#otherwise after 3 tries print " Incorrect "

password="password123"
for i in range(1,4):
    user=input("Enter the password : ")
    if user==password:
        print("Correct")
        break
    else:
        print("Incorrect")
        print(f"Attempt {i} is over ")
print("Attempts Over")