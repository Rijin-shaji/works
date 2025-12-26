#enter the first word
#enter the second word
#check wether the strings are anagram
#if "YES" print yes
#else print not

a=input("Enter the first word : ")
b=input("Enter the second word : ")
count=0
aa=len(a)
bb=len(b)
if aa==bb:
    for i in a:
        if i in b:
            count+=1
        else:
            break
    if count==aa:
        print("It's a anagram ")
    else:
        print("It's not a anagram ")  
else:
    print("It's not a anagram ")        

                

        