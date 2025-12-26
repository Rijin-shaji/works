#Enter the string
#Check each character 
#Check the vowels or not
#If its a vowel count it and remove the character
#If it's not a vowel print the character  


n=input("Eneter the word : ")
m=""
up=n.lower()
count=0
vowels="aeiou"
p="abcdefghijklmnopqrstuvwxyz"
for i in up:
        if i in p :
          if i in vowels:
            count+=1
            n.replace(i,"")
          else:
            m+=i 
        else:
         print("Invalid")  
         break             
print(count)        
print(m)
 


    
