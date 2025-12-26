"""cube=0
number=input("Enter the number : ")
length=len(number)
for i in number:
    cube+=int(i)**length
print(cube)    
if int(number)==cube:
    print("It's a Amstrng number ")
else:
    print("It's not a Amstrng number ")"""



l=[]
value=input("Enter a number")
for i in range(1,int(value)):
        cube=0
        length=len(str(i))
        for j in str(i):
         cube+=int(j)**length
         if int(i)==cube:
             l.append(i)
         else:
             continue     
print(l)            
 