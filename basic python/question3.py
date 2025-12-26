
 
count=0
d={'I':1,'V':5,'X':10,'L':50}
num="0123456789"
sc="!@#$%^&*()_+=-?/>.<,`~"
l=0
yes=0
while True:
 n=input("Enter the password : ")
 if len(n)>=12:
    if "2025"in n:
      for i in n:
        if i ==i.lower():
         yes+=1
         continue
        else:
          print("Lower case missing ")
        if i in num:
          yes+=1
        if yes>=1:
         continue
        else:
         print("Number is missing ")
        if i in sc:
         yes+=1
        if yes>=1:
         continue
        else:
         print("Special characters are missing ")
        if i==i.upper():
         if i in d:
          count+=d[i]
          yes+=1
          continue
        else:
         print("Upper case are missing ")
        if i in num:
         l+=i
         yes+=1
        continue
    else:
      print("2025 missing ")
 else:
  print("Minimum 12 characters required ")    
 if l==125:   
     yes+=1
 else:
   print("Number sum is not 125 ")        
 if yes==len(n):
  print("Password is correct ")
  break   
 else:
  print("Password is incorrect ")
  
 
        
   
    
       