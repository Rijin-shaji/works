#value we need to find is 5
#set a range 1 to 100
#find the time it takes to seach it.
#print it 

def binary(t,l):
 f=0
 last=len(l)-1 
 while f<=last:
   mid=(last+f)//2
   print(mid)
   if t<mid:
    last=mid
   elif t>mid:
    f=mid+1
   else:
    print("You got the Number ") 
    break       
 return True   
t=5
l=[x for x in range (1,100)]
print(binary(t,l))

