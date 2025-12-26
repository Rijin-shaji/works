def dt(b):
    d={"sum":sum(b),"max":max(b),"min":min(b),"Average":sum(b)/a}
    return d 

a=int(input(" Total count of numbers : "))
b=[]
for i in range(a):
    c=int(input(f"Enter the {i+1} numbers : "))
    b.append(c)
print(dt(b))

      