def recursive(list,t):
    if len(list)==0:
        return False
    else:
        mid=(len(list))//2
        if list[mid]==t:
            return True
        elif list[mid]<t:
            return recursive(list[mid+1:],t)
        else:
            return recursive(list[:mid],t)
L=[x for x in range(1,101)]
t=10
print(recursive(L,t))        
        