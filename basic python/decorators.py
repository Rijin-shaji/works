def add_cheese(hello):
    def wrapper():
        print("Added Cheese")
        hello()
        print("Cheese shawarma ready")
    return wrapper

@add_cheese
def shawarma():
    print("Shawarma here")
@add_cheese    
def mayo():
    print("Add mayo")
shawarma()    
mayo()        