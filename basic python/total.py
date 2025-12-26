def calculate_total(price, *args, tax=0.05):
    total = price + sum(args)
    final_total = total * (1 + tax) 
    return final_total

price = int(input("Enter the price: "))
args_input = input("Enter additional amounts (space-separated): ")
n = [int(x) for x in args_input.split()]
result=calculate_total(price, *n, tax=0.05)
print(result)