import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

def main():

    # Ask the user for input N (positive integer) and reads it.
    n = int(input("Please input a positive integer N: "))
    print(f"N = {n}")

    # Ask the user for input k (positive integer) and reads it.
    k = int(input("Please input a positive integer k: "))
    print(f"k = {k}")

    # Ask the user to provide N (x, y) points (one by one)
    # Read all of them: first: x value, then: y value for every point one by one. 
    # X and Y are the real numbers.
    x_train = []
    y_train = []
    print(f"\nPlease input {n} (x, y) points one by one.")
    for i in range(n):
        xi = float(input(f"Enter x for point {i + 1}: "))
        yi = float(input(f"Enter y for point {i + 1}: "))
        x_train.append(xi)
        y_train.append(yi)
    
    # Ask the user for input X 
    x = float(input("Please input X: "))
    y_true = float(input("Please input y: "))
    print(f"X = {x}")
    print(f"y = {y_true}")

    # Perform k-NN Regression
    ## Ensure k is less than or equal to N
    if k > n:
        print("Error: k cannot be greater than N.")
        return
    
    ## If k <= N, perform k-NN Regression
    ### run knn regressor
    neigh = KNeighborsRegressor(n_neighbors=n)
    neigh.fit(np.array(x_train).reshape(-1, 1), np.array(y_train))
    y_pred = neigh.predict([[x]])
    
    ### Output the result
    print(f"The predicted Y value for X={x} using {k}-NN Regression is: {y_pred[0]}")

    ### provide the coefficient of determination
    print('Coefficient of determination is:', r2_score([y_true], y_pred))

if __name__ == "__main__":
    main()
