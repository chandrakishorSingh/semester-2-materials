# precalculate fibonacci numbers so that they can be used later
def pre_calculate_fibo(n):
    fibo = [1, 1]
    for i in range(1, n):
        fibo.append(fibo[len(fibo) - 1] + fibo[len(fibo) - 2])

    return fibo

# returns the points x1, x2 in for the given range of [a, b]
def find_mid_points(interval, n, i, fibo):
    a = interval[0]
    b = interval[1]

    x1 = (fibo[n - i - 1] / fibo[n - i + 1]) * (b - a) + a
    x2 = (fibo[n - i] / fibo[n - i + 1]) * (b - a) + a

    return [x1, x2]

def fibonacci_search_method(f, inputs, max_iter=5 + 2):
    # inputs: a matrix of order n x 2, ith row represent the interval for ith variable of function f
    dimension = len(inputs)

    fibo = pre_calculate_fibo(10)

    iter_count = 1
    while iter_count <= (max_iter - 2):

        mid_points = [find_mid_points(vector, max_iter, iter_count, fibo) for vector in inputs]

        vector1 = [mid_points[i][0] for i in range(dimension)]
        vector2 = [mid_points[i][1] for i in range(dimension)]

        if f(vector1) < f(vector2):
            print(f"After {iter_count}, value of function is {f(vector1)}")
            # now range will be [a, d]
            for i in range(dimension):
                inputs[i][1] = vector2[i]
        else:
            print(f"After {iter_count}, value of function is {f(vector2)}")
            # now range will be [c, b]
            for i in range(dimension):
                inputs[i][0] = vector1[i]

        print(f"new ranges of inputs are {inputs}")

        iter_count += 1



# define function to minimize and its input
f = lambda x: x[0]**2 - 4*x[0] + x[1]**2 - x[1] - x[0]*x[1]
inputs = [[2.5, 3.5], [1.5, 2.5]]

# call the function to minimize the function
fibonacci_search_method(f, inputs)