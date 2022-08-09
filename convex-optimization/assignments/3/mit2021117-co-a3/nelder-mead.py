import numpy as np

def nelder_mead(f, inputs, max_iter=5):
    step = 0.1
    alpha = 1.0
    gamma = 2.0
    rho = -0.5
    sigma = 0.5

    # setup all the variables
    dimension = len(inputs[0])
    result = []

    for input in inputs:
        x = np.array(input)
        score = f(x)
        result.append([x, score])

    # do till `max_iter`
    iter_count = 0
    while True:
        # break after max_iter
        if iter_count >= max_iter:
            return result
        iter_count += 1

        # sort the values
        result.sort(key=lambda x: x[1])
        best = result[0][1]

        print(f"After i = {iter_count}, minimum value is:{best} and input vectors are {[vector[0] for vector in result]}")

        # calculate the centroid
        x_centroid = np.array([0.0] * dimension)
        for tup in result[:-1]:
            for index, coordinate in enumerate(tup[0]):
                x_centroid[index] += coordinate / (len(result)-1)

        # check for reflection point
        xr = x_centroid + alpha*(x_centroid - result[-1][0])
        rscore = f(xr)
        if result[0][1] <= rscore < result[-2][1]:
            # got a better point then second worst, so add it
            del result[-1]
            result.append([xr, rscore])
            continue

        # check for expansion point
        if rscore < result[0][1]:
            x_expansion = x_centroid + gamma*(x_centroid - result[-1][0])
            escore = f(x_expansion)
            if escore < rscore:
                # expansion point is better than reflection point
                del result[-1]
                result.append([x_expansion, escore])
                continue
            else:
                del result[-1]
                # reflection point is better than expansion point
                result.append([xr, rscore])
                continue

        # check for contraction point
        x_contraction = x_centroid + rho*(x_centroid - result[-1][0])
        cscore = f(x_contraction)
        if cscore < result[-1][1]:
            # contraction point is better than worst point
            del result[-1]
            result.append([x_contraction, cscore])
            continue

        # all above failed, hence shrink all other points except the smallest one
        x1 = result[0][0]
        new_result = []
        for tup in result:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            new_result.append([redx, score])
        result = new_result


# define function to minimize
f = lambda x: x[0]**2 - 4*x[0] + x[1]**2 - x[1] - x[0]*x[1]
# prepare inputs
inputs = np.array([[0, 0], [1.2, 0], [0, 0.8]])

# call nelder mead method
result = nelder_mead(f, inputs, 10)

# print result
print('Terminating the iteration with the following vectors')
for vector in result:
    print(vector[0])

print(f"Smallest value is {f(result[0][0])} with vector {result[0][0]}")
