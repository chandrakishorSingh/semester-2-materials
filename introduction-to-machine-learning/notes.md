1. Bias

- measure of how well a model fits a given dataset
- measured in terms of squares of errors
- high bias model fits the given dataset poorly while low bias model passes quite nearly through data points 
- usually a complex model(which fits the data points quite good) has a low bias while simple model has high bias

2. variance

- measure of how much the model varies in terms of bias across various dataset.
- a complex model usually has high variance because it fits one dataset quite good(usually on the training dataset) while on other it may not
- a simple model usually has low variance because unlike a complex model, it is not trained to fit the training dataset very well. its performance remains quite consistent across various datasets.

3. bias vs variance

- usually we want a model to have low variance and low bias.
- some of the techniques used for finding models with low bias and low variance are regularization, boosting and bagging.

4. overfit

- when a model fits training data very well but not testing data then it is said to be overfit

5. cross validation

- it can help in choosing which ML algorithm would be better for the task at hand.
- the idea is to compare how well ML algorithms perform on the same dataset.
- a portion of the entire dataset is taken for testing while remainning dataset is used for training the model.
- but which portion of the data should we train our model(or equivalently, test our model) ?
- in such case, we divide the data into, say 4, consecutive blocks. use 1st block for testing and remainning 3 blocks for training.
- next time we would use 2nd block for testing and all other for training. and so on. at the same time we would track the performance of the ML models on these dataset splits.
- finally, we would compare the performance of all ML models across all combinations of dataset splits.
- the one with best performance would be chosen.
- splitting the data into 4 blocks is called Four-Fold cross validation.
- in extreme case when only one sample is used for testing, it is called Leave One Out Cross Validation.
- in practice, usually data is divided into 10 blocks which is called Ten-Fold Cross Validation.

6. Probability vs Likelihood

- probability: it is area under the given distribution for some range of values in the x axis. it is the probability of particular "data" for a given distribution.
- likelihood: it is the value of y axis of the point which is present on the distribution for a particular value of x. It is the probability of the "distribution" for a particular value of the data.
