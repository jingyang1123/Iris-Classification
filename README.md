# Iris-Classification
A neural network model is constructed to classfy iris. Three different optimization methods: gradient descent, Newton's method and BFGS are implemented to train the model.
The conventional way to implement Newton's method requires to compute the Hessian matrix and solve a system of linear equations at each iteration. This large computation cumbers the rate of convergence regarding time cost. The mechanism of Newton's method also makes this algorithm not sufficiently effective on non-smooth function. The purposes of this study are: (1) to implement Newton's method by a matrix-free approach and compare its performance with gradient descent and BFGS. (2) to investigate the influence of smoothing process on the optimization.
