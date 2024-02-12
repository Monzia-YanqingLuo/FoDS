#machine_learning 

| D    | D                                                                                                                                                                                                    |     |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| What | Mean Square Error. This method is to find the minimum MSE cost to get the best $f$(model), based on [[loss \| squared loss]]                                                                         |     |
| How  | $E_m[h]=\frac{1}{m} \sum_{i=1}^m \ell\left(h\left(x_i\right), y_i\right)=\frac{1}{2 m} \sum_{i=1}^m\left(h\left(x_i\right)-y_i\right)^2<br>$                                                         |     |
| When | Curve fitting, Empirical Risk Minimization                                                                                                                                                           |     |
| Why  | **Emphasizes Larger Errors**<br>**Differentiability**: MSE is a smooth and differentiable function, which means it can be easily optimized using gradient descent and other optimization algorithms.<br>**Analytical Simplicity**<br>**Compatibility with Linear Models**: MSE has a direct mathematical relationship with the Gaussian distribution |     |
|      |                                                                                                                                                                                                      |     |


