#machine_learning 

Squared loss is common choice for curve fitting $$
\ell(h(x), y)=\frac{1}{2}(h(x)-y)^2
$$
squared loss induces mean squared error (MSE) cost
$$
E_m[h]=\frac{1}{m} \sum_{i=1}^m \ell\left(h\left(x_i\right), y_i\right)=\frac{1}{2 m} \sum_{i=1}^m\left(h\left(x_i\right)-y_i\right)^2
$$
