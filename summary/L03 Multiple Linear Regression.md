## Summary
1. [[L03 Multiple Linear Regression#^f3a87e|Augmentation]]
2. [[L03 Multiple Linear Regression#^48749e|linear basis function model]]:
	1. [[L03 Multiple Linear Regression#^7eadf9|the advantages compared to multiple linear regression]]
	2. [[L03 Multiple Linear Regression#^5b12e9|basic basis function]]
3. [[L03 Multiple Linear Regression#^facfc0|overfitting]] (variance)<--> connected to underfitting(bias)
	1. why and hot to solve this
4. [[L03 Multiple Linear Regression#^de7b14|regularization]]
	1. why: to control the model complexity --> control the number of weight (polynomial degree)
	2. $L_1, L_2$ regularization
	3. gradient descent
5. MSE in matrix formular and closed form solution:
	1. condition: peudoinverse exits <-- $X^TX$ is not singular, columns of X are linearly independent
6. Feature normalization(data normalization)
	1. original data
		1. aig-zagging of GD(SGD)
		2. small learning rate required
		3. slow convergence
	2. normalized data:
		1. gradient points better to min
		2. larger learning rate
		3. fast convergence
## Exercise
1. Implement the gradient descent method for simple linear regression in one variable.
	1. initialize parameters: weights and bias
	2. hypothesis function(simple regression)
	3. cost function(empirical error based on MSE)
	4. gradient descent
	5. update rules
2. determine suitable learning rate
	1. randomly chose a list of learning rate
	2. compare the MSE over iterations
3. the learning rate and gradient descent
	1. small learning rate: result in small updates to the weights,which makes the algorithm more liable in terms ofsteadily moving towards the min. Hoever, it's too slow to convergence.
	2. Large leraning rate: accelerate the convergence, fast. however, it can cause the algorithm overshoot the min, leading to divergence.
4. the rank to constraint solution:
   Let $X \in \mathbb{R}^{m \times n+1}$ be an augmented data matrix and $y \in \mathbb{R}^m$ the output vector.
- Show that Multiple Linear Regression has a unique solution if $\operatorname{rank}(X)=n+1$.
- State a necessary condition on $m$ for having a unique solution.
- What if $\operatorname{rank}(X)<n+1$ ? Does a solution exist? Is the solution unique? Does it depend on $y$ ? Prove your claims.
### Content 
1. Multiple linear regression
	1. polynomial(linear base)
	2. ==Augmentation==: change of notation to avoid case distinction between weights and bias ^f3a87e
		1. Original function($w_0$ is bias): $h_w(x)=w_0+w_1 x_1+w_2 x_2+\cdots+w_n x_n$
		2. augmented function($x_0 =1$): $h_w(x)=w_0 x_0+w_1 x_1+\cdots+w_n x_n=w^T x$
		3. augmented data matrix: $X=\left[\begin{array}{cccc}1 & x_{1,1} & \cdots & x_{1, n} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{m, 1} & \cdots & x_{m, n}\end{array}\right] \in \mathbb{R}^{m \times n+1}$
	3. Minimize MSE: $J(w)=\frac{1}{2 m} \sum_{i=1}^m\left(h_w\left(x_i\right)-y_i\right)^2=\frac{1}{2 m}(X w-Y)^{\top}(X w-Y)$
		--> Solution: $w=\left(X^{\top} X\right)^{-1} X^{\top} Y \quad$ (if $X^T X$ is non-singular)
			when it's singular
	1. Gradient Descent: $w \leftarrow w-\frac{\eta}{m} X^{\top}(X w-Y)$

1. Linear Basis Function Models ^48749e
	1. $h_w(x)=w_0 \phi_0(x)+w_1 \phi_1(x)+\cdots+w_k \phi_k(x)=w^{\top} \phi(x)$
		1. $\phi_0(x) \equiv 1 \quad$ dummy basis function
	2. Compared to Multiple linear regression, it can apply to non-linear problem, because of the base can be non-linear.  ^7eadf9
		1. "Linear" is because of the sum and the matrix notation
		2. every term has bias
		3. Basic biasis functions: ^5b12e9
			1. Projection: Linear function $\phi_j: \mathbb{R}^n \rightarrow \mathbb{R}, \quad x \mapsto x_j$
			2. Prepocessing functions:$\phi_j(x)=\frac{x_j-\mu_j}{\sigma_j}$ , $\mu_j=\frac{1}{m} \sum_{i=1}^m x_{i j} \quad, \quad \sigma_j=\sqrt{\frac{1}{m-1} \sum_{i=1}^m\left(x_{i j}-\mu_j\right)^2}$
				1. feature normalization, can improve convergence of gradient descent mothods
		4. Polynomial Basis functions $\phi_j: \mathbb{R} \rightarrow \mathbb{R}, \quad x \mapsto x^j$
		5. Gaussian Basis functions: $\phi_j: \mathbb{R}^n \rightarrow \mathbb{R}, \quad x \mapsto \exp \left(-\frac{\left\|x-\mu_j\right\|^2}{\sigma_j}\right)$
	3. MSE cost function:$J(w)=\frac{1}{2 m} \sum_{i=1}^m\left(h_w\left(x_i\right)-y_i\right)^2=\frac{1}{2 m}(\Phi w-Y)^{\top}(\Phi w-Y)$
	4. Solution: $w=\left(\Phi^{\top} \Phi\right)^{-1} \Phi^{\top} Y$
	5. Gradient Descent: $w \leftarrow w-\frac{\eta}{m} \Phi^{\top}(\Phi w-Y)$

3. Overfitting ^facfc0
	1. a low MSE does not guarantee a good estimate of function$f$
	2. Overfitting:
		1. low MSE but poor estimate of unknown function $f$
		2. occurs when hypothesis space is too complex for the data
		3. learns noise
	3. Solution:
		1. increse number $m$ of training samples(not always possible)
		2. Regularization(to control the complexity)

4. Regularization
	1. How to control the complexity
		1. 1-norm: use the absolute value of weight
		2. 2-norm: square of weight
	2. Regularized empirical risk:$$
E_m\left[h_w, \lambda\right]=E_m\left[h_w\right]+\lambda \Omega\left(h_w\right)
$$$$
E_m\left[h_w, \lambda\right]=\frac{1}{2 m} \sum_{i=1}^m\left(h_w\left(x_i\right)-y_i\right)^2+\frac{\lambda}{2 m} \sum_{j=1}^k w_j^2
$$
	==Remark==
		- often no regularization of bias $w_0$
		- $\lambda$ need to be carefully selected
	- **Regularization Parameter $\lambda$**: controls relative importance of regularization term and empirical risk
	- **Regularization term $\lambda \Omega\left(h_w\right)$**: discourages parameters $w_j$ from reaching large values
	3. Gradient Descent for L2-regularizaed MSE
		1. $$
w_k \leftarrow w_k-\eta\left(\frac{1}{m} \sum_{i=1}^m\left(h_w\left(x_i\right)-y_i\right) x_{i k}+\frac{\lambda}{m} w_k\right)
$$$$
w_0 \leftarrow w_0-\eta \frac{1}{m} \sum_{i=1}^m\left(h_w\left(x_i\right)-y_i\right) x_{i 0}
$$ ^de7b14
- Sufficient condition for local min: Hessian matrix is pos. def. at w.
$$
\nabla^2 \partial(w)=\frac{\partial}{\partial w}\left(\frac{1}{m} x^{\top} x w-x^{\top} Y\right)=\frac{1}{m} x^{\top} x \text {. }
$$
$X^{\top} X$ is pos. def. because for all $V \in \mathbb{R}^{n+1}, V \neq 0$ we have
$$
v^{\top} X^{\top} X_v=\left(X_v\right)^{\top}\left(X_v\right)=\|X v\|^2>0 \text { (*) } \operatorname{ramk}(X)=n+1
$$
$\Rightarrow \partial$ is locally convex every where
(the $\operatorname{ker}(x)=\{0\}$ )
$\Rightarrow$ any stationary point is a local min
$\Rightarrow \quad \omega=\left(x^{\top} X\right)^{-1} X^{\top} Y$ is the unique global min
