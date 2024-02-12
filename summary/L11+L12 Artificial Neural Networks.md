## Summary
1. [[L11+L12 Artificial Neural Networks#^8d10f9|architecture of simple neuron]]
2. [[L11+L12 Artificial Neural Networks#^3500dd|activation functions]]
3. [[L11+L12 Artificial Neural Networks#^5854c5|Multi-layer neural networks]]
	1. architecture
	2. advantages
4. [[L11+L12 Artificial Neural Networks#^1dfc1a|forward propagation]]
5. [[L11+L12 Artificial Neural Networks#^dc65b4|backpropagation]]
	1. delta and delta rule
	2. [[L11+L12 Artificial Neural Networks#^e7e37a|characters of backpropagation]]
6. [[L11+L12 Artificial Neural Networks#^7a77c2|optimization]]
	1. [[L11+L12 Artificial Neural Networks#^2653be|SGD vs. GD]] and [[L11+L12 Artificial Neural Networks#^c3d6ff|mini batch GD]]
	2. [[L11+L12 Artificial Neural Networks#^17d7b5|annealing learning rate]]
	3. [[L11+L12 Artificial Neural Networks#^9d8606|Momentum]]
7. [[L11+L12 Artificial Neural Networks#^4e6c23|trick of trade in NN]]
8. [[L11+L12 Artificial Neural Networks#^9bf7f3|hyper parameters]]
## Exercise
1. whole E10
### Content
1. Simple Neuron ^8d10f9
	1. Input activation: $a(x)=w_0+\sum_{i=1}^n w_i x_i=w_0+w^{\top} x$
	2. output activation: $h(x)=\theta(a(x))=\theta\left(w_0+w^{\top} x\right)$
	3. Regression
		1. linear regression: 
			1. activation function: $\theta(a) = a$
			2. Goal: learn unknow functon$f: R^n \Longrightarrow R$
			3. cost function: $J\left(w_0, w\right)=\frac{1}{2 m} \sum_{t=1}^m\left(h\left(x_t\right)-y_t\right)^2$
		2. logisitic regression:
			1. activation function: sigmoid function $\theta(a)=1 /(1+\exp (-a))$
			2. Goal: learn unknown function $f: R^n \Longrightarrow {0,1}$
			3. cost function: $J\left(w_0, w\right)=-\frac{1}{m} \sum_{t=1}^m y_t \log \left(h\left(x_t\right)\right)+\left(1-y_t\right) \log \left(1-h\left(x_t\right)\right)$
2. Activation function $\theta(a)$[[activation function(激活函数)]]
	1. "transfer function", transforms input activation to output activation, is non-decreasing
	2. common activation functions: ^3500dd
		1. linear function: $\theta(a) = a$
		2. rectified linear function: $\theta(a) = max(0,a)$
		3. sigmoid function: $\begin{aligned} \theta(a) & =\frac{1}{1+\exp (-a)} \\ \theta^{\prime}(a) & =\theta(a)(1-\theta(a))\end{aligned}$
		4. hyperbolic tangent function: $\begin{aligned} \theta(a) & =\frac{\exp (a)-\exp (-a)}{\exp (a)+\exp (-a)} \\ \theta^{\prime}(a) & =1-\theta^2(a)\end{aligned}$
3. One layer networks
	![[截屏2024-02-10 12.13.17.png]]
	1. output: 
		1. j-th output neuron: $h_j=\theta\left(a_j\right)=\theta\left(w_j^T x\right)$
		2. output vector: $h=\theta(W x)$
		3. weight matrix: $$W=\left(\begin{array}{llll}w_{10} & w_{11} & w_{12} & w_{13} \\w_{20} & w_{21} & w_{22} & w_{23} \\w_{30} & w_{31} & w_{32} & w_{33}\end{array}\right)$$row $w_j$ corresponds to weights of $h_j$
	2. softmax regression: 
		1. softmax activation: $\theta\left(a_j\right)=\exp \left(a_j\right) / \sum_k \exp \left(a_k\right)$
		2. Goal: learn unknown function $f: R^n \Longrightarrow {1,...,c}$
		3. cost function: $\begin{aligned} J(W) & =-\frac{1}{m} \sum_{t=1} \sum_{j=1}^c \mathbb{I}\left\{y_t=j\right\} \log h_j\left(x_t\right) \\ & =-\frac{1}{m} \sum_{t=1}^m \sum_{j=1}^c \tilde{y}_{t, j} \log h_j\left(x_t\right)\end{aligned}$
1. Multi-Layer neural networks: ^5854c5
	1. Motivation:
		1. to solve non-linear problem
		2. better representation of input data
	2. Architecture:![[截屏2024-02-10 12.19.37.png]]
		1. fully connected feed forward network
			1. Neuron $j$ of layer $l>1$ receives a weighted input from
				- every neuron $i$ of layer $l-1$ with weight $w_{j i}^{(l)}$
				- bias weighted by $w_{j 0}^{(l)}$
			2. Design decisions:
				1. number of hidden layers
				2. number of neurons in each hidden layer
				3. activation functions
				4. loss function
			3. design decisions
				1. can be critical
				2. are largely problem and data dependent
				3. therefore there is no foolproof recipe
1. Forward Propagation: ^1dfc1a
	1. Goal: compute output of network(network function)
	2. procedure; compute output of each neuron layer-wise from lowest to highest layer![[截屏2024-02-10 12.23.42.png]]
	3. **universal approaxiamtion theorem**: A 2-layer network with linear output neurons can approximate any continuous function arbitrarily well, given enough hidden units
		1. theorem holds for sigmoid, tanh and other hidden non-linear activation functions
		2. theorem doesn't mean that we can find the necessary parameter values
2. ==Backpropagation== ^dc65b4
	1. hypothesis space
		1. hyper-parameters:
			1. number of layers
			2. number of neurons per layer
			3. activation functions
		2. parameters to be learned: $W=\left(W^{(1)}, \ldots, W^{(L)}\right) \in \Lambda$
		3. hypothesis space: $\mathcal{H}=\left\{h_w: \mathbb{R}^n \rightarrow \mathbb{R}^c, W \in \Lambda\right\}$
		4. Goal: learn unknown function $f: R^n \longrightarrow R^c$ by minimizing: $J(W)=\frac{1}{m} \sum_{t=1}^m \ell\left(h_W\left(x_t\right), y_t\right)$
			1. function covers regression problems and classification problems
			2. gradient descent: $w_{j i}^{(l)} \leftarrow w_{j i}^{(l)}-\eta \cdot \frac{\partial J(W)}{\partial w_{j i}^{(l)}}$
			3. decomposition: $\begin{aligned} J(W) & =\frac{1}{m} \sum_{t=1}^m J_t(W) \\ J_t(W) & =\ell\left(h_W\left(x_t\right), y_t\right)\end{aligned}$
			4. partial derivatives: $\frac{\partial J(W)}{\partial w_{j i}^{(l)}}=\frac{1}{m} \sum_{t=1}^m \frac{\partial J_t(W)}{\partial w_{j i}^{(l)}} ; \frac{\partial J_t}{\partial w_{j i}^{(l)}}=\frac{\partial J_t}{\partial a_j^{(l)}} \cdot \frac{\partial a_j^{(l)}}{\partial w_{j i}^{(l)}}$
			5. input activation: $a_j^{(l)}=\sum_i w_{j i}^{(l)} h_i^{(l-1)}$
			6. ==delta==: $\delta_j^{(l)}=\frac{\partial J_t}{\partial a_j^{(l)}}$
			7. ==delta rule==: $\frac{\partial J_t}{\partial w_{j i}^{(l)}}=\delta_j^{(l)} \cdot h_i^{(l-1)}$
			8. ==delta==: $\delta_j^{(l)}=\left\{\begin{array}{cl}\tilde{\ell^{\prime}}\left(h_j^{(l)}\right) \cdot \theta^{\prime}\left(a_j^{(l)}\right) & : \quad l=L \\ \theta^{\prime}\left(a_j^{(l)}\right) \cdot \sum_k w_{k j}^{(l+1)} \delta_k^{(l+1)} & : \quad l<L\end{array}\right.$
	![[截屏2024-02-10 12.53.36.png]]
	2. Algorithm
		backpropagation can be very slow particularly for multilayered networks ^e7e37a
		1. the risk function is typically
			1. non-quadratic
			2. non-convex
			3. high dimensional
			4. has many local minima and/or flat regions
		2. there is no guarantee that
			1. network will converge to a good solution
			2. convergence is swift
			3. convergence even occurs at all
		3. tricks of the trade
			1. may guide practitioners to make better design decisions
			2. may help to improve the chances of finding a good solution
			3. may help to decrease the convergence time
1. optimization ^7a77c2
	1. given:
		1. large dataset of training examples
		2.  hypothesis space $\mathcal{H}=\left\{h_W: \mathbb{R}^n \rightarrow \mathbb{R}^c \mid W \in \Lambda\right\}$
		3. differentiable or convex loss function $\ell\left(h_W(\mathrm{x}), \mathrm{y}\right)$
	2. Goal: minimize cost function $J(W)=\frac{1}{m} \sum_{i=1}^m \ell\left(h_W\left(x_i\right), y_i\right)=\frac{1}{m} \sum_{i=1}^m J_i(W)$
	3. Approaches:[[Gradient Decent]]
		1. Batch gradient descent(GD)![[截屏2024-02-10 13.31.10.png]]
			1. $\nabla J(W)=\frac{1}{m} \sum_{i=1}^m \nabla J_i(W)$
			2. an epoch is a full pass through the training set
				1. one epoch = processing m training examples = one weight update
			3. Pro and Cons:
				1. Pros:
					2. wealth of optimization techniques
						1. accelerate gradient
						2. conjuge gradient
						3. quasi-Newton
						4. inexact Newton methods
					3. efficient implementations
						1. matrix-matrix multiplications
						2. parallel and distributed processing
					4. convergence analysis
						1. conditions of optimality
						2. convergence rate
				2. Cons:
					1. local solution: can converge to local minima for nonconvex problems
					2. inefficient: slow convergence in terms of cost versus arithmetric operations
					3. intractable: computationally intractable if data does not fit into memory
					4. batch only: no online/streaming updating
		2. stochastic gradient descent(SGD)![[截屏2024-02-10 13.41.15.png]]
			1. $\nabla J(W)=\frac{1}{m} \sum_{i=1}^m \nabla J_i(W)$
			2. an epoch is a full pass through the training set
				1. processing m training examples = m weight update
			3. ==variations==
			![[截屏2024-02-10 14.03.17.png]]
		3. Mini-batch gradient descent
			![[截屏2024-02-10 14.05.43.png]] ^c3d6ff
			1. b=1 $\Longrightarrow$ SGD
			2. b=m $\Longrightarrow$ GD
	1. annealing the learning rate $\eta$ ^17d7b5
		1. annealing schedule
			1. innitialize $\eta \leftarrow \eta_0$
			2. anneal $\eta \leftarrow A(\eta, \eta_0, t)$ $\Longrightarrow$ reduces $\eta$, $t$ = number of updates/epochs
		2. constant learning rate : $A\left(\eta, \eta_0, t\right)=\eta$
		3. step decay: $A\left(\eta, \eta_0, t\right)= \begin{cases}\alpha \cdot \eta & t \bmod T=0 \\ \eta & \text { otherwise }\end{cases}$
		4. exponential decay $A\left(\eta, \eta_0, t\right)=\eta_0 \exp (-\alpha \cdot t)$
		5. 1/t decay: $A\left(\eta, \eta_0, t\right)=\frac{\eta_0}{1+\alpha t}$
	2. Momentum
		1. $-\nabla J(W)$
			1. points to steepest descent at W
			2. does not necessarily point to local minimum
		2. SGD in regions with pathological curvature
			1. tends to oscillate
			2. slow convergence behavior
		3. Momentum aims at dampening oscillations and accelerating SGD
			1. SGD: $W \leftarrow W-\eta \nabla J(W)$
			2. Momentum: $\begin{aligned} V & \leftarrow \gamma V+\eta \nabla J(W) \\ W & \leftarrow W-V\end{aligned}$ ^9d8606
				1. $V$ = velocity; initial value =0
				2. $\gamma$ = momentum
					1. typically, $\gamma \in[0.5,0.99]$
					2. default value: $\gamma = 0.9$
					3. constant/ anneal
	3. further SGD techniques: Nesterov momentum; agagrad; adadelta; adam: RMSprop; Adamax; Nadam
1. Tricks of the Trade ^4e6c23
	1. Data processing
		1. normalizatior: $\tilde{x}_{i j}=\frac{x_{i j}-\mu_j}{\sigma_j}$
	2. weight initialization
	3. batch normalization
	4. regularization
		1. cost function with $L_p$ regularization： $\mathrm{J}(\mathrm{W}, \lambda)=J(W)+\frac{\lambda}{p m} \sum_{w \in W}|| w \|_p^p$
		2. update rules
			1. $L_1$ reugularization: $w \leftarrow w-\eta\left(\frac{\partial}{\partial w} J(W)+\frac{\lambda}{m}(\mathbb{I}\{w \geq 0\}-\mathbb{I}\{w<0\})\right)$
			2. $L_2$ reugularization: $w \leftarrow w-\eta\left(\frac{\partial}{\partial w} J(W)+\frac{\lambda}{m} w\right)$
		3. further methods:
			1. elastic net: combines $L_1 \text{and} L_2$ regularization
			2. Early stopping
				1. stop training when error on validation set increase or stagnates
				2. prevents overfitting
			3. Dropout
				1. training:(for mini-batch size b=1)
					1. steps
						1. consider base network
						2. select next training example x
							1. delecte hidden units with probability p
							2. forward propagate x through thinned network
							3. back propagate error through thinned network
							4. update weights of thinned network
						3. go to step 1
					2. samples from a collection of $2^n$ thinned networks(n = number of neurons)
					3. weights are shared within the collection
				2. prediction
					1. steps
						1. consider base network
						2. multiply all weights by p
						3. select next test example x
							1. forward propagate x
							2. report output
					2. whinned network are combined(similar to ensemble methods)
					3. $W \rightarrow pW$ ensures that the output at prediction time is same as the expected output at training time
					4. it suits as an approximation to averaging the $2^n$ thinned networks
				3. regularization: dropout reduces model complexity
			4. Batch Normalization
2. hyper-parameters ^9bf7f3
	1. learning rate
	2. mini-batch size
	3. number of epochs
	4. momentum
	5. regularization parameters(dropout, weight decay)
	6. activation functions
	7. number of layers
	8. number of neurons per layer

4. SGD vs. GD

| D | SGD | GD |
| ---- | ---- | ---- |
| updates of W in one epoch | m | 1 |
|  | computationally demanding | efficient matrix-matrix multiplication |
|  | use $\nabla J_i$ as (rough) approximation of $\nabla J$<br>$\Longrightarrow$ high variance<br>$\Longrightarrow$ $\nabla J(w)$ fluctuates near local minimum | gradient $\nabla J$ decreases $\nabla J(w)$ if $\eta$ is small<br>$\Longrightarrow$ convergence to local minimum |
|  | amenable to real-time operation | amenable to parallel implementations |
|  | pro: fast convergence<br>con: slow iterative computing | pro: parallelization<br>con: slow convergence |

^2653be

