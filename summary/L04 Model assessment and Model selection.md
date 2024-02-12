## Summary
1. [[L04 Model assessment and Model selection#^2d037e|model selection contains model assessment]]
2. U-shape of training error: overfitting
3. [[L04 Model assessment and Model selection#^2d037e|model selection]]
	1. find the best hyperparameters -- validation set
	2. find the best hypothesis(weights, functions...) -- train+val dataset
	3. compute the test error -- test dataset
4. [[L04 Model assessment and Model selection#^977dde|k-folder cross validation]]: Hold-out-valiadation(split the data into three sets)
	1. when: small training dataset size
	2. [[L04 Model assessment and Model selection#^977dde|Bias and variance tradeoff in cross validation]]
	3. Nested cross-validation: split training folders to do CV(mix the data and split again)

## Exercise
1. How to do nested CV
### Content
1. Definition ^2d037e
	1. Model assessment: estimate performance of a given hypothesis $h$
	2. Model selection: 
		1. asses performance of different hypothesis
		2. choose the best hypothesis

2. Model Assessment
	1. Definition: estimate performance of a given hypothesis $h$
		2. estimate generalization error of a given hypothesis $h$
		3. use test error on independent set of input-output examples
	2. Performance measures
		1. training 
			training set $\mathcal{D}_{\text {train }}=\left(x_1, y_1\right),\left(x_2, y_2\right), \ldots,\left(x_m, y_m\right)$ empirical risk $E_m[h]$ of hypothesis $h$ on training set $\mathcal{D}_{\text {train }}$
		1. generalization error
			probability distribution $P(x, y)$ on input-output space $\mathbb{R}^n \times \mathbb{R}$
			expected risk $\mathrm{E}[h]$ wrt. $P(x, y)$
			--> use the empirical error on test dataset to determie the test error
	1. problem:
		1. $P(x,y)$ is unknwon
		-> generalization error can not be determined
	==A low training error does not necessarily guarantee a low generalization error==
	4. general observation:
		complexity of $H$ increases, train error decreases, U-shape of test error

3. Model selection
	the selection contains model selection and hyper parameters selection
	1. Hyperparameters: used to adapt $\mathcal{H}_\theta$ (or learner) to a specific problem class. Usually set before training begins and fixed during training
		- Can be found using Model Selection
	2. Model parameters: learnable parameters (weights), set during training
	3. Hold-out-validation: randomly split data into three sets
		1. training set: 60-80%
		2. validation set: 10-20%
		3. test set: 10-20%
![[截屏2024-02-02 20.23.56.png]]

4. k-fold cross validation
	1. model accessment
		1. Given: small dataset
		2. problem: 
			1. train-test/ train-val-test split wasts data for fitting model -->underfitting problem
			2. test error may highly vary depending on particular split
		3. common choices for K:
			1. K=5 and K=10
			2. leave one out validation: K=r(the number of sample in D)
			==Bias and Variance trade-off==
				1. the larger K is, the training sets per folder increases -->Model bias decrease
				2. the smaller K is, the dependence of training set increase --> test error has larger variance
![[截屏2024-02-02 20.59.00.png]]![[截屏2024-02-02 20.59.29.png]] ^977dde