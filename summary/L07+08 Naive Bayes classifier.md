## Summary
1. [[L07+08 Naive Bayes classifier#^c61ae6|definition]]
2. [[L07+08 Naive Bayes classifier#^5f177c|Bernouli Naive Bayes]]
3. [[L07+08 Naive Bayes classifier#^95d7fc|Laplace smoothing]]
4. [[L07+08 Naive Bayes classifier#^5fb1d7|Multinomial Naive Bayes classifier]]
5. [[L07+08 Naive Bayes classifier#^d952ab|Gaussian Naive Bayes classifier]]
6. [[L07+08 Naive Bayes classifier#^fe921b|Evaluation metrics in 2 category]]
7. [[L07+08 Naive Bayes classifier#^dad035| EM in multi-category]]
## Exercise
1. why the Laplace smoothing is important
2. three important part of evaluation metrics and the relation
### Content
1. why does we use naive bayes
	1. Estimates $P(x, y)=P(x \mid y) P(y)$ -->联合概率密度的定义
	2. Reduces number of parameters $P\left(x_j \mid y_k\right)$ to be estimated from $c 2^n$ to $c n$
	3. Assumes conditional independence
		1. Features $x$ and $x^{\prime}$ are conditionally independent given label $y$ if$$P\left(x, x^{\prime} \mid y\right)=P(x \mid y) P\left(x^{\prime} \mid y\right)$$
2. What ^c61ae6
	1. Naive Bayes Assumption: $$p(x \mid y)=\prod_{j=1}^n p\left(x_j \mid y\right)$$
	2. Naive Bayes classifier:==The goal of bayes classifier is to maximize the posterior== $$\begin{aligned} & h(x)=\underset{k \in \mathcal{Y}}{\operatorname{argmax}} P\left(y_k\right) \prod_{j=1}^n p\left(x_j \mid y_k\right) \\ & h(x)=\underset{k \in \mathcal{Y}}{\operatorname{argmax}} \log P\left(y_k\right)+\sum_{j=1}^n \log p\left(x_j \mid y_k\right)\end{aligned}$$
	3. Bernouli Naive Bayes:
		1. Likelihood of $x$ given label $y_k$$$p\left(x \mid y_k\right)=\prod_{j=1}^n p_{k j}^{x_j} \cdot\left(1-p_{k j}\right)^{\left(1-x_j\right)}$$
		2. Bernoulli Naive Bayes classifier: $$h(x)=\underset{k \in \mathcal{Y}}{\operatorname{argmax}} P\left(y_k\right) \prod_{j=1}^n p_{k j}^{x_j}\left(1-p_{k j}\right)^{\left(1-x_j\right)}$$ ^5f177c
			1. Estimate of priors: $\hat{\pi}_k=\frac{\sum_{i=1}^m \mathbb{I}\left\{y_i=k\right\}}{m}=\frac{m_k}{m}$   --> $\begin{aligned}& \text { \# examples with label } y_k \\& \hline \text { \# all examples }\end{aligned}$
			2. Estimate of likelihood: $\hat{p}_{k j}=\frac{\sum_{i=1}^m \mathbb{I}\left\{y_i=k\right\} x_{i j}}{m_k}=\frac{m_{k j}}{m_k}$ --> $\begin{aligned}& \text { \# examples of features}  x_j=1 \text{in class} y_k \\& \hline \text { \# examples with label }y_k \end{aligned}$
			3. Prediction: $$\hat{y} \in \underset{k=1}{\operatorname{argmax}} \frac{m_k}{m} \prod_{j=1}^n\left(\frac{m_{k j}}{m_k}\right)^{x_j}\left(1-\frac{m_{k j}}{m_k}\right)^{1-x_j}$$
3. Laplace Smoothing ^95d7fc
	1. To solve zero-count problem: statistically a bad idea to estimate the probability of an event to be zero just because it is not contained in the finite training set.
	2. Estimate likelihoods with Laplace smoothing： $$\hat{p}_{k j}=\frac{m_{k j}+\alpha}{m_k+\alpha n}$$
		1. $\alpha \geq 0$ is smoothing parameter
		2. $\alpha=0$ no smoothing
		3. $\alpha=1$ add-one smoothing
4. Multinomial Naive Bayes Classifier
	1. multinomial distribution: $$P(x)=\frac{\left(x_1+\cdots+x_n\right) !}{x_{1} ! \cdots x_{n} !} \cdot p_1^{x_1} \cdots p_n^{x_n}$$
	2. Likelihood of x given label k:$$P\left(x \mid y_k\right)=\frac{\left(\sum_j x_j\right) !}{\prod_j x_{j} !} \prod_j p_{k j}^{x_j}$$
		1. $p_{k j}^{x_j}$ : probability of event j given label k
		2. $\frac{\left(\sum_j x_j\right) !}{\prod_j x_{j} !}$: varies for different x but is constant over all labels k
	3. ==Multinomial Naive Beyes classifier==:$$h(x)=\underset{k \in \mathcal{Y}}{\operatorname{argmax}} P\left(y_k\right) \prod_j p_{k j}^{x_j}$$ ^5fb1d7
	4. After laplace smoothing(Prediction):$$\hat{y} \in \underset{k=1}{\operatorname{argmax}} \frac{m_k}{m} \prod_{j=1}^n\left(\frac{n_{k j}+\alpha}{n_k+\alpha n}\right)^{x_j}$$
5. Gaussian Naive Bayes:
	1. Gaussian distribution:$$\mathcal{N}\left(\mu, \sigma^2\right)=\frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)$$
	2. Likelihood of $x_j$ given label $y_k$: $$p\left(x_j \mid y_k\right)=\mathcal{N}\left(\mu_{k j}, \sigma_{k j}^2\right)$$
	3. Gaussian naive bayes classifier: $$h(x)=\underset{k \in \mathcal{Y}}{\operatorname{argmax}} P\left(y_k\right) \prod_{j=1}^n \mathcal{N}\left(\mu_{k j}, \sigma_{k j}^2\right)$$ ^d952ab
	4. $$\widehat{y} \in \underset{k \in \mathcal{Y}}{\operatorname{argmax}} \widehat{\pi}_k \prod_{j=1}^n \mathcal{N}\left(\widehat{\mu}_{k j}, \widehat{\sigma}_{k j}^2\right)$$
6. Summary of naive Bayes
	1. Advantages:
		1. extremely fast for both training and prediction
		2. straightforward probabilistic prediction
		3. often easily interpretable
		4. few(if any) hyper-parameters
	2. when to use it
		1. as a baseline
		2. when naive assumption matches the data
		3. when model complexity is less important(well seperated categories, high-dimensional data)
	3. Bayes rule forms the basis for designing learning algorithms
		1. Goal: estimate unknown function $f(x)=P(y \mid x)$
		2. Learning: use training examples to estimate $p(x, y)=p(x \mid y) P(y)$
		3. Prediction: use estimated probabilities plus Bayes rule to classify a new input $x$
		$\Rightarrow$ Generative classifier
	4. Naïve Bayes assumption to overcome the intractability of a Bayes classifier
		1. Bayes classifier: unrealistic number of training examples(without futher assumptions)
		2. Naive Bayes: 
			1. assumes all features are conditionally independent
			2. drastically reduces number of parameters that need to be estimated
1. Evaluation Metrics
	1. Common evaluation metric
		1. $\text { error rate }=\frac{\text { number of misclassified test examples }}{\text { number of all test examples }}$
		2. Accuracy = 1- error rate
		3. Problem: unbalanced class distribution
	2. Evaluation metric - two-category case![[截屏2024-02-06 17.02.29.png]] ^fe921b
		1. $\text { accuracy }  =\frac{tp+t n}{t p+f p+t n+f n}$
		2. $\text { precision }  =\frac{t p}{t p+f p}$ 所有被识别为positive的例子中正例的比例
		3. $\text { recall }  =\frac{t p}{t p+f n}$ 被识别为positive占所有正例的比例
			1. precision and recall only focus on the positive situation
	3. $$\text { F-measure }=\left(1+\beta^2\right) \cdot \frac{\text { precesion } \cdot \text { recall }}{\beta^2 \cdot \text { precision }+ \text { recall }}$$
		1. combines precision and recall
		2. $\beta$ weights importance of precision
		3. $\beta >1$ precision is more important
		4. $\beta <1$ recall is more important
		5. $\beta =1$ standard choice(F1-score) 
2. Evaluation Metrics - Multi-category case ^dad035
	1. one vs. all: reduce to c two-category cases --> average metrics over all cases
	2. Averaging methods:
		1. Macroaveraging
			1. compute metric for each class
			2. average over classes
		2. Microaveraging:
			1. collect decisions for all classes in a single contingent table
			2. derive metrics from polled contingent table