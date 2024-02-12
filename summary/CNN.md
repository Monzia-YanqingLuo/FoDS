## Summary
1. [[FoDS/CNN#^33f3c6|shallow networks and deep networks]]
2. [[FoDS/CNN#^fac3e7|architecture of CNN]]
## Exercise
1. How to find the best parameters
### Content
1. deep neural networks ^33f3c6
	1. shallow networks:
		1. implications:
			1. shallow networks can approxiamate discrete functions
			2. shallow networks are able to represent " most" functions
		2. problems:
			1. representating complicated functions may require an exponentially large number of hidden neurons
			2. no guarante that a training algorithm will learn a good approximation of $f$
	2. deep learning assumption: the learning problem consists of discovering a set of underlying factors of variation that can in turn be described in terms of other, simpler underlying factors of variation.
		3. each hidden neuron learns a feature
		4. hidden layer represents features composed of features from previous layers
	3. causes of failure in Deep networks:
		1. lack of labeled data
		2.  many bad local optima
		3. lack of computational resources
		4. vanishing or exploding gradients
			1. unstable gradients: multi-layer perceptrons with many hidden layers suffeer from unstable gradients
			2. vanishing gradients problem: with decreasing layer index, the norm of the gradients tends to decrease (Implication: in first layer there is almost no learning)
			3. exploding gradients problem: with decreasing layer index, the norm of gradients tends to increase (implication: in first layer weight updates are unstable)
			4. why do unstable gradients occur: $$\frac{\partial J}{\partial w_1}=C \prod_{l=2}^L \sigma^{\prime}\left(a_l\right) w_l$$
				1. $\text { Assume }\left|\sigma^{\prime}\left(a_l\right) w_l\right|<\alpha<1 \text { for all } l$ $\Rightarrow\left\|\frac{\partial J}{\partial w_1}\right\|<\mathrm{C} \alpha^{L-1} \rightarrow 0$ as $L$ increases $\quad \rightarrow$ vanishing gradient
				2. $\text { Assume }\left|\sigma^{\prime}\left(a_l\right) w_l\right|>\alpha>1 \text { for all } l$ $\Rightarrow\left\|\frac{\partial J}{\partial w_1}\right\|<\mathrm{C} \alpha^{L-1} \rightarrow \inf$ as $L$ increases $\quad \rightarrow$ exploding gradient
	4. breakthrough in 2006: deep blief networks; autoencoders
		1. key idea: greedy layer-wise unsupervised pretraining on unlabeled data
	5. Succerss factors:
		1. special network architectures:
			1. CNN: convolutional neural networks
			2. LSTM: Long short term memory networks
		2. availability of large datasets
		3. more computational power(GPU)
		4. relu units, dropout, batch normalization
		5. greedy layer-wise unsupervised pretraining with autoencoders
			1. one early approach that succeeded training deep networks
			2. somewhat outdated: on average results can be worse than with other techniques
2. CNN
	1. Convolutional networks:
		1. are a special kind of multi-layer neural networks
		2. incorporate prior domain knowledge into the architecture
		3. are trained with a version of backpropagation algorithmn
		4. were some of the first deep models to perform well
		5. remain at the forefront of commercial applications of deep learning today
	2. 4 key ideas behind CNN:
		1. sparse interactions(connectivity/ weights)
		2. weight sharing
		3. equivariant representations
		4. many layers
	3. Architectures: input layer $\rightarrow$ convolutional layers $\rightarrow$ dense layers $\rightarrow$ output layer ^fac3e7
		1. input layer: pixel depitched by a grid of numbers representing intensity
		2. convolutional layer:
			1. convolution stage:
				1.  problem of fully connected networks
					1. includes no prior domain knowledge; ignores spatial structure of pixels in images
					2. number of weights of $q$ hidden neuron in first layer
						1. input layer consis of $P$ features
						2. there are $q(p+1)$ weights
					3. implications: inefficient, difficult to train
				2. convolutions in image processing
					1. important operation in signal and image processing
					2. convolution operates on twp signals and produces a new signal
						1. 1st dignal: input
						2. 2nd signal: kernel/filter
						3. output: convolution/ convolved/ filtered signal
					3. convolutions in ConvNets
						1. include prior knowledge
						2. significantly reduce number of weights
				3. local receptive field
					1. restrict weights between input and hidden neurons
					2. each hidden neuron connects to only a small contiguous region of the input
					3. the region a hidden neuron is connected to is its local receptive field
					4. weight matrix of hidden neuron operates as convolutional kernel
					5. input activation of hidden neuron: $a=b+\sum_{r=0}^2 \sum_{s=0}^2 w_{r s} \cdot x_{i+r, j+s}$
				4. feature maps
					1. collection of hidden neurons arranged in rectangular grid
					2. neurons of a feature map share the same weights
					3. different neurons of feature map correspond to different local receptive fields
					4. weights $W$ correspond to features
						1. vertical/horizontal line
						2. diagonal line
						3. computerized non-interpretable feature
					5. neurons in feature map
						1. correspond to sub-region in input image
						2. measures intensity of feature in that region
					6. featur map
						1. shows where feature occurs in an input image
						2. input activation of all neurons = output of convolution operation with $W$ modulo bias b
				5. summerize:
					1. several feature maps each of which represents a different feature
					2. neurons of each feature map share the same weights
					3. performs convolutions in parallel
					4. computes input activation
			2. detector stage
				1. computes output activation using nonlinear activation functions
				2. detects occurrence of feature in corresponding local receptive field
				3. **Common choices of activation function**:
					1. empirical observation: deep convolutional neural networks with rectified linear activation function train several times faster than with tanh activation function.
			3. pooling stage:
				1. reduces number of neurons(down-sampling)
				2. max pooling:
					1. set filter size
					2. slide over feature map without overlap
					3. select maximum output activation of input region
				3. pooling stage has no weights to learn
				4. other techniques for pooling:
					1. average pooling: average over the input region
					2. L2-pooling: L2-norm over input region
				5. translation invariance
					1. pooled feature will be active when image undergoes small translations
					2. often desirable property in object recognition
			4. Summerize:
				1. consists of several feature maps
				2. applies pooling to each feature map seperately
				3. local receptive field of a neuron in deeper layers includes a region from each input channel. same holds for first layer if input image is has multiple channels
		3. feedforward neural network:
			1. stack regular FFNN
				1. one output layer
				2. optional: hidden layers
				3. fully connected
			2. output layer
				1. e.g. softmax for multiclass-classifier
				2. estimated conditional probability of each class