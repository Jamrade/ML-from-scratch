#import the numpy library
import numpy as np
#import the pandas library
import pandas as pd

#read in the new dataset
rawX = pd.read_csv("./modData/X.csv")
#used to drop the oldest row of data from the dataset to fix shape error between
#one hot vector and inputs. Shape error is caused by sliding window dataloss
rawX.drop([0], inplace = True)
#read in the one hot vectors used for the loss value as ytrue
rawy = pd.read_csv("./modData/Y.csv")
#convert both dataframes to numpy arrays
X = rawX.to_numpy()
y = rawy.to_numpy()

#create the layer creation class (each created and connected layer is called a dense layer)
class LayerDense:

    #initialize each layers attributes and variables
    def __init__(self, nInputs, nNeurons):
        #initialize each layer's weights and biases
        self.weights = 0.01 * np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1,nNeurons))

    #forward pass function
    def forward(self,inputs):
        #used to store the input values
        self.inputs = inputs
        #calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        #calculates the dot product of the derivative and the transposed inputs to get the weights derivative
        self.dWeights = np.dot(self.inputs.T, dvalues)
        #calculates the sum of all previous bias derivatives
        self.dBiases = np.sum(dvalues, axis=0, keepdims=True)
        #calculates the dot product of the derivative of the previous pass by the transposed weights to produce the inputs
        self.dInputs = np.dot(dvalues, self.weights.T)


#ReLU activation function
class ActivationRelu:

    #forward pass function
    def forward(self,inputs):
        #save the nputs
        self.inputs = inputs
        #Calculate output values from inputs, cutting off at zero for x<=0 and x = y anywhere else
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        #makes a copy of the original values
        self.dInputs = dvalues.copy()
        #zero gradient where input values were negative
        self.dInputs[self.inputs <=0] = 0
    
#softmax activation(this function is used on the fianl output layer to convert the data into probabilities)
class ActivationSoftMax:

    #forward pass function
    def forward(self, inputs):
        #Save the inputs
        self.inputs = inputs
        #get the unnormalized probabilities
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalize them for each sample
        probabilities = expValues / np.sum(expValues, axis=1, keepdims = True)
        #sets the output equal to the noramlized probabilities
        self.output = probabilities

    def backward(self, dvalues):
        #create an uninitialized array
        self.dinputs = np.emptyLike(dvalues)
        #Enumerate outputs and gradients
        for index, (singleOutput, singledValue) in enumerate(zip(self.output, dvalues)):
            #flatten output array
            singleOutput = singleOutput.reshape(-1,1)
            #calculate jacobian matric of output
            jacobianMatrix = np.diagflat(singleOutput) - np.dot(singleOutput, singleOutput.T)
            #calculate the sample wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobianMatrix, singledValue)

#Common Loss Class
class Loss:

    #calculates the data and regularization losses
    #given model output and ground truth values
    def calculate(self,output,y):
        
        #calculate sample losses
        sampleLosses = self.forward(output,y)

        #calculate mean loss
        dataLoss = np.mean(sampleLosses)

        #return loss
        return dataLoss

#class for the categorical cross-entropy loss function
class LossCategoricalCrossEntropy(Loss):

    #forward pass function
    def forward(self, yPred, yTrue):

        #number of samples in a batch
        samples = len(yPred)

        #clip data to prevent division by zero
        #clip both sides to not drag mean towards any value
        yPredClipped = np.clip(yPred, 1e-7, 1 - 1e-7)

        #probabilities for target values -
        #only if categorical labels
        if len(yTrue.shape) == 1:
            correctConfidences = yPredClipped[range(samples), yTrue]

        #mask values - only for one-hot encoded vectors
        elif len(yTrue.shape) == 2:
            correctConfidences = np.sum(yPredClipped*yTrue,axis=1)

        #loss calculation
        negativeLogLikelihoods = -np.log(correctConfidences)
        return negativeLogLikelihoods

    def backward(self, dvalues, yTrue):
        #number of samples
        samples = len(dvalues)
        #vchecks for the labels
        labels = len(dvalues[0])
        #converts sparse labels to one hot vectors
        if len(yTrue.shape) == 1:
            yTrue = np.eye(labels)[yTrue]
        #this is the numerator of the fraction, negate the true Y and divide by previous derivative
        self.dinputs = -yTrue / dvalues
        #take the exponentiated samples and divide by the total number of samples
        self.dinputs = self.dinputs/samples

#class that combines softmax and categorical cross entropy for a faster back step
class ActivationSoftmaxLossCategoricalCrossEntropy():

    #creates activation function and loss function objects
    def __init__(self):
        #initialize constructors of given classes and make an object to reference
        self.activation = ActivationSoftMax()
        self.loss = LossCategoricalCrossEntropy()

    def forward(self, inputs, yTrue):
        #output layer's activation function through the created object
        self.activation.forward(inputs)
        #set the output
        self.output = self.activation.output
        #calculate and return loss value
        return self.loss.calculate(self.output, yTrue)

    #backward pass combined
    def backward(self, dvalues, yTrue):
        #get the total number of samples for averaging
        samples = len(dvalues)
        #convert sparse to one hot encoded labels
        if len(yTrue.shape) == 2:
            yTrue = np.argmax(yTrue, axis = 1)
        
        #copy the inputs to safely modify
        self.dinputs = dvalues.copy()
        #calculate the gradient
        self.dinputs[range(samples), yTrue] -= 1
        #normalize the gradient
        self.dinputs = self.dinputs/samples

class AdamOptimizer:

    #set hyperparameters for decay, momentum, and learning rate
    def __init__(self, learningRate = 0.001, decay=0, epsilon=1e-7,beta1=0.9,beta2=0.999):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    #adjusts the learning rate based on decay rate and number of iterations
    def preUpdateParams(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1. / (1. + self.decay * self.iterations))

    #update the aparameters
    def updateParams(self, layer):

        #If the current layer does not have cache arrays
        #create new ones
        if not hasattr(layer, 'weightCache'):
            layer.weightMomentums = np.zeros_like(layer.weights)
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasMomentums = np.zeros_like(layer.biases)
            layer.biasCache = np.zeros_like(layer.biases)

        #update momentum with current gradients
        layer.weightMomentums = self.beta1 * layer.weightMomentums + (1 - self.beta1) * layer.dWeights
        layer.biasMomentums = self.beta1 * layer.biasMomentums + (1-self.beta1) * layer.dBiases

        #get corrected momentum
        weightMomentumsCorrected = layer.weightMomentums / (1 - self.beta1 ** (self.iterations + 1))
        biasMomentumsCorrected = layer.biasMomentums / (1-self.beta1 ** (self.iterations + 1))

        #update the cache with squared current gradients
        layer.weightCache = self.beta2 * layer.weightCache + (1 - self.beta2) * layer.dWeights**2
        layer.biasCache = self.beta2 * layer.biasCache + (1-self.beta2) * layer.dBiases**2

        #get corrected cache
        weightCacheCorrected = layer.weightCache / (1 - self.beta2 ** (self.iterations + 1))
        biasCacheCorrected = layer.biasCache / (1-self.beta2 ** (self.iterations + 1))

        #regular stochastic gradient descent upadte with normalization
        #using a square rooted cache epsilon is used to prevent division
        #by zero when updating weights and biases
        layer.weights += -self.currentLearningRate * weightMomentumsCorrected / (np.sqrt(weightCacheCorrected) + self.epsilon)
        layer.biases += -self.currentLearningRate * biasMomentumsCorrected / (np.sqrt(biasCacheCorrected) + self.epsilon)

    def postUpdateParams(self):
        self.iterations += 1
        


class Accuracy:

    #define the calculation function
    def calculate(self, input, y):
        predictions = np.argmax(input, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        return accuracy


#print every attribute of every class to fully understand the process

#dense 1 creates an array of specified shape for weights and intializes random values 0 - 1
#then the number of zerod biases for the number of neurons specified
dense1 = LayerDense(4,50)
#dense 2 and 3 create layers of matching shape to dense 1 [1] == [0]
#and use the ReLU activation function
dense2 = LayerDense(50,50)
dense3 = LayerDense(50,50)
dense4 = LayerDense(50,50)
dense5 = LayerDense(50,50)
dense6 = LayerDense(50,50)
dense7 = LayerDense(50,50)
dense8 = LayerDense(50,50)
dense9 = LayerDense(50,50)
dense10 = LayerDense(50,50)
dense11 = LayerDense(50,50)

#output dense layer uses softmax
dense12 = LayerDense(50,3)
#initializes the ReLU activation function for the created layers
activation1 = ActivationRelu()
#initialize the softmax classifier combined loss and activation
lossActivation = ActivationSoftmaxLossCategoricalCrossEntropy()
#instantiate the accuracy class
acc = Accuracy()
#instantiate and initialize the optimizer
optimizer = AdamOptimizer(learningRate = 0.005, decay = 5e-4)
for epoch in range(10001):
    #runs the network using inputs and ReLU
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation1.forward(dense2.output)
    dense3.forward(activation1.output)
    activation1.forward(dense3.output)
    dense4.forward(activation1.output)
    activation1.forward(dense4.output)
    dense5.forward(activation1.output)
    activation1.forward(dense5.output)
    dense6.forward(activation1.output)
    activation1.forward(dense6.output)
    dense7.forward(activation1.output)
    activation1.forward(dense7.output)
    dense8.forward(activation1.output)
    activation1.forward(dense8.output)
    dense9.forward(activation1.output)
    activation1.forward(dense9.output)
    dense10.forward(activation1.output)
    activation1.forward(dense10.output)
    dense11.forward(activation1.output)
    activation1.forward(dense11.output)
    dense12.forward(activation1.output)
    #computes both the softmax activation function and categorical cross entropy loss
    loss = lossActivation.forward(dense12.output, y)
    #computes the accuracy of the network
    accuracy = acc.calculate(lossActivation.output, y)
    #print all useful stats to optimizing the network
    print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' + f'lr: {optimizer.currentLearningRate}')
    #perform backward pass and calculate gradients for future optimizer use
    lossActivation.backward(lossActivation.output, y)
    
    dense12.backward(lossActivation.dinputs)
    activation1.backward(dense12.dInputs)
    dense11.backward(activation1.dInputs)
    activation1.backward(dense11.dInputs)
    dense10.backward(activation1.dInputs)
    activation1.backward(dense10.dInputs)
    dense9.backward(activation1.dInputs)
    activation1.backward(dense9.dInputs)
    dense8.backward(activation1.dInputs)
    activation1.backward(dense8.dInputs)
    dense7.backward(activation1.dInputs)
    activation1.backward(dense7.dInputs)
    dense6.backward(activation1.dInputs)
    activation1.backward(dense6.dInputs)
    dense5.backward(activation1.dInputs)
    activation1.backward(dense5.dInputs)
    dense4.backward(activation1.dInputs)
    activation1.backward(dense4.dInputs)
    dense3.backward(activation1.dInputs)
    activation1.backward(dense3.dInputs)
    dense2.backward(activation1.dInputs)
    activation1.backward(dense2.dInputs)
    dense1.backward(activation1.dInputs)
    #optimizer takes these gradients and adjusts weights and biases according to learning rate
    #this learning rate's goal is to find the global minimum and adjust weights and biases accordingly
    #it is subject to momentum and learning rate decay concepts that help avoid the model getting
    #"stuck" at a local minimum and not being representative of the true equation
    optimizer.preUpdateParams()
    optimizer.updateParams(dense1)
    optimizer.updateParams(dense2)
    optimizer.updateParams(dense3)
    optimizer.updateParams(dense4)
    optimizer.updateParams(dense5)
    optimizer.updateParams(dense6)
    optimizer.postUpdateParams()
    
