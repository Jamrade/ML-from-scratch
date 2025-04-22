# Project Overview:
This Neural Network categorizes and predicts price trends for the following day using the previous weeks data. These predictions are for the price of Stocks that use “High” and “Low” columns in their datasets. The dataset tracks prices daily and includes the following points: open, close, high, low. The goal is to determine the direction of future price shifts based strictly on the previous week’s averages

**note: the vast majority of the model portion of the code is directly from the neural networks from scratch textbook, there are only so many ways you can write the chainrule, ReLU, layer initialization etc however creating, optimizing, and testing the model as well as the data module is solely mine**

**note: all datasets are from kaggle**

## Project Decision:
I have been working with neural networks and financial modeling separately
for a while now, beginning in 2022 with the Nvidia workshop, I figured I would like to understand
the logic behind the magic and, hopefully, use this increased knowledge to further optimize and
perfect my own private networks.

## How to Run:
1. Clone the repository
2. Upload your original ticker data into the "originaldata" folder
3. Adjust the datamodifer's window for your timeframe of choice (1M, 15M, 30M, 1H, 4H, 1D, 1W, 1M, 1Y)
4. Run the datamodifier to generate your One-Hot Vector
5. Run "NeuralNetwork.py" in the "convolutionalmodel" folder
6. Adjust your hyperparameters between runs (found in the ADAM Optimizer class Alpha1: Learning rate, Beta1: beta1, Beta2: beta2, decay: starting decay rate for momentum)
7. Repeat until high accuracy is achieved, then save the weights and biases using pickling.

## Instructions on use:
grab historical pricing for a stock ensuring the high and low columns are present
run the DataSlidingRule.py file
put the name of your csv in the parameters to the read_csv function (ex. dataset.read_csv(“{your dataset name}.csv”))
in the DataSlidingRule.py file call the average() function with name of desired column to have its’ average added to the new dataset
when all desired columns are added run the direction() function to generated the ytrue values in one hot vector format
then run the normalize() function to normalize your data points
then run the output() function to save your ytrue and normalized datasets as y.csv and X.csv respectively
go to neural network.py and create a new layer dense instance name “dense1” of the same shape as your dataset’s second dimension (ex. inputs = [1840, 4], layerDense(4, 20))
make as many hidden layers as you would like matching the shape of the second dimension with the first of the subsequent layer (ex. dense1 = [4, 20] dense2 = [20, 69])
instantiate the first activation fucntion, the loss function, the accuracy class, and the optimizer
Note: the following should all be done within a for loop that has the number of desired epochs as its range
run the forward pass of each layer of the network, layer one uses the normalized dataset as its input every subsequent layer uses the previous layer’s activation function output (ex. dense1.forward(X), activation1.forward(dense1.output), dense2.forward(activation1.output)
for the final dense layer run the forward pass then run the loss function (ex. dense2.forward(activation1.output), lossActivationFunction.forward(dense2.output))
run the backward passes in the reverse order of the forward, starting with the loss and moving back to the backward pass of the first dense layer
run the optimizer preParamUpdate() function
run the optimizer paramUpdate() function
run the optimizer postParamUpdate() function

