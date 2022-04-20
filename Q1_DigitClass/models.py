### code base: ai.berkeley.edu

import nn


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        "Our learning Rate"
        self.lr = .1

        self.w1 = nn.Parameter(784, 256)
        self.w2 = nn.Parameter(256, 128)
        self.w3 = nn.Parameter(128, 64)
        self.w4 = nn.Parameter(64, 10)

        self.b1 = nn.Parameter(1, 256)
        self.b2 = nn.Parameter(1, 128)
        self.b3 = nn.Parameter(1, 64)
        self.b4 = nn.Parameter(1, 10)

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        f1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        f2 = nn.ReLU(nn.AddBias(nn.Linear(f1, self.w2), self.b2))
        f3 = nn.ReLU(nn.AddBias(nn.Linear(f2, self.w3),self.b3))

        finalOutput = nn.AddBias(nn.Linear(f3, self.w4), self.b4)
        "Returns a batch size *1 node that represents the predicted scores"
        return finalOutput

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        "Uses the previous method to run the input provided"
        nodeShape = self.run(x)
        "Next we find the soft max loss between the second input and the model run of the first input"
        softMax = nn.SoftmaxLoss(nodeShape, y)
        "Returns the loss for the given inputs and outputs"
        return softMax

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        "Stores the accuracy of our modeling"
        accuracyPercentage = 0
        "Will store the total loss "
        totalLoss = float('inf')
        rangeOfLength = range(len(self.params))
        totalFill = 100

        "While the accuracy is less than 98%, we will continue to try and improve our data to learn to adapt to the set"
        "Therefore 'training' it to understand the picture"
        while .98 > accuracyPercentage:
            "For each data piece in the set"
            for position1, position2 in dataset.iterate_once(totalFill):
                "Get the amount that does not get understood and put it as a scalar"
                totalLoss = self.get_loss(position1, position2)
                "Get the change in the gradient compared to the params"
                totalChange = nn.gradients(totalLoss, self.params)

                """For each position in the length of the parameters, we will update the total change in the gradient to"
                see our correct accuracy percentage"""
                for pos in rangeOfLength:
                    self.params[pos].update(totalChange[pos], -self.lr)
            "Will get the percentage of our accuracy that has been validated"
            correctPercentage = dataset.get_validation_accuracy()


