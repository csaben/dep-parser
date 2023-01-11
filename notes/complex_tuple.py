"""
I need to make a custom loss to properly generate an average loss to compare my complex tuple labels

I also need a cutom padding function for the arrays within the complex tuple

below is more info on both things.


tuple loss

It depends on the specific structure of your tuple, but in general, you can define a loss function that takes in the output of your model (the complex tuple) and the ground truth label (the complex tuple) and compares the values within the tuples in some meaningful way.

For example, you can use the mean squared error loss for comparing the arrays within the tuples and cross-entropy loss for comparing the categorical variables(if any).
In order to calculate the loss for the entire tuple, you need to compute the individual losses for each component and aggregate them. One way to achieve this is to use a custom loss function that performs the following steps:

Compute the mean squared error between the arrays within the output tuple and the ground truth tuple.
Compute the cross-entropy loss between the categorical variables.
Sum the two losses above to get the total loss.
Here's an example of a custom loss function that handles the complex tuple structure:
"""



import torch
import torch.nn as nn

class ComplexTupleLoss(nn.Module):
    def __init__(self):
        super(ComplexTupleLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        # Extract the arrays from the tuples
        output_arr1, output_arr2 = output[0]
        target_arr1, target_arr2 = target[0]
        cat_output, cat_target = output[1], target[1]
        # Compute mean squared error loss
        mse_loss1 = self.mse(output_arr1, target_arr1)
        mse_loss2 = self.mse(output_arr2, target_arr2)
        # Compute categorical cross-entropy loss
        cat_loss = self.cross_entropy(cat_output, cat_target)
        # Sum the losses
        total_loss = mse_loss1 + mse_loss2 + cat_loss
        return total_loss

      
"""
The custom padding (you could probably use the rnn sequence one if you unpack the complex tuple first but  this way
arguably is easier to understand and less prone to ruining the fidelity of the original label structurin

When you pad your custom label, you will want to make sure that each element of the tuple has the same number of items, so that they can be processed in a batch together. This will typically involve adding padding to the shorter elements, so that they are the same length as the longest element.

One way to pad the label in your example, would be to find the maximum length of the sublists and use this value to pad all the other sublists with zeros. Here is an example of how you might implement this:
"""
import numpy as np

label = (([3,4,5],[4,2]),[9,8,7,6],([0,2],[8,0]))

# Find the maximum length of sublists
max_len = max([len(sublist) for sublist in label])

# Pad the sublists with zeros
padded_label = [[sublist + [0] * (max_len - len(sublist)) for sublist in element] for element in label]

#Convert to numpy array
padded_label = np.array(padded_label)

