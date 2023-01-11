class ComplexTupleLoss(nn.Module):
    def __init__(self):
        super(ComplexTupleLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    #NOTE: not quite sure the extraction is correct here
    def forward(self, output, target, mask):
        # Extract the arrays from the tuples
        output_arr1, output_arr2 = output[0]
        target_arr1, target_arr2 = target[0]
        cat_output, cat_target = output[1], target[1]
        # Apply the mask to the output and target arrays
        output_arr1 = output_arr1 * mask
        output_arr2 = output_arr2 * mask
        target_arr1 = target_arr1 * mask
        target_arr2 = target_arr2 * mask
        # Compute mean squared error loss
        mse_loss1 = self.mse(output_arr1, target_arr1)
        mse_loss2 = self.mse(output_arr2, target_arr2)
        # Compute categorical cross-entropy loss
        cat_loss = self.cross_entropy(cat_output, cat_target)
        # Sum the losses and take the mean
        total_loss = (mse_loss1 + mse_loss2 + cat_loss) / mask.sum()
        return total_loss

