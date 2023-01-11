import numpy as np

label = (([3,4,5],[4,2]),[9,8,7,6],([0,2],[8,0]))

def pad_label(label):
    """

    we can assume that the largest len in labels 

    """
    stack = [label]
    while stack:
        curr = stack.pop()
        if isinstance(curr, tuple):
            curr = tuple(self.tokenize_label(elem) for elem in curr)
        elif isinstance(curr, list):
            curr = [self.tokenize_label(elem) for elem in curr]
        elif isinstance(curr, str):
            # curr = '[CLS] '+curr+' [SEP]' #bert doesnt need label to have these tokens
            curr = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(curr))
            self.add_tokens([curr])
    return curr

import torch

# Create a batch of input tuples
batch = [
    (([3, 4, 5], [4, 2]), [9, 8, 7, 6], ([0, 2], [8, 0])),
    (([1, 2, 3], [4, 5]), [3, 2, 1, 0], ([1, 2], [3, 4])),
    (([1], [2]), [3], ([4], [5]))
]

# Determine the maximum length of the arrays in the tuple
max_lengths = [max([len(arr) for arr in tpl]) for tpl in batch]
# testing = [[len(arr) for arr in tpl] for tpl in batch]
# max_lengths = [max(max(len(sub_elem) for sub_elem in elem) if isinstance(elem, tuple) else len(elem) for elem in example) for example in labels]

#THIS ONE APPEARS TO WORK!!
testing = [max(max(len(sub_elem) for sub_elem in elem) if isinstance(elem, tuple) else len(elem)
               for elem in example) for example in batch]
print(max(testing))

print(testing)
max_length = max(max_lengths)
# print(max_length)

# Create a mask for each input tuple
masks = []
for tpl in batch:
    mask = []
    for arr in tpl:
        cur_mask = [1] * len(arr) + [0] * (max_length - len(arr))
        mask.append(cur_mask)
    masks.append(mask)

# Convert the masks to tensors
masks = torch.tensor(masks, dtype=torch.float32)
# print(masks)

# Use the masks during computation by element-wise multiplying
# the mask with the input tensor
# output = model(inputs) * masks

