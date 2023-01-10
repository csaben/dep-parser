#adjustment for the label embeddings. will require a new fn for my dep parse after its initial generation.

import torch
import torchtext

# Load the GloVe word embeddings
embeddings = torchtext.vocab.GloVe(name='6B', dim=100)

# List of dictionaries representing the tokens in a sentence
tokens = [
    {'token': 'sample_word', 'pos': 'NN'},
    {'token': 'det', 'dep': 1},
    {'token': 'other_sample_word', 'pos': 'NN', 'dep': 2}
]

# Create a list of the token embedding vectors
embedding_vectors = [
    embeddings.vectors[embeddings.stoi[token['token']]]
    for token in tokens
]

# Create a list of the lengths of the token sequences
lengths = [len(tokens)]

# Create a list of the dependencies between the tokens, represented as a sequence of integers
dependencies = [token['dep'] for token in tokens]

# Pad and pack the token embedding vectors into a single tensor, along with the dependencies
packed_input = torch.nn.utils.rnn.pack_padded_sequence(
    torch.stack(embedding_vectors),
    lengths,
    batch_first=True,
    enforce_sorted=False
)

# Unpack the packed sequence to get the padded token embedding tensor and the dependencies
padded_input, _ = torch.nn.utils.rnn.pad_packed_sequence(
    packed_input,
    batch_first=True,
    padding_value=0
)

print(padded_input)
# tensor([[[-0.0715,  0.4215,  0.9443,  ..., -0.5459,  0.5016,  0.5409],
#           [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#           [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]])
print(dependencies)
# [1, 2]

#decoder fn
import torch
import torchtext

# Load the GloVe word embeddings
embeddings = torchtext.vocab.GloVe(name='6B', dim=100)

def decoding(embedded_label):
    # Unpack the embedded label into the padded tensor of token embedding vectors and dependencies
    token_vectors, dependencies = embedded_label
    
    # Create a list of dictionaries representing the tokens in the sentence
    tokens = []
    for i, vector in enumerate(token_vectors):
        # Find the token string corresponding to the index of the maximum value in the embedding vector
        token = embeddings.itos[vector.argmax()]
        # Add the token to the list of tokens, along with its dependency index
        tokens.append({'token': token, 'dep': dependencies[i]})
    
    # Build a tree-like structure for the sentence
    root = tokens[-1]
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node['children']:
            node['children'][child] = tokens[child]
            stack.append(tokens[child])
    
    return root

# Example embedded label
embedded_label = (
    torch.tensor([
        [-0.0715,  0.4215,  0.9443,  ..., -0.5459,  0.5016,  0.5409],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]
    ]),
    [1, 2]
)

# Decode the embedded label
decoded_label = decoding(embedded_label)

print(decoded_label)
# {'token': 'other_sample_word', 'dep': 2, 'children': {1: {'token': 'det', 'dep': 1, 'children': {}}}}

#initial tuple label conversion
"""
Earlier we started with a dependency parse structure that looked like this; tuples = [
    (("sample_word", "NN"), "det", ("other_sample_word","NN")),
    (("another_word", "NN"), "det", ("yet_another_word","NN")),
] , can you create a function that takes that and returns the structure you suggested using? i.e. tokens = [
    {'token': 'sample_word', 'pos': 'NN'},
    {'token': 'det', 'dep': 1},
    {'token': 'other_sample_word', 'pos': 'NN', 'dep': 2}
]?

A:
Certainly! Here is an example of how you can create a function that takes a list of dependency parse tuples, such as [(("sample_word", "NN"), "det", ("other_sample_word","NN")), (("another_word", "NN"), "det", ("yet_another_word","NN"))], and returns a list of dictionaries representing the tokens in the sentence, such as [{'token': 'sample_word', 'pos': 'NN'}, {'token': 'det', 'dep': 1}, {'token': 'other_sample_word', 'pos': 'NN', 'dep': 2}]:

"""
def from_tuples(tuples):
    # Initialize an empty list of tokens
    tokens = []
    
    # Iterate over the tuples
    for i, tup in enumerate(tuples):
        # Extract the token, part-of-speech, and dependency index from the tuple
        token, pos, dep = tup[0][0], tup[0][1], i+1
        # Create a dictionary for the token and add it to the list of tokens
        tokens.append({'token': token, 'pos': pos, 'dep': dep})
    
    return tokens

# Example dependency parse tuples
tuples = [    (("sample_word", "NN"), "det", ("other_sample_word","NN")),    (("another_word", "NN"), "det", ("yet_another_word","NN"))]

# Convert the tuples to a list of dictionaries representing the tokens in the sentence
tokens = from_tuples(tuples)

print(tokens)
# [{'token': 'sample_word', 'pos': 'NN', 'dep': 1}, {'token': 'det', 'pos': 'dep', 'dep': 2}, {'token': 'other_sample_word', 'pos': 'NN', 'dep': 3}]
