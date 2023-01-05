import torchtext


# define the fields for the dataset
fields = [("text", torchtext.legacy.data.Field(sequential=True, use_vocab=True)),
          ("label", torchtext.legacy.data.Field(sequential=False, use_vocab=False))]

# create the dataset
dataset = torchtext.legacy.datasets.text_classification_datasets.TextClassificationDataset("./ptb.train.txt", fields)

# split the dataset into train and test sets
train_data, test_data = dataset.split(split_ratio=0.8, random_state=torch.random.manual_seed(123))
# from torchtext.legacy import data
# from torchtext.legacy import datasets
# import torchtext

# # define the fields for the dataset
# fields = [("text", torchtext.legacy.data.Field(sequential=True, use_vocab=True)),
#           ("label", torchtext.legacy.data.Field(sequential=False, use_vocab=False))]

# # load the dataset
# dataset = torchtext.legacy.datasets.PennTreebank.TabularDataset(path="../input/ptb.train.txt", format="tsv", fields=fields)

# # split the dataset into train and test sets
# train_data, test_data = dataset.split(split_ratio=0.8, random_state=torch.random.manual_seed(123))

# # Download the PTB dataset
# # ptb_train, ptb_valid, ptb_test =datasets.PennTreebank.splits#.splits(torchtext.datasets.Treebank)

# # # Print the number of examples in each split
# # print(f"Number of training examples: {len(ptb_train)}")
# # print(f"Number of validation examples: {len(ptb_valid)}")
# # print(f"Number of test examples: {len(ptb_test)}")

# # import torchtext

# # # Specify the path to the dataset file
# # path = '../input/ptb.train.txt'

# # # Define the fields for the dataset
# # fields = [('text', data.Field(sequential=True))]

# # # Load the dataset
# # dataset = torchtext.datasets.TabularDataset(path=path, format='tsv', fields=fields)

# # # Split the dataset into train, validation, and test sets
# # train_dataset, valid_dataset, test_dataset = dataset.random_split([0.7, 0.15, 0.15])
