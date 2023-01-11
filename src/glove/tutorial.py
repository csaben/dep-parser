import torch
import torchtext
import spacy

def main():
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    glove = torchtext.vocab.GloVe(name='6B', dim=100)
    train = ['I prefer the flight to Denver.']
    TEXT.build_vocab(train, vectors="glove.6B.100d")


#Initialize Unknown Words Randomly

def init_emb(vocab, init="randn", num_special_toks=2):
    emb_vectors = vocab.vectors
    sweep_range = len(vocab)
    running_norm = 0.
    num_non_zero = 0
    total_words = 0
    for i in range(num_special_toks, sweep_range):
        if len(emb_vectors[i, :].nonzero()) == 0:
            # std = 0.05 is based on the norm of average GloVE 100-dim word vectors
            if init == "randn":
                torch.nn.init.normal(emb_vectors[i], mean=0, std=0.05)
        else:
            num_non_zero += 1
            running_norm += torch.norm(emb_vectors[i])
        total_words += 1
    logger.info("average GloVE norm is {}, number of known words are {}, total number of words are {}".format(
        running_norm / num_non_zero, num_non_zero, total_words))

def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


if __name__ == '__main__':
    main()
