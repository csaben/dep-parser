# Dependency Parser 

## Usage

To parse the dependencies in a given sentence (using stanford parser) , run the following command:

```
python src/playback.py custom "Your sentence here"
```

To parse the dependencies in a given sentence using my parser (TBD)

```
python src/inference.py custom "Your sentence here"
```

This will output the parsed dependencies in a tree structure.

## Developers

### Generating the Dataset

To generate the dataset, you can use the NLTK Stanford NLP pipeline. Here's how you can set it up:

1. Download the Stanford CoreNLP server from [here](https://stanfordnlp.github.io/CoreNLP/download.html).

2. Extract the downloaded file and navigate to the root directory:
```
cd stanford-corenlp-full-2018-02-27
```

3. Run the server:

```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload t
```

You can find more information about using the Stanford CoreNLP server in NLTK [here](https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK).


## Literature

1. [stanford ch14 dependency parsing](https://web.stanford.edu/~jurafsky/slp3/14.pdf)


