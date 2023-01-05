# Dependency Parsing Application


Welcome to the Dependency Parsing Application! This project provides a command-line interface for parsing the dependencies in a given sentence.

## Setup (tentative; final will take documents and output parsed trees and a key to their context in a separate file)

To use this project, you will need to have the following requirements installed on your system:

- Docker

To install the required dependencies, follow these steps:

1. Clone this repository onto your local machine:

```
git clone https://github.com/<your-username>/dependency-parsing-app.git
```

2. Navigate to the root directory of the repository:

```
cd dependency-parsing-app
```

3. Build the Docker container:

```
docker build -t dependency-parsing-app .
```

## Usage

To parse the dependencies in a given sentence, run the following command:

```
docker run dependency-parsing-app -s "Your sentence here"
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


