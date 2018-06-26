# Rosetta

I found it hard to reproduce state-of-the-art seq2seq models.
On the one hand, a lot of papers are without code.
And code examples from the web are often outdated (using older versions of tensorflow, keras, python, ...) or incomplete/not-standalong-runnable (missing important details) or full-bloated (with lots of additional stuff) or they only concentrate on one technique but not combining them with others (like attention model but without beam search).
This project is my attempt to learn and apply the state-of-the-art techniques step by step.

## Roadmap

* for toy problems, machine translation, summaries and chat botting
* in Keras first, then Tensorflow, maybe PyTorch / Tensorflow Hub / tf.keras
* from ground up simple model adding more higher level approaches (bytepairencodings, beam search, attentions, ...) 

I'm not explaining a lot, I concentrate on implementation details here. There a lot of better tutorials outside to understand seq2seq models and their terminology.

## Models step for step:

1. [Simple Model for adding and subtracting numbers end-to-end on chars](SimpleModelForAddingAndSubstraction.ipynb)
2. [Simple Model char-level end-to-end for Machine Translation](SimpleModelForMachineTranslation.ipynb)
3. [Bytepairencoding embeddings instead for Machine Translation](BytepairencodingForMachineTranslation.ipynb)
4. [Implementing BeamSearch model](BeamSearchForMachineTranslation.ipynb)
5. [BeamSearch model trained on a larger dataset](BeamSearchOnLargeDataset.ipynb)
6. [Attention model with Tensorflow trained on a larger dataset](AttentionModelForMachineTranslationWithTensorflow.ipynb)
7. [Attention model trained on full en-de europarliament dataset](AttentionModelOnFullDataset.ipynb)

## Usage / Installation

I'm using Python 3.6 with tensorflow 1.8.0 and keras 2.2.0. For details look into the Pipfiles.

I use [pipenv](https://github.com/pypa/pipenv) to track all dependencies and create a virtualenv.
Follow the instruction to install pipenv and then

    git clone git@github.com:hanfried/rosetta.git
    cd rosetta

    pipenv install
    pipenv run jupyter notebook

to start a jupyter notebook environment with all required modules installed and running in a virtualenv.

## See also

* For a complete overview of the state of the art,
look to the [NLP-Progress](https://github.com/sebastianruder/NLP-progress) project of [Sebastian Ruder](http://ruder.io/) (and subscribe to his newsletter).
* [Combining Recent Advances in Neural Machine Translation](https://arxiv.org/pdf/1804.09849.pdf) is a recent paper with a good comprehension which techniques are how effective (if used together).
* Xueyou Luo has a similiar project like mine on, look to his [my\_seq2seq project](https://github.com/xueyouluo/my_seq2seq)
