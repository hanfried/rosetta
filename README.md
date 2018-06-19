## State of the art overview for seq2seq models

* for toy problems, machine translation, summaries and chat botting
* in Keras first, then Tensorflow, maybe PyTorch / Tensorflow Hub
* from ground up simple model adding more higher level approaches (bytepairencodings, beam search, attentions, ...) 

I'm not explaining a lot, I concentrate on implementation details here. There a lot of better tutorials outside to understand seq2seq models and their terminology.

## Models step for step:

1. [Simple Model for adding and subtracting numbers end-to-end on chars](SimpleModelForAddingAndSubstraction.ipynb)
2. [Simple Model char-level end-to-end for Machine Translation](SimpleModelForMachineTranslation.ipynb)
3. [Bytepairencoding embeddings instead for Machine Translation](BytepairencodingForMachineTranslation.ipynb)
4. [Implementing BeamSearch model](BeamSearchForMachineTranslation.ipynb)
5. [BeamSearch model trained on a larger dataset](BeamSearchOnLargeDataset.ipynb)
6. [Attention model with Tensorflow trained on a larger dataset](AttentionModelForMachineTranslationWithTensorflow.ipynb)

## Usage / Installation

I use [pipenv](https://github.com/pypa/pipenv) to track all dependencies and create a virtualenv.
Follow the instruction to install pipenv and then

    pipenv install
    pipenv run jupyter notebook

to start a jupyter notebook environment with all required modules installed and running in a virtualenv.
