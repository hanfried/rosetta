import gc
import os

import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import pandas as pd

import bytepairencoding as bpe
from utils.linguistic import read_europarl


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.set_random_seed(RANDOM_STATE)
print(f"Fixed random seed to {RANDOM_STATE}")


def check_gpu_working():
    print("Availabe devices:", device_lib.list_local_devices())

    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

    with tf.Session() as sess:
        assert np.all(sess.run(c) == np.array([[22, 28], [49, 64]]))
        print("Cuda/Cudnn/GPU works as intended")


class Europarl:

    def __init__(
        self,
        path='data',
        input_lang='en',
        target_lang='de',
        merge_operations=5000,
        embedding_dim=300,
    ):
        self.path = path
        self.input_lang = input_lang
        self.target_lang = target_lang
        self.merge_operations = merge_operations
        self.embedding_dim = embedding_dim

        corpus_fname = f'{target_lang}-{input_lang}.tgz'
        self.external_resources = {
            corpus_fname: f'http://statmt.org/europarl/v7/{corpus_fname}'
        }
        for lang in [input_lang, target_lang]:
            self.external_resources.update({
                name: self.bpe_url(lang) + name
                for name
                in [self.bpe_model_name(lang), self.bpe_word2vec_name(lang) + '.tar.gz']
            })
            
    def bpe_url(self, lang):
        return f'http://cosyne.h-its.org/bpemb/data/{lang}/'

    def bpe_model_name(self, lang):
        return f'{lang}.wiki.bpe.op{self.merge_operations}.model'

    def bpe_word2vec_name(self, lang):
        return f'{lang}.wiki.bpe.op{self.merge_operations}.d{self.embedding_dim}.w2v.bin'

    def load_and_preprocess(self, max_input_length=None, max_target_length=None):
        df = pd.DataFrame(data={
            'input_texts': read_europarl(self.input_lang),
            'target_texts': read_europarl(self.target_lang)
        })
        print("Total number of unfiltered translations", len(df))
        
        df['input_length'] = df.input_texts.apply(len)
        df['target_length'] = df.target_texts.apply(len)

        # there are empty phrases like '\n' --> 'Frau Präsidentin\n'
        non_empty = (df.input_length > 1) & (df.target_length > 1)
        short_inputs = (
            (df.input_length < max_input_length) & (df.target_length < max_target_length)
        )
        print(
            f'Filtered translations with length between',
            f'(1, input={max_input_length}/target={max_target_length}) characters:',
            sum(non_empty & short_inputs)
        )
        df = df[non_empty & short_inputs]
        # df with filtered sentences is significant smaller, so time to garbage collect
        gc.collect()

        bpe_input, bpe_target = [bpe.Bytepairencoding(
            word2vec_fname=os.path.join(self.path, self.bpe_word2vec_name(lang)),
            sentencepiece_fname=os.path.join(self.path, self.bpe_model_name(lang)),
        ) for lang in [self.input_lang, self.target_lang]]

        df['input_sequences'] = df.input_texts.apply(bpe_input.subword_indices)
        df['target_sequences'] = df.target_texts.apply(bpe_target.subword_indices)

        self.df = df
        self.bpe_input = bpe_input
        self.bpe_target = bpe_target
