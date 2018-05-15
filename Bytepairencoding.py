from gensim.models import KeyedVectors
import numpy as np
import sentencepiece as spm


class Bytepairencoding:

    def __init__(self, word2vec_fname, sentencepiece_fname):
        self.wordvec = KeyedVectors.load_word2vec_format(word2vec_fname, binary=True)
        self.sentencepiece = spm.SentencePieceProcessor()
        self.sentencepiece.Load(sentencepiece_fname)

        self.wordvec_index = {
            word: index
            for index, word
            in enumerate(['<pad>', '<s>', '</s>'] + self.wordvec.index2word)
        }  # haven't found start/stop tokens, so add them manually
        self.unk_index = self.wordvec_index['<unk>']

        self.embedding_dim = self.wordvec.vector_size + 2  # incl. start/stop
        self.embedding_shape = ((len(self.wordvec_index), self.embedding_dim))
        self.embedding_matrix = np.zeros(self.embedding_shape, dtype=np.float32)
        # pad symbol as close to zero
        self.embedding_matrix[0, :] = 1e-6 * np.random.standard_normal(self.embedding_dim)
        self.embedding_matrix[1, -1] = 1  # one hot encode start symbol
        self.embedding_matrix[2, -2] = 1  # one hot encode stop symbol
        self.embedding_matrix[3:, :-2] = self.wordvec.vectors

    def subword_indices(self, text):
        pieces = self.sentencepiece.EncodeAsPieces(text)
        return [
            self.wordvec_index.get(subword, self.unk_index)
            for subword in ['<s>'] + pieces + ['</s>']  # automatic add start/stop index
        ]
