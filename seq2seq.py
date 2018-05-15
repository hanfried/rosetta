import keras.layers as L
from keras.models import Model
import tensorflow as tf


class Seq2SeqWithBPE:

    def __init__(
        self, bpe_input, bpe_target, max_len_input, max_len_target,
        latent_dim=512, dropout=0.5, train_embeddings=True
    ):
        self.bpe_input = bpe_input
        self.bpe_target = bpe_input
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.nr_input_tokens = len(bpe_input.wordvec_index)
        self.nr_target_tokens = len(bpe_target.wordvec_index)

        self.encoder_gru = L.Bidirectional(
            L.GRU(
                latent_dim // 2,
                dropout=dropout,
                return_state=True,
                name='encoder_gru',
                dtype=tf.float32
            ),
            name='encoder_bidirectional'
        )
        self.decoder_gru = L.GRU(
            latent_dim,
            dropout=dropout,
            return_sequences=True,
            return_state=True,
            name='decoder_gru',
            dtype=tf.float32
        )
        self.decoder_dense = L.Dense(
            self.nr_target_tokens,
            activation='softmax',
            name='decoder_outputs',
            dtype=tf.float32
        )

        self.input_embedding = L.Embedding(
            self.nr_input_tokens,
            bpe_input.embedding_dim,
            mask_zero=True,
            weights=[bpe_input.embedding_matrix],
            trainable=train_embeddings,
            name='input_embedding',
            dtype=tf.float32,
        )
        self.target_embedding = L.Embedding(
            self.nr_target_tokens,
            bpe_target.embedding_dim,
            mask_zero=True,
            weights=[bpe_target.embedding_matrix],
            trainable=train_embeddings,
            name='target_embedding',
            dtype=tf.float32,
        )

        self.encoder_inputs = L.Input((max_len_input, ), dtype='int32', name='encoder_inputs')
        self.encoder_embeddings = self.input_embedding(self.encoder_inputs)
        _, self.encoder_state_1, self.encoder_state_2 = self.encoder_gru(
            self.encoder_embeddings
        )
        self.encoder_states = L.concatenate([self.encoder_state_1, self.encoder_state_2])

        self.decoder_inputs = L.Input(
            shape=(max_len_target - 1, ),
            dtype='int32',
            name='decoder_inputs'
        )
        self.decoder_mask = L.Masking(mask_value=0)(self.decoder_inputs)
        self.decoder_embeddings_inputs = self.target_embedding(self.decoder_mask)
        self.decoder_embeddings_outputs, _ = self.decoder_gru(
            self.decoder_embeddings_inputs,
            initial_state=self.encoder_states
        )
        self.decoder_outputs = self.decoder_dense(self.decoder_embeddings_outputs)

        self.model = Model(
            inputs=[self.encoder_inputs, self.decoder_inputs],
            outputs=self.decoder_outputs
        )

        self.inference_encoder_model = Model(
            inputs=self.encoder_inputs,
            outputs=self.encoder_states
        )
            
        self.inference_decoder_state_inputs = L.Input(
            shape=(latent_dim, ),
            dtype='float32',
            name='inference_decoder_state_inputs'
        )
        self.inference_decoder_embeddings_outputs, \
            self.inference_decoder_states = self.decoder_gru(
                self.decoder_embeddings_inputs,
                initial_state=self.inference_decoder_state_inputs
            )
        self.inference_decoder_outputs = self.decoder_dense(
            self.inference_decoder_embeddings_outputs
        )

        self.inference_decoder_model = Model(
            inputs=[self.decoder_inputs, self.inference_decoder_state_inputs],
            outputs=[self.inference_decoder_outputs, self.inference_decoder_states]
        )
