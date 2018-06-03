import keras
from keras import regularizers, constraints, initializers, activations
import keras.backend as K
import keras.layers as L
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf


class Seq2SeqWithBPE:

    def __init__(
        self, bpe_input, bpe_target, max_len_input, max_len_target,
        latent_dim=512, dropout=0.5, train_embeddings=True
    ):
        self.bpe_input = bpe_input
        self.bpe_target = bpe_target
        self.max_len_input = max_len_input
        self.max_len_target = max_len_target
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

    def create_batch_generator(
        self, samples_ids, input_sequences, target_sequences, batch_size
    ):
    
        def batch_generator():
            nr_batches = np.ceil(len(samples_ids) / batch_size)
            while True:
                shuffled_ids = np.random.permutation(samples_ids)
                batch_splits = np.array_split(shuffled_ids, nr_batches)
                for batch_ids in batch_splits:
                    batch_X = pad_sequences(
                        input_sequences.iloc[batch_ids],
                        padding='post',
                        maxlen=self.max_len_input
                    )
                    batch_y = pad_sequences(
                        target_sequences.iloc[batch_ids],
                        padding='post',
                        maxlen=self.max_len_target
                    )
                    batch_y_t_output = keras.utils.to_categorical(
                        batch_y[:, 1:],
                        num_classes=self.nr_target_tokens
                    )
                    batch_x_t_input = batch_y[:, :-1]
                    yield ([batch_X, batch_x_t_input], batch_y_t_output)
        
        return batch_generator()

    def decode_beam_search(self, input_seq, beam_width):
        initial_states = self.inference_encoder_model.predict(input_seq)
        
        top_candidates = [{
            'states': initial_states,
            'idx_sequence': [self.bpe_target.start_token_idx],
            'token_sequence': [self.bpe_target.start_token],
            'score': 0.0,
            'live': True
        }]
        live_k = 1
        dead_k = 0
        
        for _ in range(self.max_len_target):
            if not(live_k and dead_k < beam_width):
                break
            new_candidates = []
            for candidate in top_candidates:
                if not candidate['live']:
                    new_candidates.append(candidate)
                    continue
             
                target_seq = np.zeros((1, self.max_len_target - 1))
                target_seq[0, 0] = candidate['idx_sequence'][-1]
                output, states = self.inference_decoder_model.predict(
                    [target_seq, candidate['states']]
                )
                probs = output[0, 0, :]
            
                for idx in np.argsort(-probs)[:beam_width]:
                    new_candidates.append({
                        'states': states,
                        'idx_sequence': candidate['idx_sequence'] + [idx],
                        'token_sequence': (
                            candidate['token_sequence'] + [self.bpe_target.tokens[idx]]
                        ),
                        # sum -log(prob) numerical more stable than to multiplikate probs
                        # goal now to minimize the score
                        'score': candidate['score'] - np.log(probs[idx]),
                        'live': idx != self.bpe_target.stop_token_idx,
                    })
            
            top_candidates = sorted(
                new_candidates, key=lambda c: c['score']
            )[:beam_width]
            
            alive = np.array([c['live'] for c in top_candidates])
            live_k = sum(alive == True)
            dead_k = sum(alive == False)
            
        return self.bpe_target.sentencepiece.DecodePieces(top_candidates[0]['token_sequence'])


# copy+pasted mainly from
# https://github.com/datalogue/keras-attention/blob/master/models/custom_recurrents.py
class AttentionDecoder(L.Recurrent):

    def __init__(self, units, output_dim,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Implements an AttentionDecoder that takes in a sequence encoded by an
        encoder and outputs the decoded states
        :param units: dimension of the hidden state and the attention matrices
        :param output_dim: the number of labels in the output space
        references:
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
            "Neural machine translation by jointly learning to align and translate."
            arXiv preprint arXiv:1409.0473 (2014).
        """
        self.units = units
        self.output_dim = output_dim
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences

    def build(self, input_shape):
        """
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.
        """

        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [None, None]  # y, s

        """
            Matrices for creating the context vector
        """

        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the r (reset) gate
        """
        self.C_r = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_r = self.add_weight(shape=(self.units, self.units),
                                   name='U_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_r = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_r = self.add_weight(shape=(self.units, ),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
            Matrices for the z (update) gate
        """
        self.C_z = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_z = self.add_weight(shape=(self.units, self.units),
                                   name='U_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_z = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_z = self.add_weight(shape=(self.units, ),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the proposal
        """
        self.C_p = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_p = self.add_weight(shape=(self.units, self.units),
                                   name='U_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_p = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_p = self.add_weight(shape=(self.units, ),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for making the final prediction vector
        """
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
                                   name='C_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_o = self.add_weight(shape=(self.units, self.output_dim),
                                   name='U_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim),
                                   name='W_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_o = self.add_weight(shape=(self.output_dim, ),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_s',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)

        self.input_spec = [
            keras.engine.InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    def call(self, x):
        # store the whole sequence so we can "attend" to it at each timestep
        self.x_seq = x

        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.units)

        return super(AttentionDecoder, self).call(x)

    def get_initial_state(self, inputs):
        print('inputs shape:', inputs.get_shape())

        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def step(self, x, states):

        ytm, stm = states

        # repeat the hidden state to the length of the sequence
        _stm = K.repeat(stm, self.timesteps)

        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_a)

        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.
        et = K.dot(activations.tanh(_Wxstm + self._uxpb),
                   K.expand_dims(self.V_a))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        # ~~~> calculate new hidden state
        # first calculate the "r" gate:

        rt = activations.sigmoid(
            K.dot(ytm, self.W_r) + K.dot(stm, self.U_r) + K.dot(context, self.C_r) + self.b_r
        )

        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z) +
            K.dot(stm, self.U_z) +
            K.dot(context, self.C_z) +
            self.b_z
        )

        # calculate the proposal hidden state:
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p) +
            K.dot((rt * stm), self.U_p) +
            K.dot(context, self.C_p) +
            self.b_p
        )

        # new hidden state:
        st = (1 - zt) * stm + zt * s_tp

        yt = activations.softmax(
            K.dot(ytm, self.W_o) +
            K.dot(stm, self.U_o) +
            K.dot(context, self.C_o) +
            self.b_o
        )

        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# copy+pasted from https://github.com/datalogue/keras-attention/blob/master/models/tdd.py
def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x
