import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.initializers import Constant
from transformer import create_masks, Decoder, Pointer_Generator
from creates import log
from configuration import config
from bert_model import b_model, bert_layer



# def _embedding_from_bert():
#     """
#     Extract the preratined word embeddings from a BERT model
#     Returns a numpy matrix with the embeddings
#     """
#     log.info("Extracting pretrained word embeddings weights from BERT")
    
#     bert_layer = hub.KerasLayer(BERT_MODEL_URL, trainable=False)
#     embedding_matrix = bert_layer.get_weights()[0]   
                        
#     log.info(f"Embedding matrix shape '{embedding_matrix.shape}'")
#     return embedding_matrix

class AbstractiveSummarization(tf.keras.Model):
    """
    Pretraining-Based Natural Language Generation for Text Summarization 
    https://arxiv.org/pdf/1902.09243.pdf
    """
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, input_seq_len, output_seq_len, add_stage_1, add_stage_2, rate=0.1):
        super(AbstractiveSummarization, self).__init__()
        
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.vocab_size = vocab_size
        self.bert = b_model.predict        
        embedding_matrix = bert_layer.get_weights()[0]        
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, d_model, trainable=False,
            embeddings_initializer=Constant(embedding_matrix)
        )
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, rate)
        self.d_model = d_model

        if config.copy_gen:
            self.pointer_generator   = Pointer_Generator()
                
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inp, tar, training):
        # (batch_size, seq_len) x3
        input_ids, input_mask, input_segment_ids = inp
        
        # (batch_size, seq_len + 1) x3
        target_ids, target_mask, target_segment_ids = tar
        target_ids = target_ids[:, :-1]

        # (batch_size, 1, 1, seq_len), (_), (batch_size, 1, 1, seq_len)
        _, combined_mask, dec_padding_mask = create_masks(input_ids, target_ids)

        # (batch_size, seq_len, d_bert)
        _, enc_output = self.bert([[input_ids],[input_masks],[input_segments]])

        if add_stage_1:        
            # (batch_size, seq_len, d_bert)
            embeddings = self.embedding(target_ids) 

            # (batch_size, seq_len, d_bert), (_)            
            dec_outputs, attention_dist = self.decoder(embeddings, enc_output, training, combined_mask, dec_padding_mask)

        if add_stage_2:
            N = tf.shape(enc_output)[0]
            T = self.output_seq_len
            # since we are using teacher forcing we do not need an autoregressice mechanism here
            # (batch_size x (seq_len - 1), seq_len) 
            dec_inp_ids = tile_and_mask_diagonal(target_ids, mask_with=MASK_ID)
            # (batch_size x (seq_len - 1), seq_len) 
            dec_inp_mask = tf.tile(target_mask[:, :-1], [T-1, 1])
            # (batch_size x (seq_len - 1), seq_len) 
            dec_inp_segment_ids = tf.tile(target_segment_ids[:, :-1], [T-1, 1])
            # (batch_size x (seq_len - 1), seq_len, d_bert) 
            enc_output = tf.tile(enc_output, [T-1, 1, 1])
            # (batch_size x (seq_len - 1), 1, 1, seq_len) 
            padding_mask = tf.tile(dec_padding_mask, [T-1, 1, 1, 1])
            # (batch_size x (seq_len - 1), seq_len, d_bert)
            _, context_vectors = self.bert([[dec_inp_ids],[dec_inp_mask],[dec_inp_segment_ids]])

            # (batch_size x (seq_len - 1), seq_len, d_bert), (_)
            dec_outputs, attention_dist = self.decoder(
                                                        context_vectors,
                                                        enc_output,
                                                        training,
                                                        look_ahead_mask=None,
                                                        padding_mask=padding_mask
                                                    )
            # (batch_size x (seq_len - 1), seq_len - 1, d_bert)
            dec_outputs = dec_outputs[:, 1:, :]
            # (batch_size x (seq_len - 1), (seq_len - 1))
            diag = tf.linalg.set_diag(tf.zeros([T-1, T-1]), tf.ones([T-1]))
            diag = tf.tile(diag, [N, 1])
            
            where = tf.not_equal(diag, 0)
            indices = tf.where(where)
            
            # (batch_size x (seq_len - 1), d_bert)
            dec_outputs = tf.gather_nd(dec_outputs, indices)
            
            # (batch_size, seq_len - 1, d_bert)
            dec_outputs = tf.reshape(dec_outputs, [N, T-1, -1])
            # (batch_size, seq_len, d_bert)
            dec_outputs = tf.concat(
                               [tf.tile(tf.expand_dims(tf.one_hot([CLS_ID], self.d_model), axis=0), [N, 1, 1]), dec_outputs],
                               axis=1
                               )


        # (batch_size, seq_len, vocab_len)
        logits = self.final_layer(dec_outputs)

        if config.copy_gen: 
            logits = self.pointer_generator(
                                            dec_outputs, 
                                            logits, 
                                            attention_dist, 
                                            input_ids, 
                                            tf.shape(input_ids)[1], 
                                            tf.shape(target_ids)[1], 
                                            training=training
                                            )
        return logits, attention_dist, dec_outputs