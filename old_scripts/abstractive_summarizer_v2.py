import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.initializers import Constant
from transformer import create_masks, Decoder, Pointer_Generator
from creates import log
from configuration import config
from bert_model import BertLayer

# Special Tokens
UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103

def tile_and_mask_diagonal(x, mask_with):
    """    
    Masks each word in the summary draft one by one with the [MASK] token
    At t-th time step the t-th word of input summary is
    masked, and the decoder predicts the refined word given other
    words of the summary.
    
    x :: (N, T)
    returrn :: (N, T-1, T)
    
    We do not mask the first and last postition (corresponding to [CLS]
    """

    N, T = tf.shape(x)[0], tf.shape(x)[1]

    first = tf.reshape(tf.tile(x[:, 0], [T-1]), [N, T-1, 1])
    
    x = x[:, 1:]
    T = T - 1
    
    masked = tf.reshape(tf.tile(x, [1, T]), [N, T, T])
    
    diag = tf.ones([N, T], dtype=masked.dtype) * mask_with
    masked = tf.linalg.set_diag(masked, diag)
    
    masked = tf.concat([first, masked], axis=2)
    
    masked = tf.reshape(masked, [N*T, T+1])
    
    return masked

def _embedding_from_bert():
  log.info("Extracting pretrained word embeddings weights from BERT")
  BERT_MODEL_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
  vocab_of_BERT = hub.KerasLayer(BERT_MODEL_URL, trainable=False)
  embedding_matrix = vocab_of_BERT.get_weights()[0]   
  log.info(f"Embedding matrix shape '{embedding_matrix.shape}'")
  return (embedding_matrix, vocab_of_BERT)

class draft_summary(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, rate=0.1):
        super(draft_summary, self).__init__()
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, rate)        
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self,
              embeddings,
              enc_output,
              look_ahead_mask,
              padding_mask,
              target_ids,
              training):
        
        # (batch_size, seq_len, d_bert), (_)            
        draft_dec_outputs, draft_attention_dist = self.decoder(
                                                                embeddings, 
                                                                enc_output, 
                                                                training, 
                                                                look_ahead_mask, 
                                                                padding_mask
                                                                )
        # (batch_size, seq_len, vocab_len)
        draft_logits = self.final_layer(draft_dec_outputs)
        return draft_logits, draft_attention_dist, draft_dec_outputs

class refine_summary(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, output_seq_len, rate=0.1):
        super(refine_summary, self).__init__()
        self.bert = BertLayer(d_embedding=d_model, trainable=False)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, rate)
        self.final_layer = tf.keras.layers.Dense(vocab_size)
        self.output_seq_len = output_seq_len
        self.d_model = d_model

    def call(self, 
             enc_output, 
             target, 
             padding_mask, 
             training):
        N = tf.shape(enc_output)[0]
        T = self.output_seq_len
        # (batch_size, seq_len) x3
        dec_inp_ids, dec_inp_mask, dec_inp_segment_ids = target
        # since we are using teacher forcing we do not need an autoregressice mechanism here
        # (batch_size x (seq_len - 1), seq_len) 
        dec_inp_ids = tile_and_mask_diagonal(dec_inp_ids, mask_with=MASK_ID)
        # (batch_size x (seq_len - 1), seq_len) 
        dec_inp_mask = tf.tile(dec_inp_mask, [T-1, 1])
        # (batch_size x (seq_len - 1), seq_len) 
        dec_inp_segment_ids = tf.tile(dec_inp_segment_ids, [T-1, 1])
        # (batch_size x (seq_len - 1), seq_len, d_bert) 
        enc_output = tf.tile(enc_output, [T-1, 1, 1])
        # (batch_size x (seq_len - 1), 1, 1, seq_len) 
        padding_mask = tf.tile(padding_mask, [T-1, 1, 1, 1])
        # (batch_size x (seq_len - 1), seq_len, d_bert)
        context_vectors = self.bert((dec_inp_ids, dec_inp_mask, dec_inp_segment_ids))

        # (batch_size x (seq_len - 1), seq_len, d_bert), (_)
        dec_outputs, refine_attention_dist = self.decoder(
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
        refine_dec_outputs = tf.concat(
                           [tf.tile(
                                    tf.expand_dims(
                                                   tf.one_hot(
                                                              [CLS_ID], 
                                                              self.d_model
                                                              ),
                                                   axis=0
                                                  ), 
                                    [N, 1, 1]
                                    ), 
                            dec_outputs
                           ],
                           axis=1
                           )


        # (batch_size, seq_len, vocab_len)
        refine_logits = self.final_layer(refine_dec_outputs)
        return refine_logits, refine_attention_dist, refine_dec_outputs

class AbstractiveSummarization(tf.keras.Model):
    """
    Pretraining-Based Natural Language Generation for Text Summarization 
    https://arxiv.org/pdf/1902.09243.pdf
    """
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, output_seq_len, rate=0.1):
        super(AbstractiveSummarization, self).__init__()
    
        self.output_seq_len = output_seq_len
        self.vocab_size = vocab_size
        self.bert = BertLayer(d_embedding=d_model, trainable=False)       
        embedding_matrix, self.vocab_of_BERT = _embedding_from_bert()
        self.embedding = tf.keras.layers.Embedding(
                                                    vocab_size, d_model, trainable=False,
                                                    embeddings_initializer=Constant(embedding_matrix)
                                                   )
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, rate)
        self.draft_summary = draft_summary(num_layers, d_model, num_heads, dff, vocab_size, rate=0.1)
        self.refine_summary = refine_summary(num_layers, d_model, num_heads, dff, vocab_size, output_seq_len, rate=0.1)
        self.d_model = d_model
        if config.copy_gen:
            self.pointer_generator   = Pointer_Generator()
                
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inp, tar, training):
        # (batch_size, seq_len) x3
        input_ids, input_mask, input_segment_ids = inp
        
        # (batch_size, seq_len + 1) x3
        target_ids, target_mask, target_segment_ids = tar

        # (batch_size, 1, 1, seq_len), (_), (batch_size, 1, 1, seq_len)
        _, combined_mask, dec_padding_mask = create_masks(input_ids, target_ids[:, :-1])

        # (batch_size, seq_len, d_bert)
        enc_output = self.bert((input_ids, input_mask, input_segment_ids))

        # (batch_size, seq_len, d_bert)
        embeddings = self.embedding(target_ids[:, :-1]) 

        draft_logits,\
        draft_attention_dist,\
        draft_dec_outputs = self.draft_summary(
                                                embeddings,
                                                enc_output,
                                                combined_mask,
                                                dec_padding_mask,
                                                target_ids[:, :-1],
                                                training
                                              )

        if config.copy_gen: 
            draft_logits = self.pointer_generator(
                                                draft_dec_outputs, 
                                                draft_logits, 
                                                draft_attention_dist, 
                                                input_ids, 
                                                tf.shape(input_ids)[1], 
                                                tf.shape(target_ids[:, :-1])[1], 
                                                training
                                                )


        refine_logits,\
        refine_attention_dist,\
        refine_dec_outputs = self.refine_summary(
                                                  enc_output,
                                                  (target_ids[:, :-1], target_mask[:, :-1], 
                                                   target_segment_ids[:, :-1]),            
                                                  dec_padding_mask,
                                                  training
                                                 )
                                                
              
        return (draft_logits, draft_attention_dist, draft_dec_outputs, 
                refine_logits, refine_attention_dist, refine_dec_outputs)
