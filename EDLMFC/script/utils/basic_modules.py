from keras.layers import Dense, Conv1D, concatenate, BatchNormalization, MaxPooling1D, Bidirectional, LSTM
from keras.layers import Dropout, Flatten, Input
from keras.models import Model



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import pickle as pk


# https://keras.io/examples/nlp/text_classification_with_transformer/
#Transformer block with multihead attention layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
#         super(TransformerBlock, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        #Dropout rate
        self.rate = rate
        #Attention block
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),] )
        #Layer normalizaiton
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        super(TransformerBlock, self).__init__(**kwargs)        
    
    def get_config(self):
        config = {'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'ff_dim': self.ff_dim,
                'rate': self.rate}
        
        base_config = super(TransformerBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training):
        #Attention block
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        #Feed forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)



# https://keras.io/examples/nlp/text_classification_with_transformer/
# positional embeddings and word embeddings
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs ):
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        #word embeddings
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        #positional embeddings
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)        
    
    def get_config(self):
        config = {'maxlen': self.maxlen,
                    'vocab_size': self.vocab_size, 
                      'embed_dim': self.embed_dim}
        base_config = super(TokenAndPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        #positional embeddings
        positions = self.pos_emb(positions)
        #word embeddings
        x = self.token_emb(x)
#         Sum up the positional embeddings and word embeddings
        return x + positions



def conjoint_struct_cnn_blstm(pro_coding_length, rna_coding_length, vector_repeatition_cnn):

    if type(vector_repeatition_cnn)==int:
        vec_len_p = vector_repeatition_cnn
        vec_len_r = vector_repeatition_cnn
    else:
        vec_len_p = vector_repeatition_cnn[0]
        vec_len_r = vector_repeatition_cnn[1]


    # NN for protein feature analysis by one hot encoding
    xp_in_conjoint_struct_cnn_blstm = Input(shape=(pro_coding_length, vec_len_p))
    xp_cnn = Conv1D(filters=45, kernel_size=6, strides=1, activation='relu')(xp_in_conjoint_struct_cnn_blstm)
    xp_cnn = MaxPooling1D(pool_size=2)(xp_cnn)
    xp_cnn = BatchNormalization()(xp_cnn)
    xp_cnn = Dropout(0.2)(xp_cnn)
    xp_cnn = Conv1D(filters=64, kernel_size=6, strides=1, activation='relu')(xp_cnn)
    xp_cnn = MaxPooling1D(pool_size=2)(xp_cnn)
    xp_cnn = BatchNormalization()(xp_cnn)
    xp_cnn = Dropout(0.2)(xp_cnn)
    xp_cnn = Conv1D(filters=86, kernel_size=6, strides=1, activation='relu')(xp_cnn)
    xp_cnn = BatchNormalization()(xp_cnn)
    xp_cnn = Dropout(0.2)(xp_cnn)
    xp_cnn = layers.GlobalAveragePooling1D()(xp_cnn)
    xp_cnn = layers.Dropout(0.1)(xp_cnn)
    xp_cnn = layers.Dense(20, activation="relu")(xp_cnn)
    xp_cnn = layers.Dropout(0.1)(xp_cnn) 
    print("xp_cnn++++++++++++++++++++++++++++++++++++++++++++++",xp_cnn)
    # xp_cnn = Bidirectional(LSTM(45,return_sequences=True))(xp_cnn)
    # xp_cnn = Flatten()(xp_cnn)
    # xp_out_conjoint_cnn_blstm = Dense(64)(xp_cnn)
    # xp_out_conjoint_cnn_blstm = Dropout(0.2)(xp_out_conjoint_cnn_blstm)

    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 20


    inputs_Pr = layers.Input(shape=(xp_cnn,))
    # print("shape of input", inputs.shape)
    embedding_layer = TokenAndPositionEmbedding(xp_cnn, vocab_size, embed_dim)
    x = embedding_layer(inputs_Pr)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    
    xp_out_conjoint_cnn_blstm = Dense(64)(x)
    xp_out_conjoint_cnn_blstm = Dropout(0.2)(xp_out_conjoint_cnn_blstm)





    # NN for RNA feature analysis  by one hot encoding
    xr_in_conjoint_struct_cnn_blstm = Input(shape=(rna_coding_length, vec_len_r))
    xr_cnn = Conv1D(filters=45, kernel_size=6, strides=1, activation='relu')(xr_in_conjoint_struct_cnn_blstm)
    xr_cnn = MaxPooling1D(pool_size=2)(xr_cnn)
    xr_cnn = BatchNormalization()(xr_cnn)
    xr_cnn = Dropout(0.2)(xr_cnn)
    xr_cnn = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu')(xr_cnn)
    xr_cnn = MaxPooling1D(pool_size=2)(xr_cnn)
    xr_cnn = BatchNormalization()(xr_cnn)
    xr_cnn = Dropout(0.2)(xr_cnn)
    xr_cnn = Conv1D(filters=86, kernel_size=5, strides=1, activation='relu')(xr_cnn)
    xr_cnn = MaxPooling1D(pool_size=2)(xr_cnn)
    xr_cnn = BatchNormalization()(xr_cnn)
    xr_cnn = Dropout(0.2)(xr_cnn)
    print("xr_cnn++++++++++++++++++++++++++++++++++++++++++++++",xr_cnn)

    # xr_cnn = Bidirectional(LSTM(45,return_sequences=True))(xr_cnn)
    # xr_cnn = Flatten()(xr_cnn)
    # xr_out_conjoint_cnn_blstm = Dense(64)(xr_cnn)
    # xr_out_conjoint_cnn_blstm = Dropout(0.2)(xr_out_conjoint_cnn_blstm)
    
    
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    vocab_size = 20000  # Only consider the top 20k words
    maxlenrna = 86


    inputs_RNA = layers.Input(shape=(xr_cnn,))
    # print("shape of input", inputs.shape)
    embedding_layer = TokenAndPositionEmbedding(xr_cnn, vocab_size, embed_dim)
    x_RNA = embedding_layer(inputs_RNA)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x_RNA = transformer_block(x_RNA)
    x_RNA = layers.GlobalAveragePooling1D()(x_RNA)
    x_RNA = layers.Dropout(0.1)(x_RNA)
    xr_out_conjoint_cnn_blstm = Dense(64)(x_RNA)
    xr_out_conjoint_cnn_blstm = Dropout(0.2)(xr_out_conjoint_cnn_blstm)

    x_out_conjoint_cnn_blstm = concatenate([xp_out_conjoint_cnn_blstm, xr_out_conjoint_cnn_blstm])
    x_out_conjoint_cnn_blstm = Dense(128, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint_cnn_blstm)
    x_out_conjoint_cnn_blstm = Dropout(0.25)(x_out_conjoint_cnn_blstm)
    x_out_conjoint_cnn_blstm = BatchNormalization()(x_out_conjoint_cnn_blstm)
    x_out_conjoint_cnn_blstm = Dropout(0.3)(x_out_conjoint_cnn_blstm)
    x_out_conjoint_cnn_blstm = Dense(64, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint_cnn_blstm)
    y_conjoint_cnn_blstm = Dense(2, activation='softmax')(x_out_conjoint_cnn_blstm)

    model_conjoint_struct_cnn_blstm = Model(inputs=[xp_in_conjoint_struct_cnn_blstm, xr_in_conjoint_struct_cnn_blstm], outputs=y_conjoint_cnn_blstm)


    return model_conjoint_struct_cnn_blstm