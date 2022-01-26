from keras.layers import Dense, Conv1D, concatenate, BatchNormalization, MaxPooling1D, Bidirectional, LSTM
from keras.layers import Dropout, Flatten, Input
from keras.models import Model



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import pickle as pk


# def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
#     # Normalization and Attention
#     x = layers.LayerNormalization(epsilon=1e-6)(inputs)
#     x = layers.MultiHeadAttention(
#         key_dim=head_size, num_heads=num_heads, dropout=dropout
#     )(x, x)
#     x = layers.Dropout(dropout)(x)
#     res = x + inputs

#     # Feed Forward Part
#     x = layers.LayerNormalization(epsilon=1e-6)(res)
#     x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#     return x + res

# def build_model(
#     input_shape,
#     head_size,
#     num_heads,
#     ff_dim,
#     num_transformer_blocks,
#     mlp_units,
#     dropout=0,
#     mlp_dropout=0,
# ):
#     inputs = keras.Input(shape=input_shape)
#     x = inputs
#     for _ in range(num_transformer_blocks):
#         x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

#     x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
#     for dim in mlp_units:
#         x = layers.Dense(dim, activation="relu")(x)
#         x = layers.Dropout(mlp_dropout)(x)
#     outputs = layers.Dense(n_classes, activation="softmax")(x)
#     return keras.Model(inputs, outputs)










class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)



class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

embed_dim = 1  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
vocab_size = 20000  # Only consider the top 20k words
maxlen = 102 

def conjoint_struct_cnn_blstm(pro_coding_length, rna_coding_length, vector_repeatition_cnn):

    if type(vector_repeatition_cnn)==int:
        vec_len_p = vector_repeatition_cnn
        vec_len_r = vector_repeatition_cnn
    else:
        vec_len_p = vector_repeatition_cnn[0]
        vec_len_r = vector_repeatition_cnn[1]

    # NN for protein feature analysis by one hot encoding
    xp_in_conjoint_struct_cnn_blstm = Input(shape=(pro_coding_length, vec_len_p))
    
    print ("pro_coding_length, pro_coding_length")
    print ("vec_len_p, vec_len_p")
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
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",xp_cnn)
    # embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    # xp_cnn = embedding_layer(xp_cnn)
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",xp_cnn)

    # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # xp_cnn = transformer_block(xp_cnn)
    # print("////////////////////////////////////////////////////////////",xp_cnn)

    # xp_cnn = layers.GlobalAveragePooling1D()(xp_cnn)
    # xp_cnn = layers.Dropout(0.1)(xp_cnn)
    # xp_cnn = layers.Dense(20, activation="relu")(xp_cnn)
    # xp_cnn = layers.Dropout(0.1)(xp_cnn)
    
    
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",xp_cnn)

    xp_cnn = Bidirectional(LSTM(45,return_sequences=True))(xp_cnn)
    xp_cnn = Flatten()(xp_cnn)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",xp_cnn)
   
    xp_out_conjoint_cnn_blstm = Dense(64)(xp_cnn)
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
    
    
    xr_cnn = Bidirectional(LSTM(45,return_sequences=True))(xr_cnn)
    xr_cnn = Flatten()(xr_cnn)
    
    print("xr_cnn ",xr_cnn)
    # embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    # xr_cnn = embedding_layer(xr_cnn)
    # print("@@@@@@@@@@@@@@@",xr_cnn)

    # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # xr_cnn = transformer_block(xr_cnn)
    # print("////////////////////////",xr_cnn)

    # xr_cnn = layers.GlobalAveragePooling1D()(xr_cnn)
    # xr_cnn = layers.Dropout(0.1)(xr_cnn)
    # xr_cnn = layers.Dense(20, activation="relu")(xr_cnn)
    # xr_cnn = layers.Dropout(0.1)(xr_cnn)
    
    # print("@@@@@@@@",xr_cnn)
    
    xr_out_conjoint_cnn_blstm = Dense(64)(xr_cnn)
    xr_out_conjoint_cnn_blstm = Dropout(0.2)(xr_out_conjoint_cnn_blstm)

# 
    x_out_conjoint_cnn_blstm = concatenate([xp_out_conjoint_cnn_blstm, xr_out_conjoint_cnn_blstm])
    x_out_conjoint_cnn_blstm = Dense(128, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint_cnn_blstm)
    x_out_conjoint_cnn_blstm = Dropout(0.25)(x_out_conjoint_cnn_blstm)
    x_out_conjoint_cnn_blstm = BatchNormalization()(x_out_conjoint_cnn_blstm)
    x_out_conjoint_cnn_blstm = Dropout(0.3)(x_out_conjoint_cnn_blstm)
    x_out_conjoint_cnn_blstm = Dense(64, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint_cnn_blstm)
    y_conjoint_cnn_blstm = Dense(2, activation='softmax')(x_out_conjoint_cnn_blstm)

    model_conjoint_struct_cnn_blstm = Model(inputs=[xp_in_conjoint_struct_cnn_blstm, xr_in_conjoint_struct_cnn_blstm], outputs=y_conjoint_cnn_blstm)


    return model_conjoint_struct_cnn_blstm
