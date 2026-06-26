import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Concatenate, Lambda, Dense, Dropout

def build_siamese_model(vocab_size, embedding_dim=100, max_len=100):
    # Inputs
    input_a = Input(shape=(max_len,), dtype='int32', name='input_a')
    input_b = Input(shape=(max_len,), dtype='int32', name='input_b')
    
    # Shared layers
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='shared_embedding')
    bilstm_layer = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.0), name='shared_bilstm')
    
    # Twin features
    feat_a = bilstm_layer(embedding_layer(input_a))
    feat_b = bilstm_layer(embedding_layer(input_b))
    
    # Calculate absolute difference and product
    abs_diff = Lambda(lambda x: __import__('tensorflow').abs(x[0] - x[1]), output_shape=lambda x: x[0], name='abs_difference')([feat_a, feat_b])
    prod = Lambda(lambda x: x[0] * x[1], output_shape=lambda x: x[0], name='elementwise_product')([feat_a, feat_b])
    
    # Concatenate representations
    merged = Concatenate(name='merge_features')([feat_a, feat_b, abs_diff, prod])
    
    # Classification head
    x = Dense(64, activation='relu', name='dense_1')(merged)
    x = Dropout(0.3, name='dropout_1')(x)
    output = Dense(1, activation='sigmoid', name='similarity_output')(x)
    
    # Model
    model = Model(inputs=[input_a, input_b], outputs=output, name='Siamese_BiLSTM')
    return model

if __name__ == '__main__':
    model = build_siamese_model(vocab_size=5000, embedding_dim=100, max_len=100)
    model.summary()
