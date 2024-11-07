# model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate

def create_rnn_model(smiles_vocab_size, protein_vocab_size, max_smiles_len, max_protein_len):
    # Drug (SMILES) Input
    drug_input = Input(shape=(max_smiles_len,), name="Drug_Input")
    drug_embedding = Embedding(input_dim=smiles_vocab_size + 1, output_dim=128)(drug_input)
    drug_rnn = LSTM(64)(drug_embedding)
    
    # Protein (Amino Acid Sequence) Input
    target_input = Input(shape=(max_protein_len,), name="Target_Input")
    target_embedding = Embedding(input_dim=protein_vocab_size + 1, output_dim=128)(target_input)
    target_rnn = LSTM(64)(target_embedding)
    
    # Concatenate and output layer
    combined = Concatenate()([drug_rnn, target_rnn])
    output = Dense(1, activation="linear")(combined)
    
    model = Model(inputs=[drug_input, target_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    return model
