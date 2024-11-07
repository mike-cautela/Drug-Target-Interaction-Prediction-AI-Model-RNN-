# train.py
from data_loader import load_data, preprocess_data
from model import create_rnn_model
import tensorflow as tf

def train_model():
    # Load and preprocess data
    df = load_data()
    X_drug_train, X_drug_test, X_target_train, X_target_test, y_train, y_test = preprocess_data(df)
    
    # Vocabulary sizes
    smiles_vocab_size = len(set("".join(df['Drug'])))
    protein_vocab_size = len(set("".join(df['Target'])))
    
    # Define model
    model = create_rnn_model(smiles_vocab_size, protein_vocab_size, 50, 200)
    
    # Train model
    model.fit(
        [X_drug_train, X_target_train], y_train,
        validation_data=([X_drug_test, X_target_test], y_test),
        epochs=10,
        batch_size=32
    )
    
  
    # Save model in Keras format instead of HDF5
    model.save("dti_rnn_model.keras")
