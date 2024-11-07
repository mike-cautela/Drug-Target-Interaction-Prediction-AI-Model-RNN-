# data_loader.py
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data():
    from tdc.multi_pred import DTI
    data = DTI(name="BindingDB_Kd")
    df = data.get_data()
    return df

def preprocess_data(df, max_smiles_len=50, max_protein_len=200):

    # Initialize scalers for Y
    scaler = MinMaxScaler()
    df['Y'] = scaler.fit_transform(df[['Y']])


    # Tokenizers for SMILES and Protein sequences
    smiles_tokenizer = Tokenizer(char_level=True)
    protein_tokenizer = Tokenizer(char_level=True)
    
    smiles_tokenizer.fit_on_texts(df['Drug'])
    protein_tokenizer.fit_on_texts(df['Target'])
    
    # Convert to integer sequences
    drug_sequences = smiles_tokenizer.texts_to_sequences(df['Drug'])
    target_sequences = protein_tokenizer.texts_to_sequences(df['Target'])
    
    # Pad sequences
    drug_padded = pad_sequences(drug_sequences, maxlen=max_smiles_len, padding='post')
    target_padded = pad_sequences(target_sequences, maxlen=max_protein_len, padding='post')
    
    # Split the data
    X_drug_train, X_drug_test, X_target_train, X_target_test, y_train, y_test = train_test_split(
        drug_padded, target_padded, df['Y'].values, test_size=0.2, random_state=42
    )
    
    return X_drug_train, X_drug_test, X_target_train, X_target_test, y_train, y_test
