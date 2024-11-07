# evaluate.py
from tensorflow.keras.models import load_model
from data_loader import load_data, preprocess_data
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model():
    # Load data and model
    df = load_data()
    X_drug_train, X_drug_test, X_target_train, X_target_test, y_train, y_test = preprocess_data(df)
    
    # Load the model in Keras format
    model = load_model("dti_rnn_model.keras")

    
    # Make predictions
    y_pred = model.predict([X_drug_test, X_target_test])
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
