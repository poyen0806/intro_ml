import argparse
import joblib
import pandas as pd
from src.data_preprocessing import load_and_process_data, split_data
from src.model_training import train_and_evaluate_model, save_model, save_selector
from src.model_prediction import predict_target

# Argument parser for choosing between training and prediction
parser = argparse.ArgumentParser(description="Train models or make predictions.")
parser.add_argument('--mode', choices=['train', 'predict'], required=True, help="Choose whether to train the model or make predictions.")
args = parser.parse_args()

if args.mode == 'train':
    # Training phase
    print("Starting training phase...")
    
    # Train models and save them
    models = {}
    selectors = {}
    for target in ['gender', 'hold racket handed', 'play years', 'level']:
        print(f"Training model for {target}...")
        
        # Load data for the specific target
        features, targets = load_and_process_data('data/train.csv')
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = split_data(features, targets)
        
        # Train and evaluate the model
        model, selector = train_and_evaluate_model(X_train, y_train, X_test, y_test, target)
        
        # Save model and selector
        models[target] = model
        selectors[target] = selector
        save_model(model, f'models/{target}_model.pkl')
        save_selector(selector, f'selectors/{target}_selector.pkl')

    print("Training completed and models saved.")

elif args.mode == 'predict':
    # Prediction phase
    print("Starting prediction phase...")

    # Load saved models and selectors
    models = {target: joblib.load(f'models/{target}_model.pkl') for target in ['gender', 'hold racket handed', 'play years', 'level']}
    selectors = {target: joblib.load(f'selectors/{target}_selector.pkl') for target in ['gender', 'hold racket handed', 'play years', 'level']}

    # Initialize an empty DataFrame for storing all predictions
    res = pd.DataFrame({'data_ID': pd.read_csv('data/test.csv')['data_ID']})

    # Predict and adjust the result DataFrame
    res = predict_target(models, selectors, './data/test.csv', res)
    
    # Save the final result to a CSV file
    res.to_csv('data/test_predictions.csv', index=False)
    print("Prediction completed and results saved.")