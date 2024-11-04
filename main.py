import argparse
import joblib
from src.data_preprocessing import load_and_process_data, scale_data, split_data
from src.model_training import train_and_evaluate_model, save_model, save_selector
from src.model_prediction import predict_and_export

# Argument parser for choosing between training and prediction
parser = argparse.ArgumentParser(description="Train models or make predictions.")
parser.add_argument('--mode', choices=['train', 'predict'], required=True, help="Choose whether to train the model or make predictions.")
args = parser.parse_args()

if args.mode == 'train':
    # Training phase
    print("Starting training phase...")
    
    # Load and preprocess data
    features, targets = load_and_process_data('data/train.csv')
    X, scaler = scale_data(features)
    X_train, X_test, y_train, y_test = split_data(X, targets)

    # Train models and save them
    models = {}
    selectors = {}
    for target in ['gender', 'hold racket handed', 'play years', 'level']:
        model, selector = train_and_evaluate_model(X_train, y_train, X_test, y_test, target)
        models[target] = model
        selectors[target] = selector
        save_model(model, f'models/{target}_model.pkl')
        save_selector(selector, f'selectors/{target}_selector.pkl')

    # Save scaler for later use in prediction
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Training completed and models saved.")

elif args.mode == 'predict':
    # Prediction phase
    print("Starting prediction phase...")
    
    # Load saved models, scaler, and selectors
    scaler = joblib.load('models/scaler.pkl')
    models = {target: joblib.load(f'models/{target}_model.pkl') for target in ['gender', 'hold racket handed', 'play years', 'level']}
    selectors = {target: joblib.load(f'selectors/{target}_selector.pkl') for target in ['gender', 'hold racket handed', 'play years', 'level']}

    # Perform prediction and export results
    predict_and_export('data/test.csv', scaler, selectors, models, 'data/test_predictions.csv')
    print("Prediction completed and results saved.")
