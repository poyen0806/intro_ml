import pandas as pd
import joblib

def load_model(file_path):
    return joblib.load(file_path)

def predict_and_export(file_path, scaler, selectors, models, output_path):
    test_data = pd.read_csv(file_path)
    features = test_data.drop(columns=['player_ID', 'gender', 'play years', 'hold racket handed', 'level'], errors='ignore')
    X_scaled = scaler.transform(features)

    predictions = {}
    for target in ['gender', 'hold racket handed']:
        selector = selectors[target]
        X_selected = selector.transform(X_scaled)
        predictions[target] = models[target].predict(X_selected)

    play_years_test_selected = selectors['play years'].transform(X_scaled)
    level_test_selected = selectors['level'].transform(X_scaled)

    play_years_pred = models['play years'].predict(play_years_test_selected)
    level_pred = models['level'].predict(level_test_selected)

    play_years_df = pd.get_dummies(play_years_pred, prefix='play years', drop_first=False).astype(int)
    level_df = pd.get_dummies(level_pred, prefix='level', drop_first=False).astype(int)

    required_play_years_columns = ['play years_0', 'play years_1', 'play years_2']
    required_level_columns = ['level_0', 'level_1', 'level_2']
    
    for col in required_play_years_columns:
        if col not in play_years_df.columns:
            play_years_df[col] = 0
    for col in required_level_columns:
        if col not in level_df.columns:
            level_df[col] = 0

    play_years_df = play_years_df[required_play_years_columns]
    level_df = level_df[required_level_columns]

    output_df = pd.DataFrame()
    output_df['data_ID'] = test_data['data_ID']
    output_df['gender'] = predictions['gender'].astype(int)
    output_df['hold racket handed'] = predictions['hold racket handed'].astype(int)

    output_df = pd.concat([output_df, play_years_df, level_df], axis=1)

    output_df = output_df[['data_ID', 'gender', 'hold racket handed', 
                           'play years_0', 'play years_1', 'play years_2', 
                           'level_0', 'level_1', 'level_2']]
    
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
