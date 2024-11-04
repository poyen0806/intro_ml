import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import TEST_SIZE, RANDOM_STATE

def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    features = data.drop(columns=['player_ID', 'gender', 'play years', 'hold racket handed', 'level'], errors='ignore')
    targets = data[['gender', 'play years', 'hold racket handed', 'level']]
    return features, targets

def scale_data(features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return X_scaled, scaler

def split_data(X, y):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
