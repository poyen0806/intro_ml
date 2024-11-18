import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE

def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    
    features = data.drop(columns=['data_ID', 'player_ID', 'gender', 'play years', 'hold racket handed', 'level'])
    targets = data[['gender', 'play years', 'hold racket handed', 'level']]
    
    return features, targets

def split_data(X, y):
    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # 對訓練集進行 fit 和 transform
    X_test = scaler.transform(X_test)       # 對測試集僅進行 transform

    return X_train, X_test, y_train, y_test
