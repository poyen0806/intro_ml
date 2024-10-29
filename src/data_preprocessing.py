import pandas as pd

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def preprocess_data(data):
    # 分離 feature 和 target
    features = data.drop(columns=['gender', 'play years', 'hold racket handed', 'level'])
    targets = data[['gender', 'play years', 'hold racket handed', 'level']]
    
    