import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

def predict_target(models, selectors, test_data_path, res):
    """
    預測目標變數的機率，並將結果添加到 res DataFrame。
    """
    # 讀取測試資料
    test_data = pd.read_csv(test_data_path).drop(columns=['data_ID'])
    
    # 標準化
    test_data = StandardScaler().fit_transform(test_data)
    
    # 預測各目標變數的機率
    y_gender_pred = models['gender'].predict_proba(selectors['gender'].transform(test_data))
    y_hold_racket_pred = models['hold racket handed'].predict_proba(selectors['hold racket handed'].transform(test_data))
    y_play_years_pred = models['play years'].predict_proba(selectors['play years'].transform(test_data))
    y_level_pred = models['level'].predict_proba(selectors['level'].transform(test_data))
    
    # 添加二元分類目標的機率 (取第二類的機率值)
    res['gender'] = y_gender_pred[:, 1]
    res['hold racket handed'] = y_hold_racket_pred[:, 1]
    
    # 添加多類別目標的機率，展開為多個欄位
    play_years_df = pd.DataFrame(y_play_years_pred, columns=['play years_0', 'play years_1', 'play years_2'])
    level_df = pd.DataFrame(y_level_pred, columns=['level_0', 'level_1', 'level_2'])

    # 合併結果到 res DataFrame
    res = pd.concat([res, play_years_df, level_df], axis=1)
    
    return res