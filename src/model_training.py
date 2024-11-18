import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from config import RANDOM_STATE, FEATURE_SELECTION_K

def train_and_evaluate_model(X_train, y_train, X_test, y_test, target_name):
    # Feature selection
    selector = SelectKBest(f_classif, k=FEATURE_SELECTION_K)
    X_train_selected = selector.fit_transform(X_train, y_train[target_name])
    X_test_selected = selector.transform(X_test)

    # Dynamic model selection and hyperparameter tuning
    if target_name in ['gender', 'hold racket handed']:
        xgb_model = XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.005,
        )
        rf_model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
        )
        knn_model = KNeighborsClassifier(
            n_neighbors=7, 
            weights='distance',
        )
        model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('knn', knn_model),
            ],
            voting='soft',
            weights=[0.4, 0.4, 0.2],
        )
    elif target_name in ['play years', 'level']:
        rf_model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=500,
            max_depth=25,
            min_samples_split=5,
            class_weight='balanced',
        )
        lgbm_model = LGBMClassifier(
            random_state=RANDOM_STATE,
            n_estimators=800,
            learning_rate=0.05,
            max_depth=20,
            force_col_wise=True,
        )
        model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('lgbm', lgbm_model),
            ],
            voting='soft',
            weights=[0.4, 0.6],
        )
    else:
        raise ValueError(f"Unsupported target_name: {target_name}")

    # Training
    model.fit(X_train_selected, y_train[target_name])

    # Prediction and evaluation
    predictions_proba = model.predict_proba(X_test_selected)
    
    if target_name in ['play years', 'level']:
        roc_auc = roc_auc_score(y_test[target_name], predictions_proba, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_test[target_name], predictions_proba[:, 1])

    report = classification_report(y_test[target_name], model.predict(X_test_selected))
    
    # Output results
    print(f"{target_name} Prediction ROC AUC Score:")
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'Classification Report:\n{report}')
    
    return model, selector

def save_model(model, file_path):
    joblib.dump(model, file_path)

def save_selector(selector, file_path):
    joblib.dump(selector, file_path)
