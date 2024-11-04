import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from config import RANDOM_STATE, FEATURE_SELECTION_K

def train_and_evaluate_model(X_train, y_train, X_test, y_test, target_name):
    selector = SelectKBest(f_classif, k=FEATURE_SELECTION_K)
    X_train_selected = selector.fit_transform(X_train, y_train[target_name])
    X_test_selected = selector.transform(X_test)

    if target_name == 'gender':
        model = KNeighborsClassifier(n_neighbors=10, weights='distance')
    elif target_name == 'hold racket handed':
        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', max_depth=10)
    elif target_name == 'play years':
        model = SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced', kernel='rbf', C=1, gamma='scale')
    elif target_name == 'level':
        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', max_depth=10)
    
    model.fit(X_train_selected, y_train[target_name])

    predictions_proba = model.predict_proba(X_test_selected)

    if target_name in ['play years', 'level']:
        roc_auc = roc_auc_score(y_test[target_name], predictions_proba, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_test[target_name], predictions_proba[:, 1])

    report = classification_report(y_test[target_name], model.predict(X_test_selected))

    print(f"{target_name} Prediction ROC AUC Score:")
    print(f'ROC AUC: {roc_auc}')
    print(f'Report:\n{report}')
    
    return model, selector

def save_model(model, file_path):
    joblib.dump(model, file_path)

def save_selector(selector, file_path):
    joblib.dump(selector, file_path)
