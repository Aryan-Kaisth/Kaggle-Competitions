# src/models.py
import os
import gc
import numpy as np
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pytabkit import (
    XGB_TD_Classifier, 
    LGBM_TD_Classifier, 
    CatBoost_TD_Classifier,
    Resnet_RTDL_D_Classifier,
    RealMLP_TD_Classifier,
    TabM_D_Classifier,
)

def _sample_weights(y_train, y_valid):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_map = dict(zip(classes, weights))
    train_w = np.array([weight_map.get(y, 1.0) for y in y_train])
    valid_w = np.array([weight_map.get(y, 1.0) for y in y_valid])
    return train_w, valid_w

def _get_cat_cols(X):
    return X.select_dtypes(include=["object", "category"]).columns.tolist()

def _get_num_cols(X):
    return X.select_dtypes(include=["number"]).columns.tolist()

def _cast_cats(X, cat_cols):
    X[cat_cols] = X[cat_cols].astype("category")

# GBDT Models
def train_lgbm(X_train, X_valid, X_test, y_train, y_valid, params):
    params = params.copy()
    num_iterations = params.pop("num_iterations", 5000)
    early_stopping = params.pop("early_stopping_rounds", 100)

    cat_cols = _get_cat_cols(X_train)
    _cast_cats(X_train, cat_cols)
    _cast_cats(X_valid, cat_cols)
    _cast_cats(X_test,  cat_cols)

    train_w, valid_w = _sample_weights(y_train, y_valid)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=train_w)
    dvalid = lgb.Dataset(X_valid, label=y_valid, weight=valid_w, reference=dtrain)

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_iterations,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=False)
        ],
    )

    oof_preds  = model.predict(X_valid, num_iteration=model.best_iteration)
    test_preds = model.predict(X_test,  num_iteration=model.best_iteration)
    
    del dtrain, dvalid
    gc.collect()
    
    return oof_preds, test_preds, model

def train_catboost(X_train, X_valid, X_test, y_train, y_valid, params):

    cat_cols = _get_cat_cols(X_train)

    dtrain = cb.Pool(X_train, label=y_train, cat_features=cat_cols)
    dvalid = cb.Pool(X_valid, label=y_valid, cat_features=cat_cols)
    dtest  = cb.Pool(X_test, cat_features=cat_cols)

    model = cb.train(
        params=params,
        pool=dtrain,
        eval_set=dvalid
    )

    oof_preds = model.predict(dvalid, prediction_type="Probability")
    test_preds = model.predict(dtest,  prediction_type="Probability")

    del dtrain, dvalid, dtest
    gc.collect()

    return oof_preds, test_preds, model

def train_histgbm(X_train, X_valid, X_test, y_train, y_valid, params):
    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    oof_preds  = model.predict_proba(X_valid)
    test_preds = model.predict_proba(X_test)
    return oof_preds, test_preds, model

# Pytabkit Models
def train_LGBM_TD(X_train, X_valid, X_test, y_train, y_valid, params):
    cat_cols = _get_cat_cols(X_train)

    model = LGBM_TD_Classifier(**params)
    model.fit(X_train, y_train, X_valid, y_valid, cat_col_names=cat_cols)

    oof_preds  = model.predict_proba(X_valid)
    test_preds = model.predict_proba(X_test)
    return oof_preds, test_preds, model

def train_XGB_TD(X_train, X_valid, X_test, y_train, y_valid, params):
    cat_cols = _get_cat_cols(X_train)

    model = XGB_TD_Classifier(**params)
    model.fit(X_train, y_train, X_valid, y_valid, cat_col_names=cat_cols)

    oof_preds  = model.predict_proba(X_valid)
    test_preds = model.predict_proba(X_test)
    return oof_preds, test_preds, model

def train_CatBoost_TD(X_train, X_valid, X_test, y_train, y_valid, params):
    cat_cols = _get_cat_cols(X_train)

    model = CatBoost_TD_Classifier(**params)
    model.fit(X_train, y_train, X_valid, y_valid, cat_col_names=cat_cols)

    oof_preds  = model.predict_proba(X_valid)
    test_preds = model.predict_proba(X_test)
    return oof_preds, test_preds, model

def train_Resnet_RTDL_D(X_train, X_valid, X_test, y_train, y_valid, params):
    cat_cols = _get_cat_cols(X_train)

    model = Resnet_RTDL_D_Classifier(**params)
    model.fit(X_train, y_train, X_valid, y_valid, cat_col_names=cat_cols)

    oof_preds  = model.predict_proba(X_valid)
    test_preds = model.predict_proba(X_test)
    return oof_preds, test_preds, model

def train_RealMLP_TD(X_train, X_valid, X_test, y_train, y_valid, params):
    cat_cols = _get_cat_cols(X_train)

    model = RealMLP_TD_Classifier(**params)
    model.fit(X_train, y_train, X_valid, y_valid, cat_col_names=cat_cols)

    oof_preds  = model.predict_proba(X_valid)
    test_preds = model.predict_proba(X_test)
    return oof_preds, test_preds, model

def train_TabM_D(X_train, X_valid, X_test, y_train, y_valid, params):
    cat_cols = _get_cat_cols(X_train)

    model = TabM_D_Classifier(**params)
    model.fit(X_train, y_train, X_valid, y_valid, cat_col_names=cat_cols)

    oof_preds  = model.predict_proba(X_valid)
    test_preds = model.predict_proba(X_test)
    return oof_preds, test_preds, model

# Logistic Regression 
def train_logistic(X_train, X_valid, X_test, y_train, y_valid, params):
    X_train, X_valid, X_test = X_train.copy(), X_valid.copy(), X_test.copy()
    
    cat_cols = _get_cat_cols(X_train)
    num_cols = _get_num_cols(X_train)

    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('scaler', StandardScaler(), num_cols)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(**params))
    ])

    model.fit(X_train, y_train)
    
    oof_preds  = model.predict_proba(X_valid)
    test_preds = model.predict_proba(X_test)
    
    return oof_preds, test_preds, model