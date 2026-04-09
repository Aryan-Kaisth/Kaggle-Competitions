# src/models.py
import numpy as np
import lightgbm as lgb
import xgboost as xgb
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
    CatBoost_TD_Classifier
) 

def _sample_weights(y_train, y_valid):
    """Balanced sample weights for train and validation sets."""
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_map = dict(zip(classes, weights))
    train_w = np.array([weight_map.get(y, 1.0) for y in y_train])
    valid_w = np.array([weight_map.get(y, 1.0) for y in y_valid])
    return train_w, valid_w

def _get_cat_cols(X):
    """Return categorical column names."""
    return X.select_dtypes(include=["object", "category"]).columns.tolist()


def _cast_cats(X, cat_cols):
    """Cast known categorical columns to category dtype."""
    if cat_cols:
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
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds  = model.predict(X_valid, num_iteration=model.best_iteration)
    test_preds = model.predict(X_test,  num_iteration=model.best_iteration)
    return oof_preds, test_preds, model

def train_xgb(X_train, X_valid, X_test, y_train, y_valid, params):
    params = params.copy()
    num_boost_round    = params.pop("num_boost_round", 5000)
    early_stopping     = params.pop("early_stopping_rounds", 100)

    cat_cols = _get_cat_cols(X_train)
    _cast_cats(X_train, cat_cols)
    _cast_cats(X_valid, cat_cols)
    _cast_cats(X_test,  cat_cols)

    train_w, valid_w = _sample_weights(y_train, y_valid)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_w, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, weight=valid_w, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test,                                 enable_categorical=True)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )

    oof_preds = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
    test_preds = model.predict(dtest,  iteration_range=(0, model.best_iteration + 1))
    return oof_preds, test_preds, model

def train_catboost(X_train, X_valid, X_test, y_train, y_valid, params):
    params = params.copy()
    params.setdefault("auto_class_weights", "Balanced")
    verbose = params.pop("verbose", False)

    cat_cols = _get_cat_cols(X_train)

    dtrain = cb.Pool(X_train, label=y_train, cat_features=cat_cols)
    dvalid = cb.Pool(X_valid, label=y_valid, cat_features=cat_cols)
    dtest  = cb.Pool(X_test, cat_features=cat_cols)

    model = cb.train(
        params=params,
        pool=dtrain,
        eval_set=dvalid,
        verbose=verbose,
    )

    oof_preds = model.predict(dvalid, prediction_type="Probability")
    test_preds = model.predict(dtest,  prediction_type="Probability")
    return oof_preds, test_preds, model

def train_histgbm(X_train, X_valid, X_test, y_train, y_valid, params):
    params = params.copy()

    cat_cols = _get_cat_cols(X_train)
    _cast_cats(X_train, cat_cols)
    _cast_cats(X_valid, cat_cols)
    _cast_cats(X_test,  cat_cols)

    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    oof_preds  = model.predict_proba(X_valid)
    test_preds = model.predict_proba(X_test)
    return oof_preds, test_preds, model

def train_extratrees(X_train, X_valid, X_test, y_train, y_valid, params):
    X_train, X_valid, X_test = X_train.copy(), X_valid.copy(), X_test.copy()
    
    cat_cols = X_train.select_dtypes(include=['category', 'object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int), cat_cols)
        ],
        remainder='passthrough' 
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ExtraTreesClassifier(**params))
    ])

    model.fit(X_train, y_train)
    
    oof_preds  = model.predict_proba(X_valid)
    test_preds = model.predict_proba(X_test)
    
    return oof_preds, test_preds, model

# Pytabkit Models
def train_LGBM_TD(X_train, X_valid, X_test, y_train, y_valid, params):
    params = params.copy()

    cat_cols = _get_cat_cols(X_train)
    _cast_cats(X_train, cat_cols)
    _cast_cats(X_valid, cat_cols)
    _cast_cats(X_test,  cat_cols)

    model = LGBM_TD_Classifier(**params)
    model.fit(X_train, y_train)

    oof_preds  = model.predict_proba(X_valid)
    test_preds = model.predict_proba(X_test)
    return oof_preds, test_preds, model

def train_XGB_TD(X_train, X_valid, X_test, y_train, y_valid, params):
    params = params.copy()

    cat_cols = _get_cat_cols(X_train)
    _cast_cats(X_train, cat_cols)
    _cast_cats(X_valid, cat_cols)
    _cast_cats(X_test,  cat_cols)

    model = XGB_TD_Classifier(**params)
    model.fit(X_train, y_train)

    oof_preds  = model.predict_proba(X_valid)
    test_preds = model.predict_proba(X_test)
    return oof_preds, test_preds, model

def train_CatBoost_TD(X_train, X_valid, X_test, y_train, y_valid, params):
    params = params.copy()

    cat_cols = _get_cat_cols(X_train)

    model = CatBoost_TD_Classifier(**params)
    model.fit(X_train, y_train, cat_col_names=cat_cols)

    oof_preds  = model.predict_proba(X_valid)
    test_preds = model.predict_proba(X_test)
    return oof_preds, test_preds, model


# Linear Models 
def train_logistic(X_train, X_valid, X_test, y_train, y_valid, params):
    X_train, X_valid, X_test = X_train.copy(), X_valid.copy(), X_test.copy()
    
    cat_cols = X_train.select_dtypes(include=['category', 'object']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['number']).columns.tolist()

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

# Registry
MODELS = {
    "lgbm": train_lgbm,
    "xgb": train_xgb,
    "catboost": train_catboost,
    "histgbm": train_histgbm,
    "extratrees": train_extratrees,
    "logistic": train_logistic,
    "CatBoost_TD": train_CatBoost_TD,
    "LGBM_TD": train_LGBM_TD,
    "XGB_TD": train_XGB_TD
}