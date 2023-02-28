from time import sleep
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import mode
import time

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping

import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers



def training_lgb_by_cross_validation(config, df_train, df_test, features, params, callbacks):
    # Create a numpy array to store test predictions
    test_predictions = np.zeros((len(df_test), config.n_folds))

    # Create a numpy array to store out of folds predictions
    oof_predictions = np.zeros(len(df_train))

    feature_importance_df = pd.DataFrame(index=features)
    y_valids, val_preds =[],[]

    df_score = pd.DataFrame(np.zeros((1, config.n_folds)), columns=[f'fold-{i+1}' for i in range(config.n_folds)])

    # K-Fold
    # kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    # for fold, (train_idx, valid_idx) in enumerate(kfold.split(df_train, df_train[config.target])):

    # Stratified-K-Fold
    kfold = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(df_train, df_train[config.target])):

        print(' ')
        print('-'*50)
        print(f'Training fold {fold+1} with {len(features)} features...')

        X_train, X_val = df_train[features].iloc[train_idx], df_train[features].iloc[valid_idx]
        y_train, y_val = df_train[config.target].iloc[train_idx], df_train[config.target].iloc[valid_idx]

        # Over Sampling
        # sm = SMOTE(random_state=config.random_seed)
        # X_train, y_train = sm.fit_resample(X_train, y_train)

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_val, y_val)

        model = lgb.train(params=params, train_set=lgb_train, valid_sets=[lgb_train, lgb_valid], valid_names=['train', 'valid'], callbacks=callbacks)
        print(f'================================== training {fold+1} fin. ==================================')

        # Predict validation data
        print(f'================================== validation-data predicting ... ==================================')
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        #val_pred = np.argmax(val_pred, axis=1)
        oof_predictions[valid_idx] = val_pred

        # Predict test data
        print(f'================================== test-data predicting ... ==================================')
        test_pred = model.predict(df_test[features], num_iteration=model.best_iteration)
        test_predictions[:, fold] += test_pred

        # save results
        y_valids.append(y_val)
        val_preds.append(val_pred)
        feature_importance_df["Importance_Fold"+str(fold+1)]=model.feature_importance(importance_type='gain')

        # Compute fold metric
        val_pred = pd.DataFrame(data={'prediction': val_pred})
        y_val = pd.DataFrame(data={'target': y_val.reset_index(drop=True)})

        score = roc_auc_score(y_val, val_pred)
        df_score.iloc[0, fold] = score

        print(f'Fold {fold+1} CV result')
        print(f'metric : {score}')

        del X_train, X_val, y_train, y_val, lgb_train, lgb_valid

        # 表示が流れないように待機
        sleep(3)

    # Compute out of folds metric
    oof_predictions = pd.DataFrame(data={'prediction': oof_predictions})
    #y_true = pd.DataFrame(data=config.target: df_train[config.target]})

    # Create a dataframe to store out of folds predictions
    oof_df = pd.DataFrame({config.row_id: df_train[config.row_id], config.target: df_train[config.target], 'prediction': oof_predictions['prediction']})

    # Create a dataframe to store test prediction
    test_predictions = test_predictions.mean(axis=1)
    test_df = pd.DataFrame({config.row_id: df_test[config.row_id], config.target: test_predictions})

    return feature_importance_df, oof_df, test_df


def root_mean_squared_error(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))


def setup_model():
    activation = 'relu'
    kernel_initializer = 'he_normal'

    model = Sequential()

    model.add(Dense(64, kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    # model.add(Dropout(0.25))

    model.add(Dense(48, kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    # model.add(Dropout(0.5))

    model.add(Dense(32, kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    # model.add(Dropout(0.25))

    model.add(Dense(24, kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    # model.add(Dropout(0.25))

    model.add(Dense(16, kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.25))

    model.add(Dense(11, activation='softmax'))

    optimizer = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=True)
    # optimizer = optimizers.SGD(learning_rate=0.001)

    # binary classification
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
    # customize
    # model.compile(optimizer=optimizer, loss=root_mean_squared_error, metrics=[root_mean_squared_error])

    return model


def setup_callbacks():
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=5, verbose=1)
    callbacks = [es, lr]

    return callbacks


mlp_param = {
    'epochs': 300,
    'batch_size': 256,
    'verbose': 1,
}


def training_MLP_by_seed_validation(config, X_train, y_train, X_val, y_val, df_val, df_test, features, model, callbacks, mlp_param):

    # Create a numpy array to store out of folds predictions
    oof_predictions = np.zeros((len(X_val), config.n_folds))
    # oof_predictions = np.zeros(len(df_train))

    # Create a numpy array to store test predictions
    test_predictions = np.zeros((len(df_test), config.n_folds))

    feature_importance_df = pd.DataFrame(index=features)
    y_valids, val_preds =[],[]

    df_score = pd.DataFrame(np.zeros((1, config.n_folds)), columns=[f'fold-{i+1}' for i in range(config.n_folds)])

    # 学習履歴の記録
    training_history = {}

    # Stratified-K-Fold
    # kfold = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    # for fold, (train_idx, valid_idx) in enumerate(kfold.split(df_train, df_train[config.target])):
    for fold in range(config.n_folds):
        np.random.seed(config.random_seed + fold)
        tf.random.set_seed(config.random_seed + fold)

        print(' ')
        print('-'*50)
        print(f'Training fold {fold+1} with {len(features)} features...')

        #X_train, X_val = df_train[features].iloc[train_idx], df_train[features].iloc[valid_idx]
        #y_train, y_val = df_train[config.target].iloc[train_idx], df_train[config.target].iloc[valid_idx]

        # model = setup_model()
        # callbacks = setup_callbacks()
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=mlp_param['epochs'], batch_size=mlp_param['batch_size'], callbacks=callbacks, verbose=mlp_param['verbose'])


        print(f'================================== training {fold+1} fin. ==================================')
        training_history[f'fold-{fold}'] = history

        # Predict validation data
        print(f'================================== validation-data predicting ... ==================================')
        val_pred = model.predict(X_val)
        #oof_predictions[valid_idx] = val_pred.reshape(-1)
        oof_predictions[:, fold] += val_pred.reshape(-1)

        # Predict test data
        print(f'================================== test-data predicting ... ==================================')
        test_pred = model.predict(df_test[features])
        test_predictions[:, fold] += test_pred.reshape(-1)

        # save results
        y_valids.append(y_val)
        val_preds.append(val_pred)

        score = roc_auc_score(y_val, val_pred)
        df_score.iloc[0, fold] = score

        print(f'Fold {fold+1} CV result')
        print(f'metric : {score}')

    # Create a dataframe to store test prediction
    oof_predictions = oof_predictions.mean(axis=1)
    oof_df = pd.DataFrame({config.row_id: df_val[config.row_id], config.target: df_val[config.target], 'prediction': oof_predictions})
    #oof_df = pd.DataFrame({config.row_id: df_train[config.row_id], config.target: df_train[config.target], 'prediction': oof_predictions})

    # Create a dataframe to store test prediction
    test_predictions = test_predictions.mean(axis=1)
    test_df = pd.DataFrame({config.row_id: df_test[config.row_id], config.target: test_predictions})

    return training_history, oof_predictions, oof_df, test_df, df_score