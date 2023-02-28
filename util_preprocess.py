
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import make_pipeline


def clip_outlier(df, col):
    '''clip outlier
    '''

    Q1 = np.percentile(df[col], 25)
    Q3 = np.percentile(df[col], 75)

    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR

    df.loc[:, col] = np.maximum(df[col], Q1 - outlier_step)
    df.loc[:, col] = np.minimum(df[col], Q3 + outlier_step)

    return df


def clip_manual_outlier(df, col, quantile, upper=True):
    '''clip manual outlier

    quantile: 0 ~ 1 e.g.)0.01
    '''

    Q = df[col].quantile(quantile)

    if upper:
        df.loc[:, col] = np.minimum(df[col], Q)
    else:
        df.loc[:, col] = np.maximum(df[col], Q)

    return df


def preprocess_dataset(df):
    """Preprocess the data (select columns and scale)
    ### MinMaxScaler: 正規化（値を0~1の範囲に）
    ### RobustScalar: 正規化（ただし、四分位範囲を分母とする）
    ### StandardScaler: 標準化（平均との差をとり標準偏差で割る）
    """
    # preproc = make_pipeline(MinMaxScaler(), StandardScaler(with_std=False))
    preproc = make_pipeline(RobustScaler(), StandardScaler(with_std=False))
    df_f = pd.DataFrame(preproc.fit_transform(df), columns=df.columns, index=df.index)

    return df_f


def normalize_dataset(df):
    """Preprocess the data (select columns and scale)
    ### MinMaxScaler: 正規化（値を0~1の範囲に）
    """
    preproc = make_pipeline(MinMaxScaler())
    df_f = pd.DataFrame(preproc.fit_transform(df), columns=df.columns, index=df.index)

    return df_f


def standardize_dataset(df):
    """Preprocess the data (select columns and scale)
    ### StandardScaler: 標準化（平均との差をとり標準偏差で割る）
    """
    preproc = make_pipeline(StandardScaler(with_std=False))
    df_f = pd.DataFrame(preproc.fit_transform(df), columns=df.columns, index=df.index)

    return df_f


def fit_PCA(df_train, df_test, col_list):
    """fit PCA

    Args:
        df_train (DataFrame): train data
        df_test (DataFrame): test data
        col_list (list): columns list for PCA

    Returns:
        pca (PCA): PCA class
        df_train_pca (DataFrame): result table for train data
        df_test_pca (DataFrame): result table for test data
        df_train_exp (DataFrame): ressult table of cumulative contribution rate
    """

    pca = PCA()
    pca.fit(df_train[col_list])

    train_pca = pca.transform(df_train[col_list])
    train_pca_cols = ["PCA{}".format(x + 1) for x in range(len(col_list))]
    df_train_pca = pd.DataFrame(train_pca, columns=train_pca_cols)

    test_pca = pca.transform(df_test[col_list])
    test_pca_cols = ["PCA{}".format(x + 1) for x in range(len(col_list))]
    df_test_pca = pd.DataFrame(test_pca, columns=test_pca_cols)

    df_train_exp = pd.DataFrame(pca.explained_variance_ratio_, index=train_pca_cols)

    return pca, df_train_pca, df_test_pca, df_train_exp


def concat_PCA_result(df, df_pca, col_num):

    if 'index' not in df.columns:
        df = df.reset_index()
        df = pd.concat(objs=[df, df_pca[["PCA{}".format(x + 1) for x in range(col_num)]]], axis=1)
        df = df.set_index('index')
    else:
        df = df.reset_index()
        df = pd.concat(objs=[df, df_pca[["PCA{}".format(x + 1) for x in range(col_num)]]], axis=1)
        df = df.set_index('level_0')

    return df
