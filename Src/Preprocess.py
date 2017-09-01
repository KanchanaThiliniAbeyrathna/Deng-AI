# =====================imports========================================

from __future__ import print_function
from __future__ import division

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA

from warnings import filterwarnings

filterwarnings('ignore')

# ===================================================================


def preprocess_data(features,data_path, labels_path=None):

    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    df = df[features]

    # fill missing values
    df.fillna(method='ffill', inplace=True)
    # df["reanalysis_avg_temp_k"].fillna(df["reanalysis_avg_temp_k"].mean(), inplace=True)
    # df["reanalysis_dew_point_temp_k"].fillna(df["reanalysis_dew_point_temp_k"].mean(), inplace=True)
    # df["reanalysi s_min_air_temp_k"].fillna(df["reanalysis_min_air_temp_k"].mean(), inplace=True)
    # df["reanalysis_max_air_temp_k"].fillna(df["reanalysis_max_air_temp_k"].mean(), inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)

    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']

    return sj, iq

pca_features = ['PC1', 'PC2', 'PC3', 'PC4']

def preprocess_data_pca(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    df.fillna(method='ffill', inplace=True)
    del df['week_start_date']
    df = cal_pca(df);

    df.diff().hist(color='r', alpha=0.5)
    plt.suptitle("PCA")
    plt.legend()
    plt.show()

    scaler = MinMaxScaler(feature_range=(0, 1))
    df[pca_features] = scaler.fit_transform(df[pca_features])

    df = binning(df,'PC1')
    df = binning(df, 'PC2')
    df = binning(df, 'PC3')
    df = binning(df, 'PC4')

    df.fillna(method='ffill', inplace=True)

    print(df.head())
    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)

    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']

    return sj, iq


# ===================================================================

def binning(df, feature):
    bins = [0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
    group_names = [0, 1, 2, 3, 4, 5, 6, 7]
    df['dummy_' + feature] = pd.cut(df[feature], bins, labels=group_names)
    del df[feature]
    return df;


def cal_pca(df):
    pca = PCA(copy=True, n_components=4, whiten=False)
    pca.fit(df)
    new_df_4d = pd.DataFrame(pca.transform(df))
    new_df_4d.index = df.index
    new_df_4d.columns = pca_features
    return new_df_4d;

