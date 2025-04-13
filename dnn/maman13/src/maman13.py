import os
import pandas as pd
import numpy as np
import torch


data_path = os.path.abspath(os.path.join(__file__, '../..', 'data'))
diabetes_path = os.path.join(data_path, 'diabetes.csv')


def preprocess_df(df_path):
    # reading df, tab seperated
    diabetes_df = pd.read_csv(df_path, sep='\t')

    # Calculating quantiles (which won't help us,
    # since pandas got a better way to classify the quantiles)
    y_quantiles = diabetes_df[['Y']].quantile(q=np.arange(0.1, 1.1, 0.1))

    # Classify the quantiles, renaming column to "Class"
    y_categorical = pd.qcut(diabetes_df['Y'], 10, labels=False)
    y_categorical = y_categorical.rename("Class")

    diabetes_df = pd.concat([diabetes_df, y_categorical], axis=1)
    return diabetes_df


def main():
    diabetes_df = preprocess_df(diabetes_path)


if __name__ == '__main__':
    main()


