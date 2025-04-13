import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


data_path = os.path.abspath(os.path.join(__file__, '../..', 'data'))
diabetes_path = os.path.join(data_path, 'diabetes.csv')


class DiabetesTabularDataset(Dataset):
    def __init__(self, df_path, n_cuts):
        self.data = preprocess_df(df_path, n_cuts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]


def preprocess_df(df_path, n_cuts: int=10):
    # reading df, tab seperated
    diabetes_df = pd.read_csv(df_path, sep='\t')

    # Calculating quantiles (which won't help us,
    # since pandas got a better way to classify the quantiles)
    # y_quantiles = diabetes_df[['Y']].quantile(q=np.arange(0.1, 1.1, 0.1))

    # Classify the quantiles, renaming column to "Class"
    y_categorical = pd.qcut(diabetes_df['Y'], n_cuts, labels=False)
    y_categorical = y_categorical.rename("Class")

    diabetes_df = pd.concat([diabetes_df, y_categorical], axis=1)
    return diabetes_df


def main():
    diabetes_df = preprocess_df(diabetes_path, 10)
    dataset = DiabetesTabularDataset(diabetes_path, 100)
    for d in dataset:
        print(d)


if __name__ == '__main__':
    main()


