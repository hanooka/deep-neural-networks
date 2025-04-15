import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

RANDOM_STATE = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = os.path.abspath(os.path.join(__file__, '../..', 'data'))
diabetes_path = os.path.join(data_path, 'diabetes.csv')


class ANeuralNetwork(nn.Module):
    def __init__(self, task, input_dim, output_dim):
        super().__init__()
        task = task.lower()
        _tasks = {'classification', 'regression'}
        assert task in _tasks, f"task: {task} should be either {_tasks}"
        self.output_dim = output_dim
        if task == 'regression':
            self.output_dim = 1
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        if task == 'regression':
            self.out = nn.Linear(16, 1)
        elif task == 'classification':

            self.out = nn.Sequential(
                nn.Linear(16, self.output_dim),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.backbone(x)
        pred = self.out(x)
        return pred


class DiabetesTabularDataset(Dataset):
    def __init__(self, diab_df, exclude_Y=False):
        """ Implementing a custom case where we can put Y in and out of x.
        exclude_Y = True ==> Y will not be in x ("known" variables)
        We also return 2 kind of labels:
            One is for classification (Class)
            Other is for regression (Y)
        We can think about rescaling all variables.
        """
        self.orig_columns = diab_df.columns.values
        self.exclude_Y = exclude_Y
        drops = ['Class']

        if exclude_Y:
            drops = ['Y', 'Class']

        self.x = torch.tensor(diab_df.drop(labels=drops, axis=1).values, dtype=torch.float32, device=DEVICE)
        self.y_reg = torch.tensor(diab_df['Y'].values, dtype=torch.int16, device=DEVICE)
        self.y_cat = torch.tensor(diab_df['Class'].values, dtype=torch.int8, device=DEVICE)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx) -> tuple:
        """ returns, x, y_categorical, y_regression """
        return self.x[idx], self.y_cat[idx], self.y_reg[idx]


def preprocess_df(df_path, n_cuts: int = 10):
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



def questions_6_7():
    """ Question 6 and 7, we create a Dataloader.
    We create batches of size 10.
    We print one batch.
    """
    diabetes_df = preprocess_df(diabetes_path, 10)
    train_diab_df, test_diab_df = train_test_split(
        diabetes_df, test_size=0.2, random_state=RANDOM_STATE)

    diabetes_dataset = DiabetesTabularDataset(train_diab_df)
    diabetes_dataloder = DataLoader(diabetes_dataset, batch_size=10, shuffle=True)

    x, y_reg, y_cat = next(iter(diabetes_dataloder))
    # test all y_cat values are between 0 and 9 included

    print(x, y_reg, y_cat)
    print(len(diabetes_dataloder))


def training_loop(model, train_df, test_df):
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.05)


def question_8(n_cuts=10):
    """ We create our x using Y, and pred is Class `exclude_Y=False`
    Also notice that n_cuts determine amount of classes (and our output dim) """
    diabetes_df = preprocess_df(diabetes_path, n_cuts)
    train_diab_df, test_diab_df = train_test_split(
        diabetes_df, test_size=0.2, random_state=RANDOM_STATE)

    diabetes_dataset = DiabetesTabularDataset(train_diab_df, exclude_Y=False)
    diabetes_dataloder = DataLoader(diabetes_dataset, batch_size=10, shuffle=True)

    x, y_reg, y_cat = next(iter(diabetes_dataloder))

    model = ANeuralNetwork(task='classification', input_dim=x.shape[0], output_dim=n_cuts)
    model.to(DEVICE)
    model = training_loop(model, train_diab_df, test_diab_df)

def main():
    diabetes_df = preprocess_df(diabetes_path, 10)
    train_diab_df, test_diab_df = train_test_split(
        diabetes_df, test_size=0.2, random_state=RANDOM_STATE)

    diabetes_dataset = DiabetesTabularDataset(train_diab_df)
    diabetes_dataloder = DataLoader(diabetes_dataset, batch_size=10, shuffle=True)

    x, y_reg, y_cat = next(iter(diabetes_dataloder))
    # test all y_cat values are between 0 and 9 included

    print(x, y_reg, y_cat)
    print(len(diabetes_dataloder))

    model = ANeuralNetwork(task='classification', input_dim=10, output_dim=10)
    model.to(DEVICE)
    print(model)
    y_pred = model(x)
    print(y_pred)


if __name__ == '__main__':
    main()
