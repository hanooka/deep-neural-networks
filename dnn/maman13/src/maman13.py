import os
from enum import Enum

import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

RANDOM_STATE = 1337
AVB_TASKS = {'classification', 'regression'}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_fn_map = {
    'classification': nn.CrossEntropyLoss(),
    'regression': nn.MSELoss()
}

data_path = os.path.abspath(os.path.join(__file__, '../..', 'data'))
diabetes_path = os.path.join(data_path, 'diabetes.csv')


class Task(str, Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


class ANeuralNetwork(nn.Module):
    def __init__(self, task, input_dim, output_dim):
        super().__init__()
        task = task.lower()
        assert task in AVB_TASKS, f"task: {task} should be either {AVB_TASKS}"
        self.output_dim = output_dim
        if task == Task.REGRESSION:
            self.output_dim = 1
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        self.out = nn.Linear(16, self.output_dim)

    def forward(self, x):
        x = self.backbone(x)
        return self.out(x)

    def predict_proba(self, x):
        logits = self.forward(x)
        if self.task == Task.CLASSIFICATION:
            return torch.softmax(logits, dim=1)
        else:
            raise ValueError("probabilities <==> classification")

    def predict(self, x):
        if self.task == Task.CLASSIFICATION:
            return torch.argmax(self.forward(x), dim=1)
        else:
            return self.forward(x)


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
        self.y_cat = torch.tensor(diab_df['Class'].values, dtype=torch.int64, device=DEVICE)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx) -> tuple:
        """ returns, x, y_regression, y_categorical """
        return self.x[idx], self.y_reg[idx], self.y_cat[idx]


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


def validation_loop(model, val_loader, loss_fn, task):
    model.eval()
    val_loss = 0.
    correct = 0
    task = task.lower()
    assert task in AVB_TASKS, f"{task} should be in {AVB_TASKS}"

    with torch.no_grad():
        for x, y_reg, y_cls in val_loader:
            preds = model(x)
            y_true = y_cls if task == 'classification' else y_reg
            loss = loss_fn(preds, y_true)
            val_loss += loss.item()

            # TODO Calculate accuracy

    avg_val_loss = val_loss / len(val_loader)
    print(f"Val loss: {avg_val_loss:.4f}")
    return avg_val_loss


def train_model(
        model,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        task: str,
        epochs: int = 10,
        lr: float = 3e-4,
        wd: float = 0.05):
    task = task.lower()
    assert task in AVB_TASKS, f"{task} should be in {AVB_TASKS}"
    loss_fn = loss_fn_map[task]

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        running_loss = 0.
        model.train()
        for i, (x, y_reg, y_cls) in enumerate(train_loader):
            opt.zero_grad()
            preds = model(x)
            y_true = y_cls if task == 'classification' else y_reg
            loss = loss_fn(preds, y_true)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if i % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch + 1}/{epochs}], "
                      f"Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        avg_val_loss = validation_loop(model, valid_loader, loss_fn, task)


def question_8(n_cuts=10, exclude_Y=False, task='classification', epochs=500):
    """ We create our x using Y, and pred is Class `exclude_Y=False`
    Also notice that n_cuts determine amount of classes (and our output dim) """
    diabetes_df = preprocess_df(diabetes_path, n_cuts)

    # Stratification should be considered here.
    # Weight balancing can be ignored because of how we created the labels.
    train_diab_df, valid_diab_df = train_test_split(
        diabetes_df, test_size=0.2, random_state=RANDOM_STATE)

    diab_training_ds = DiabetesTabularDataset(train_diab_df, exclude_Y=exclude_Y)
    diab_training_dl = DataLoader(diab_training_ds, batch_size=10, shuffle=True)

    diab_validation_ds = DiabetesTabularDataset(valid_diab_df, exclude_Y=exclude_Y)
    diab_validation_dl = DataLoader(diab_validation_ds, batch_size=10, shuffle=True)

    x, y_reg, y_cat = next(iter(diab_training_dl))

    model = ANeuralNetwork(task='classification', input_dim=x.shape[1], output_dim=n_cuts)
    model.to(DEVICE)
    model = train_model(model, diab_training_dl, diab_validation_dl, task=task, epochs=epochs)


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
    question_8()
