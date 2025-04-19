import os
import torch
import pandas as pd

from enum import Enum
from torch import nn
from torcheval.metrics import MulticlassAccuracy, R2Score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

RANDOM_STATE = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Task(str, Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


loss_fn_map = {
    Task.CLASSIFICATION: nn.CrossEntropyLoss(),
    Task.REGRESSION: nn.MSELoss()
}

metric_map = {
    Task.CLASSIFICATION: (MulticlassAccuracy, "acc"),
    Task.REGRESSION: (R2Score, "r2")
}

AVB_TASKS = {Task.CLASSIFICATION, Task.REGRESSION}

data_path = os.path.abspath(os.path.join(__file__, '../..', 'data'))
diabetes_path = os.path.join(data_path, 'diabetes.csv')


class ANeuralNetwork(nn.Module):
    def __init__(self, task, input_dim, output_dim):
        super().__init__()
        task = task.lower()
        assert task in AVB_TASKS, f"task: {task} should be either {AVB_TASKS}"
        self.output_dim = output_dim
        if task == Task.REGRESSION:
            self.output_dim = 1
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.out = nn.Linear(1024, self.output_dim)

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
        self.y_reg = torch.tensor(diab_df['Y'].values, dtype=torch.float32, device=DEVICE)
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


def validation_loop(model, val_loader, loss_fn, task) -> (float, float):
    """ run validation on val_loader, using loss_fn and metric using metric map and task.
    returns the avg_loss, and metric value. """
    val_loss = 0.
    assert task in AVB_TASKS, f"{task} should be in {AVB_TASKS}"
    metric, metric_name = metric_map[task]
    metric = metric(device=DEVICE)
    model.eval()
    with torch.no_grad():
        for x, y_reg, y_cls in val_loader:
            preds = model(x).squeeze()
            y_true = y_cls if task == 'classification' else y_reg
            loss = loss_fn(preds, y_true)
            metric.update(preds, y_true)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, metric.compute()


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        task: Task,
        epochs: int = 10,
        verbose: int = 1,
        verbose_batch: int = 1,
        lr: float = 1e-4,
        wd: float = 0.05) -> nn.Module:
    """
    Given train/validation set, train the model `epochs` epochs, and validates at each epoch over
    the validation set.
    Required metric is Accuracy.

    :param model:
    :param train_loader:
    :param valid_loader:
    :param task: Task (currently 'classification' or 'regression')
    :param epochs:
    :param verbose: [0, 1, 2] Level of printing information (0 None, 2 Max)
    :param verbose_batch: if verbose is 2, how many batches before printing metrices and loss.
    :param lr: learning rate
    :param wd: weight decay
    :return: a model
    """

    assert task in AVB_TASKS, f"{task} should be in {AVB_TASKS}"
    loss_fn = loss_fn_map[task]
    metric, metric_name = metric_map[task]
    metric = metric(device=DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        running_loss = 0.
        model.train()
        metric.reset()
        for i, (x, y_reg, y_cls) in enumerate(train_loader):
            opt.zero_grad()
            preds = model(x).squeeze()
            y_true = y_cls if task == Task.CLASSIFICATION else y_reg
            loss = loss_fn(preds, y_true)
            metric.update(preds, y_true)
            loss.backward()
            opt.step()
            running_loss += loss.item()

            # Print every `verbose_batch` batches
            if verbose >= 2 and i % verbose_batch == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], "
                      f"Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}", sep=',')

        # End of epoch. Run validation and print outcomes
        avg_val_loss, metric_val = validation_loop(model, valid_loader, loss_fn, task)
        if verbose >= 1:
            print(f"Epoch [{epoch + 1:4}/{epochs}]", end=f", ")
            print(f"trn los: {running_loss / (epoch + 1):8.4f},", f"trn {metric_name}: {metric.compute():6.4f}",
                  end=', ')
            print(f"val loss: {avg_val_loss:8.4f}, val {metric_name}: {metric_val:6.4f}")

    return model


def general_solver(n_cuts=10,
                   exclude_Y=False,
                   task=Task.CLASSIFICATION,
                   epochs=100, verbose=1,
                   verbose_batch=10,
                   lr: float = 1e-4):
    """ we read the diabetes dataset, and create the "Class" column using n_cuts
    We split to train and test (0.2)
    We wrap the dataset with our custom DiabetesTabularDataset
    and generate a DataLoader wrapping the dataset. batch_size is 10 and shuffle is True
    We initialize (and instantiate) the neural network based on data (x.shape[1]) and n_cuts - which is
    the output dim.
    notice that n_cuts determine amount of classes (and our output dim)
    We train, and watch resulted printed to screen. """
    diabetes_df = preprocess_df(diabetes_path, n_cuts)

    # Stratification should be considered here.
    # Weight balancing can be semi-ignored because of how we created the labels.
    train_diab_df, valid_diab_df = train_test_split(
        diabetes_df, test_size=0.2, random_state=RANDOM_STATE)

    diab_training_ds = DiabetesTabularDataset(train_diab_df, exclude_Y=exclude_Y)
    diab_training_dl = DataLoader(diab_training_ds, batch_size=10, shuffle=True)

    diab_validation_ds = DiabetesTabularDataset(valid_diab_df, exclude_Y=exclude_Y)
    diab_validation_dl = DataLoader(diab_validation_ds, batch_size=10, shuffle=True)

    x, y_reg, y_cat = next(iter(diab_training_dl))

    model = ANeuralNetwork(task=task, input_dim=x.shape[1], output_dim=n_cuts)
    model.to(DEVICE)
    model = train_model(model, diab_training_dl, diab_validation_dl, task=task, epochs=epochs, verbose=verbose,
                        verbose_batch=verbose_batch, lr=lr)


def question_8():
    general_solver(n_cuts=10, exclude_Y=False, task=Task.CLASSIFICATION, epochs=100)
    # Epoch [   1/100], trn los: 190.9444, trn acc: 0.1530, val loss:   2.3679, val acc: 0.2472
    # Epoch [ 100/100], trn los:   0.3279, trn acc: 0.5609, val loss:   0.9889, val acc: 0.5730


def question_9():
    general_solver(n_cuts=10, exclude_Y=True, task=Task.CLASSIFICATION, epochs=500)
    # Epoch [   5/500], trn los:  21.1454, trn acc: 0.1133, val loss:   2.5425, val acc: 0.0787
    # Epoch [ 500/500], trn los:   0.1397, trn acc: 0.2975, val loss:   2.0022, val acc: 0.2135


def question_12():
    general_solver(n_cuts=100, exclude_Y=False, task=Task.CLASSIFICATION, epochs=200, lr=3e-5)
    # We lowered the learning rate a bit here, We also increased the dropout % for this training session
    # Epoch [   1/200], trn los: 470.0365, trn acc: 0.0057, val loss:   6.4315, val acc: 0.0225
    # Epoch [ 200/200], trn los:   0.5688, trn acc: 0.1161, val loss:   3.5989, val acc: 0.1124


def question_13():
    general_solver(n_cuts=100, exclude_Y=True, task=Task.CLASSIFICATION, epochs=200, lr=1e-4)
    # Epoch [   1/200], trn los: 311.1128, trn acc: 0.0113, val loss:   5.0817, val acc: 0.0000
    # Epoch [  67/200], trn los:   2.2001, trn acc: 0.0397, val loss:   4.9264, val acc: 0.0337
    # Epoch [ 200/200], trn los:   0.3902, trn acc: 0.3399, val loss:   7.6031, val acc: 0.0000

    # So we can see a beautiful overfit here. Where we started with 1% accuracy (which fits 100 cuts)
    # At our best (around epoch 67, we can see the train acc is about 4% and validation at 3.3%)
    # After that we overfit and get a great 34% accuracy which worth nothing because validation is 0.


def question_14():
    general_solver(n_cuts=1, exclude_Y=True, task=Task.REGRESSION, epochs=1000, lr=1e-4, verbose=1)
    # Epoch [   1/1000], trn los: 331214.6368, trn r2: -0.5712, val loss: 5517.4431, val r2: 0.0567
    # Epoch [ 100/1000], trn los: 1174.3142, trn r2: 0.4493, val loss: 3013.6842, val r2: 0.4818
    # Epoch [ 728/1000], trn los: 121.1772, trn r2: 0.5838, val loss: 2487.0921, val r2: 0.5720
    # Epoch [1000/1000], trn los:  73.8316, trn r2: 0.6607, val loss: 2743.2695, val r2: 0.5307

    # We can see we start the train terrible with a huge loss and negative r2.
    # In epoch 100, we can see we’ve improved and not overfitting yet (train and validation at the same level)
    # In epoch 728 we can see we kept improving, and this is the maximum r2 and best epoch to stop.
    # In epoch 1000 we can see the overfit with trn r2 >> val r2.


def main():
    qs = [question_8, question_9, question_12, question_13, question_14]
    for q in qs:
        q()


if __name__ == '__main__':
    main()
