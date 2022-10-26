# @File          :   benchmark.py
# @Last modified :   2022/10/24 19:48:54
# @Author        :   Matthias Gueltig

import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np



wine = datasets.load_wine()

# prepare data, -> binary classification problem
# create pandas dataframe
data = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns = wine['feature_names']+['target'])

# prepare dataset
data = data[data.target != 2.0]

data_X = data[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']]
data_Y = data[['target']]

X = data_X.to_numpy()
y = data_Y.to_numpy()

# 130, 13 features
n_samples, n_features = X.shape

accs = []
trainsizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
epochs = [10, 100, 1000]



metaresult = dict()
number_of_benchmarks = 50
for i in range(1,number_of_benchmarks+1):
    result = []
    for trainsize in trainsizes:

        for epoch_top in epochs:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainsize, random_state=1234)
            # scale
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            X_train = torch.from_numpy(X_train.astype(np.float32))
            X_test = torch.from_numpy(X_test.astype(np.float32))
            y_train = torch.from_numpy(y_train.astype(np.float32))
            y_test = torch.from_numpy(y_test.astype(np.float32))
            # make row to column vector
            y_train = y_train.view(y_train.shape[0], 1)
            y_test = y_test.view(y_test.shape[0], 1)
            # model
            # f = wx + b, sigmoid at the end

            class WineDecision(nn.Module):
                """class to build neural network.

                Args:
                    nn (nn.Module): Inheritance from pyTorch's nn.Module.
                """

                def __init__(self, n_input_features: int):
                    """constructor

                    Args:
                        n_input_features (int): features on which decision is based.
                    """
                    super(WineDecision, self).__init__()
                    self.linear = nn.Linear(n_input_features, 1)

                def forward(self, x):
                    # constrain values (0 <= y <= 1) -> reduce calculation power
                    y_predicted = torch.sigmoid(self.linear(x))
                    return y_predicted

            model = WineDecision(n_features)

            # loss and optimizer
            learning_rate = 0.01
            criterion = nn.BCELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            # training
            num_epochs = epoch_top
            for epoch in range(num_epochs):
                # forward pass and loss calculation
                y_predicted = model(X_train)
                loss = criterion(y_predicted, y_train)

                # backward pass
                loss.backward()

                # updates
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                X_test = X_test[:52,:]
                y_predicted = model(X_test)
                y_test = y_test[:52,:]
                # for each correct decision add +1
                acc = y_predicted.round().eq(y_test).sum() / float(y_test.shape[0])
                result.append([trainsize, epoch_top, acc.item()])

    result = np.array(result)
    resultdf = pd.DataFrame(result, columns=['train_size', 'epochs', 'acc'])
    metaresult[i] = resultdf

print(metaresult)
benchmark_result = []

for train_size in trainsizes:
    for epoch in epochs:
        for i, (run, results) in enumerate(metaresult.items()):
            values = np.ndarray(number_of_benchmarks)
            values[i] = results.loc[(results["train_size"] == train_size) & (results["epochs"] == epoch)]['acc'].item()
            print(values)
        benchmark_result.append([train_size, epoch, np.mean(values), np.std(values)])

benchmark_df = pd.DataFrame(benchmark_result, columns=['train_size', 'epochs', 'mean_acc', 'st_dev_acc'])

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
for epoch in epochs:
    benchmark_df_sel = benchmark_df.loc[benchmark_df["epochs"] == epoch]
    color = next(ax._get_lines.prop_cycler)['color']
    ax.errorbar(benchmark_df_sel["train_size"], benchmark_df_sel["mean_acc"], yerr=benchmark_df_sel["st_dev_acc"], color=color, label=epoch, alpha=0.8)
    ax.scatter(benchmark_df_sel["train_size"], benchmark_df_sel["mean_acc"], color=color, label=epoch)

ax.set_title("Benchmark DNN")
ax.set_xlabel("Fraction of data used for training [%]")
ax.set_ylabel("Accurancy of tested data [%]")
ax.legend(loc='best', title="# of epochs")
plt.show()
