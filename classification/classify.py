import os
import time
import pickle
import argparse
from torch.utils.data import DataLoader
from dataset import DatasetFromDirectory
import numpy
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description="Spectral Classification Toolbox")
parser.add_argument('--data_root', type=str, default='../dataset_kiwi/classification/')
parser.add_argument('--fruit', type=str, default='kiwi')
opt = parser.parse_args()

def fit_model(model, X_train, X_test, y_train, y_test, model_name):
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("\n", model_name, ":")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Time taken:", time.time() - start)

def plot_losses(model, filename):
        plt.plot(model.loss_curve_)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.show()

def main():
        train_data = DatasetFromDirectory(data_root,"", fruit)
        print("Total Samples:", len(train_data))
        data = DataLoader(dataset=train_data)

        X, y = [],[]

        for sig, label in data:
                X.append(sig.squeeze().numpy())
                y.append(label.ravel().numpy())

        X = numpy.asarray(X)
        y = numpy.asarray(y)
        print("X, y",X.shape, y.shape)

        kf = StratifiedKFold(n_splits=4, random_state = 0, shuffle=True)

        scaler = MinMaxScaler()

        mlp = MLPClassifier(hidden_layer_sizes=(200,150,100), max_iter=300, activation='relu', solver='adam', alpha=0.0001)

        pipeline = Pipeline([('scaler', scaler), ('mlp', mlp)])
        for k, (train_index, val_index) in enumerate(kf.split(X,y)):
            print("Fold:",k)
            fit_model(pipeline, X[train_index], X[val_index], y[train_index], y[val_index], "MLP" + fruit + str(k))
            #plot_losses(mlp, "MLP")
            pickle.dump(pipeline, open(os.path.join("Models", "MLP_" + fruit + "_k" + str(k) + ".pkl"), "wb"))



if __name__ == "__main__":
        data_root = opt.data_root
        fruit = opt.fruit
        main()
