import os
import time
import pickle

from torch.utils.data import DataLoader
from dataset import DatasetFromDirectory
import numpy
import sklearn
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

def fit_model(model, X_train, X_test, y_train, y_test, model_name):
        start = time.time()
        model.fit(X_train, y_train)
        pickle.dump(model, open(os.path.join("Models", model_name.split(" ")[0]+"_akb.pkl"), "wb"))
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
        train_data = DatasetFromDirectory(os.path.join("../dataset_apple/classification/"), "")
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
        pickle.dump(scaler, open(os.path.join("Models", "MLP_scalar.pkl"), "wb"))

        mlp = MLPClassifier(hidden_layer_sizes=(200,150,100), max_iter=300, activation='relu', solver='adam', alpha=0.0001)

        for k, (train_index, val_index) in enumerate(kf.split(X,y)):
            scaler.fit(X[train_index])
            X[train_index] = scaler.transform(X[train_index])
            X[val_index] = scaler.transform(X[val_index])
            print("Fold:",k)
            fit_model(mlp, X[train_index], X[val_index], y[train_index], y[val_index], "MLP 3 Hidden Layers, each with 100 neurons, 300 Iterations")
            #plot_losses(mlp, "MLP")


if __name__ == "__main__":
        main()
