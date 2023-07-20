import pickle
import sklearn
import argparse
from torch.utils.data import DataLoader
from dataset_test import DatasetFromDirectory
import numpy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description="Spectral Classification Toolbox")
parser.add_argument('--data_root', type=str, default='../dataset_kiwi/classification/')
parser.add_argument('--fruit', type=str, default='kiwi')
parser.add_argument('--pretrained_classifier', type=str, default='./pretrained_classifier/MLP_kiwi.pkl')
opt = parser.parse_args()


def main():
        train_data = DatasetFromDirectory(data_root,"", fruit)
        #print("Total Samples:", len(train_data))
        data = DataLoader(dataset=train_data)

        X, y = [],[]

        for sig, label in data:
                X.append(sig.squeeze().numpy())
                y.append(label.ravel().numpy())

        X = numpy.asarray(X)
        y = numpy.asarray(y)

        
        pipeline = pickle.load(open(model,'rb'))
        y_pred = pipeline.predict(X)
        print("Classification Accuracy (" + fruit + "):", accuracy_score(y, y_pred))
        #print("Cls Report:",classification_report(y, y_pred))


if __name__ == "__main__":
        data_root = opt.data_root
        fruit = opt.fruit
        model = opt.pretrained_classifier
        main()
