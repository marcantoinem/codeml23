import dataset_loader
from soumission import write_csv, predictions
import random
import tqdm
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
    train, test = dataset_loader.read_csv("desjardins.csv")
    X_train, X_test, y_train, ids_test = dataset_loader.create_dataset_and_tokenizer(train, test)

    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)

    c = list(zip(X_train, y_train))
    random.shuffle(c)
    X_train, y_train = zip(*c)

    #model = DecisionTreeClassifier()
    #model = KNeighborsClassifier(1)
    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    data = predictions(model, X_test, ids_test)
    write_csv('result.csv', data)

    
if __name__ == '__main__':
    main()
