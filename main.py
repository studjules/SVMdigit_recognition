import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class HandwrittenDigitRecognition:
    def __init__(self, data_train):
        self.data_train = data_train
        self.df = self.load_data()
        self.X, self.y = self.split_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()

    def load_data(self):
        df = pd.read_csv(self.data_train, header=None)
        return df

    def split_data(self):
        X = self.df.iloc[:, 1:].values
        y = self.df.iloc[:, 0].values
        return X, y

    def plot_data(self):
        fig = plt.figure(figsize=(20, 20))
        for i in range(10):
            ax = fig.add_subplot(1, 10, i + 1)
            ax.matshow(self.X[i].reshape((28, 28)), cmap=cm.gray)
            ax.set_title(self.y[i])
        plt.show()

    def scale_data(self):
        X = self.X / 255
        return X

    def split_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def train_SVM_model(self):
        model = SVC(kernel='linear', C=1)
        model.fit(self.X_train, self.y_train)
        return model

    def predict(self, model):
        y_pred = model.predict(self.X_test)
        return y_pred

    def evaluate(self, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def main(self):
        self.plot_data()
        self.X = self.scale_data()
        model = self.train_SVM_model()
        y_pred = self.predict(model)
        accuracy = self.evaluate(y_pred)
        print("Accuracy:", accuracy)

if __name__ == "__main__":
    data_train = "C:\\Users\\marce\\Desktop\\mnist_train.csv"
    digit_recognition = HandwrittenDigitRecognition(data_train)
    digit_recognition.main()

