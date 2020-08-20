import numpy as np
import pandas as pd


class NaiveBayes(object):
    '''
    统计学习方法 李航 例4.2
    '''

    def __init__(self, lambd=1):
        self.lambd = lambd  # Laplace smoothing
        self.y_count = None
        self.y_prob = None
        self.x_prob = {}  # conditional probability

    def fit(self, X_train, y):
        self.y_types = np.unique(y)
        X = pd.DataFrame(X_train)
        y = pd.DataFrame(y)
        self.y_count = y[0].value_counts()
        self.y_prob = (self.y_count + self.lambd) / \
            (y.shape[0]+len(self.y_types)*self.lambd)

        for idx in X.columns:
            for j in self.y_types:
                p_x_y = X[(y == j).values][idx].value_counts()
                for i in p_x_y.index:
                    self.x_prob[(idx, i, j)] = (p_x_y[i]+self.lambd) / \
                        (self.y_count[j]+p_x_y.shape[0]*self.lambd)

    def predict(self, X):
        res = []
        for y in self.y_types:
            p_xy = 1
            for idx, x in enumerate(X):
                p_xy *= self.x_prob[(idx, x, y)]
            res.append(p_xy*self.y_prob[y])
        for i in range(len(self.y_types)):
            print("[{}]'s probability is {:.2%}".format(
                self.y_types[i], res[i]))
        return self.y_types[np.argmax(res)]


if __name__ == "__main__":
    X_train = np.array([[1, 1], [1, 2], [1, 2], [1, 1], [1, 1],
                        [2, 1], [2, 2], [2, 2], [2, 3], [2, 3],
                        [3, 3], [3, 2], [3, 2], [3, 3], [3, 3]])  # S,M,L -> 1,2,3
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    x_predict = np.array([2, 1])

    nb = NaiveBayes(lambd=1)
    nb.fit(X_train, y)
    print(nb.predict(x_predict))

    '''
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import preprocessing

    encode = preprocessing.OneHotEncoder()  # necessary preprocessing
    encode.fit(X_train)
    X_train = encode.transform(X_train).toarray()
    x_predict = encode.transform(np.array([[2, 1]])).toarray()

    mnb = MultinomialNB()
    mnb.fit(X_train, y.reshape(len(y),))
    print(mnb.predict(x_predict))
    print(mnb.predict_proba(x_predict))
    '''
