import numpy as np


class PerceptronDual(object):
    '''
    统计学习方法 李航 例2.2
    '''

    def __init__(self, tau=1):
        self.tau = tau

    def fit(self, x, y):
        self.b = 0
        self.alpha = np.zeros(y.shape)
        self.gram = x.dot(x.T)
        index = 0
        while index < x.shape[0]:
            if y[index]*(sum(self.alpha*y*self.gram[index]) + self.b) <= 0:
                self.alpha[index] += self.tau
                self.b += y[index]*self.tau
                index = 0
            else:
                index += 1
        print("alpha="+str(self.alpha))
        print("w="+str((x.T).dot(y*self.alpha)))
        print("b="+str(self.b))


if __name__ == "__main__":

    x_train = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])

    per = PerceptronDual()
    per.fit(x_train, y)
    '''
    from sklearn.linear_model import Perceptron
    percep = Perceptron()
    percep.fit(x_train, y)
    print(percep.coef_, percep.intercept_, percep.n_iter_)
    res = percep.score(x_train, y)
    print(res)
    '''
