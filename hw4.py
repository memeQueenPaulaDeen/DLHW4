import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class myLogistic(LogisticRegression):


    def fit(self,X,y,num_steps=100,learningRate=.001,plot = False):


        costVect = []
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        self.classes_ = np.unique(y)
        lam = .1

        X0 = np.ones(len(X))  # column of ones for the theta0 feature
        #self.intercept_ = -10
        self.intercept_ = 0
        #self.intercept_ = np.random.randn(1)[0]
        #self.coef_ = np.random.randn(len(X[0]))*np.sqrt(2/len(X[0]))
        self.coef_ = np.random.randn(len(X[0]))
        #self.coef_ = np.zeros(len(X[0]))
        self.coef_ = self.coef_.reshape(1, len(X[0]))

        for x in range(num_steps):
            #self.coef_ = np.array(self.coef_ - (learningRate*1/len(y)*(self.sigmoid(X,self.coef_).reshape(y.shape)-y))@X  +lam/len(y)* self.coef_).reshape(1,2)

            idx = 0
            for singleDataPoint in X:

                ypredVect= self.predict(X)#self.sigmoid(X, self.coef_).reshape(y.shape)
                ypred = ypredVect[idx]
                self.coef_ = self.coef_ - np.array((learningRate * 1 / len(y) * (ypred - y[idx])) * singleDataPoint).reshape(1, len(X[0]))
                self.intercept_ = self.intercept_ - learningRate * 1 / len(y) * (ypred - y[idx]) #* 1 implied
                cost = self.cost(X,y)
                costVect.append(cost)
                idx = idx+1

            # lookback = 2
            # if x > lookback:
            #     last = costVect[-lookback]
            #     if cost > last:
            #         learningRate = learningRate/2



        if plot:
            plt.plot(costVect)
            plt.xlabel('Number of iterations')
            plt.ylabel('Cost')
            plt.title("Cost vs Iterations for learning rate: " +str(learningRate))
            plt.show()

        return self


    def sigmoid(self,X,theta):
        if np.isscalar(theta):
            return 1/(1+np.exp(-X*theta))
        theta = theta.reshape(theta.shape)
        return 1/(1+np.exp(-X@theta.T))

    def cost(self,X,y):

        def sigmoid(Z):
            return 1 / (1 + np.exp(-Z))

        return -(y@np.log(sigmoid(self.predict(X)))+(1-y)@np.log(1-sigmoid(self.predict(X))))

if __name__ == '__main__':

    size  = 200
    dist1 = np.random.multivariate_normal([3,3],[[1,7],[7,1]],size)
    dist2 = np.random.multivariate_normal([8,12],[[2,3],[3,2]],size)

    df1 = pd.DataFrame(dist1,columns=['x1','x2'])
    df1['y'] = 0

    df2 = pd.DataFrame(dist2,columns=['x1','x2'])
    df2['y'] = 1

    df = df1.append(df2)
    plt.scatter(df.x1,df.x2,c=df.y)
    plt.xlabel('feature x1')
    plt.ylabel('feature x2')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
    df[['x1','x2']], df['y'], test_size = 0.2, random_state = np.random.randint(50))

    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    skpred = clf.predict(X_test)
    print("sk score:" + (str(clf.score(X_test,y_test))))
    print("sk params:" + (str(clf.coef_)))
    print("sk intercept:" + (str(clf.intercept_)))
    print()


    for alpha in ([.1,.01,.001]):
        myclf = myLogistic()

        myclf.fit(X_train, y_train,300,alpha,plot=True)
        mypred = myclf.predict(X_test)
        print('alpha: ' + str(alpha))
        print("my score:" + (str(myclf.score(X_test, y_test))))
        print("my params:" + (str(myclf.coef_)))
        print("my intercept:" + (str(myclf.intercept_)))
        print()

    print('\n\n\n')
    print('Classifications for Iris data set:')
    X, y = load_iris(return_X_y=True)
    X = X[np.logical_or(y==0 , y==1)]
    y = y[np.logical_or(y==0 , y==1)]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = np.random.randint(50))

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    skpred = clf.predict(X_test)
    print("sk score:" + (str(clf.score(X_test, y_test))))
    print()

    for alpha in ([.1, .01, .001]):
        myclf = myLogistic()
        myclf.fit(X_train, y_train, 300, alpha, plot=True)
        mypred = myclf.predict(X_test)
        print("my score:" + (str(myclf.score(X_test, y_test))))
        print()

