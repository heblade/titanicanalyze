from __future__ import division
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def load_data(file_name, is_train):
    data = pd.read_csv(file_name)
    if 'Survived' not in data.columns:
        data['Survived'] = None
    #性别
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)

    #补齐船票价格缺失值
    if len(data.Fare[data.Fare.isnull()]) > 0:
        fare = np.zeros(3)
        for f in range(0, 3):
            fare[f] = data[data.Pclass == f + 1]['Fare'].dropna().median()
        for f in range(0, 3):
            data.loc[(data.Fare.isnull()) & (data.Pclass == f + 1), 'Fare'] = fare[f]

    print('随机森林预测缺失年龄: --start--')
    if is_train:
        data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    else:
        data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    age_exist = data_for_age.loc[(data.Age.notnull())]
    age_null = data_for_age.loc[(data.Age.isnull())]
    # print(age_exist)

    x = age_exist.values[:, 1:]
    y = age_exist.values[:, 0]
    rfr = RandomForestRegressor(n_estimators=1000)
    rfr.fit(x, y)
    age_hat = rfr.predict(age_null.values[:, 1:])
    # print(age_hat)
    data.loc[(data.Age.isnull()), 'Age'] = age_hat
    print('随机森林预测缺失年龄: --end--')

    return data[['Age', 'Sex', 'Fare', 'Parch', 'SibSp', 'Pclass']].values, \
           data[['Survived']].values, data

def startjob():
    x_train, y_train, dftrain = load_data('./data/train.csv', True)
    x_test, y_test, dftest = load_data('./data/test.csv', False)
    print(dftest.head())
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state=0)
    # print(y_test)

    if not os.path.exists('./output'):
        os.mkdir('./output')
    outputcsv = './output/%s.csv'
    #Logistc回归
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train)
    # y_hat = lr.predict(x_test)
    y_hat = lr.predict(x_train)
    print('Logistic回归: ', y_hat)
    dftrain['Survived_hat'] = y_hat
    dftrain.to_csv(outputcsv % ('PredictbyLogisticRegression'), encoding='utf-8')
    show_accuracy(y_hat, y_train, 'Logistic回归')

    #随机森林
    rfc = RandomForestClassifier(n_estimators= 100)
    rfc.fit(x_train, y_train)
    # y_hat = rfc.predict(x_test)
    y_hat = rfc.predict(x_train)
    print('随机森林: ', y_hat)
    dftrain['Survived_hat'] = y_hat
    dftrain.to_csv(outputcsv % ('PredictbyRandomForestClassifier'), encoding='utf-8')
    show_accuracy(y_hat, y_train, '随机森林')

    #XGBoost
    data_train = xgb.DMatrix(x_train, label=y_train)
    # data_test = xgb.DMatrix(x_test, label=y_test)
    # watch_list = [(data_test, 'eval'), (data_train, 'train')]
    watch_list = [(data_train, 'train')]
    param = {'max_depth':3, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic'}
    bst = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)
    # y_hat = bst.predict(data_test)
    y_hat = bst.predict(data_train)
    print('XGBoost: ', y_hat)
    # dftest['Survived'] = y_hat
    dftrain['Survived_hat'] = y_hat
    dftrain.to_csv(outputcsv % ('PredictbyXGBoost'), encoding='utf-8')
    show_accuracy(y_hat, y_train, 'XGBoost')

def show_accuracy(y_hat, y_test, algorithmname):
    if(len(y_hat) != len(y_test)):
        print("The length: %d of y_hat is different with the length: %d of y_test",
              y_hat,
              y_test)
    else:
        total = len(y_hat)
        rightcount = 0
        for i in range(total):
            if(round(y_hat[i]) == round(y_test[i][0])):
                rightcount += 1
        rightratio = (rightcount / total) * 100
        print('%s准确率: %f%%' % (algorithmname, rightratio))


if __name__ == '__main__':
    startjob()