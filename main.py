import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def run(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    LR_model = LinearRegression()
    LR_model.fit(X_train, y_train)
    y_pred = LR_model.predict(X_test)
    print('LR MSE:', mean_squared_error(y_test, y_pred))
    print('LR MAE:', mean_absolute_error(y_test, y_pred))
    print('KFold LR MSE:', np.mean(-cross_val_score(LR_model,
          X, y, scoring='neg_mean_squared_error', cv=10)))
    print('KFold LR MAE:', np.mean(-cross_val_score(LR_model,
          X, y, scoring='neg_mean_absolute_error', cv=10)))

    RFR_model = RandomForestRegressor()
    RFR_model.fit(X_train, y_train)
    y_pred = RFR_model.predict(X_test)
    print('RFR MSE:', mean_squared_error(y_test, y_pred))
    print('RFR MAE:', mean_absolute_error(y_test, y_pred))
    print('KFold RFR MSE:', np.mean(-cross_val_score(RFR_model,
          X, y, scoring='neg_mean_squared_error', cv=10)))
    print('KFold RFR MAE:', np.mean(-cross_val_score(RFR_model,
          X, y, scoring='neg_mean_absolute_error', cv=10)))
    print('')


# Load data
df = pd.read_excel('Dataset_floodseason_76_20_NEW_3.11.xlsx',
                   skiprows=range(0, 2), header=None)

print('Case 1')
X = df.iloc[:, [4, 5]].values
y = df.iloc[:, 3].values
run(X, y)

print('Case 2')
X = df.iloc[:, [7, 8]].values
y = df.iloc[:, 3].values
run(X, y)

print('Case 3')
X = df.iloc[:, [4, 5, 7, 8]].values
y = df.iloc[:, 3].values
run(X, y)
