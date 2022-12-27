import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def run(X, y, kfold=None):
    LR_model = LinearRegression()
    RFR_model = RandomForestRegressor()
    LGBMR_model = LGBMRegressor()

    if kfold:
        print('Linear Regression')
        print(
            f'MAE {kfold}: {np.mean(-cross_val_score(LR_model, X, y, scoring="neg_mean_absolute_error", cv=kfold)):5f}')
        print(
            f'MSE {kfold}: {np.mean(-cross_val_score(LR_model, X, y, scoring="neg_mean_squared_error", cv=kfold)):5f}')
        print(
            f'RMSE {kfold}: {np.mean(-cross_val_score(LR_model, X, y, scoring="neg_root_mean_squared_error", cv=kfold)):5f}')
        print('')

        print('Random Forest Regression')
        print(
            f'MAE {kfold}: {np.mean(-cross_val_score(RFR_model, X, y, scoring="neg_mean_absolute_error", cv=kfold)):5f}')
        print(
            f'MSE {kfold}: {np.mean(-cross_val_score(RFR_model, X, y, scoring="neg_mean_squared_error", cv=kfold)):5f}')
        print(f'RMSE {kfold}: {np.mean(-cross_val_score(RFR_model, X, y, scoring="neg_root_mean_squared_error", cv=kfold)):5f}')
        print('')

        print('Light Gradient Boosting Machine Regression')
        print(
            f'MAE {kfold}: {np.mean(-cross_val_score(LGBMR_model, X, y, scoring="neg_mean_absolute_error", cv=kfold)):5f}')
        print(
            f'MSE {kfold}: {np.mean(-cross_val_score(LGBMR_model, X, y, scoring="neg_mean_squared_error", cv=kfold)):5f}')
        print(f'RMSE {kfold}: {np.mean(-cross_val_score(LGBMR_model, X, y, scoring="neg_root_mean_squared_error", cv=kfold)):5f}')
        print('')
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
            
        print('Linear Regression')
        LR_model.fit(X_train, y_train)
        y_pred = LR_model.predict(X_test)
        print(f'MAE: {mean_absolute_error(y_test, y_pred):5f}')
        print(f'MSE: {mean_squared_error(y_test, y_pred):5f}')
        print(f'RMSE: {root_mean_squared_error(y_test, y_pred):5f}')
        print('')

        print('Random Forest Regression')
        RFR_model.fit(X_train, y_train)
        y_pred = RFR_model.predict(X_test)
        print(f'MAE: {mean_absolute_error(y_test, y_pred):5f}')
        print(f'MSE: {mean_squared_error(y_test, y_pred):5f}')
        print(f'RMSE: {root_mean_squared_error(y_test, y_pred):5f}')
        print('')

        print('Light Gradient Boosting Machine Regression')
        LGBMR_model.fit(X_train, y_train)
        y_pred = LGBMR_model.predict(X_test)
        print(f'MAE: {mean_absolute_error(y_test, y_pred):5f}')
        print(f'MSE: {mean_squared_error(y_test, y_pred):5f}')
        print(f'RMSE: {root_mean_squared_error(y_test, y_pred):5f}')
        print('')


# Load data
columns = ['Year', 'Month', 'Day', 'H_KG',
           'H_LT', 'H_DH', 'R_KG', 'R_LT', 'R_DH']
df = pd.read_excel('Dataset_floodseason_76_20_NEW_3.11.xlsx',
                   skiprows=range(0, 2), header=None)
df.columns = columns
years = df.Year.unique()

# Case 1
# x: H (t-3, t-2, t-1, t) => 12
# y: H (t+1)
print("==> Case 1 <==")
k = 4
X = []
y = []

for year in years:
    dt = df[df.Year == year]
    max_len = len(dt)-k

    x = np.empty((max_len, 12))
    for i in range(max_len):
        dx = dt.iloc[i:i+k][['R_KG', 'R_LT', 'R_DH']].values
        dx = dx.reshape(1, -1)
        x[i] = dx
    X.append(x)
    y.append(dt.iloc[k:].H_KG.values)

X = np.vstack(X)
y = np.hstack(y)

run(X, y)
run(X, y, kfold=5)

# Case 2
# x: H (t-3, t-2, t-1, t) | R (t-3, t-2, t-1, t) => 24
# y: H (t+1)
print("==> Case 2 <==")
k = 4
X = []
y = []

for year in years:
    dt = df[df.Year == year]
    max_len = len(dt)-k

    x = np.empty((max_len, 24))
    for i in range(max_len):
        dx = dt.iloc[i:i+k][['R_KG', 'R_LT',
                             'R_DH', 'H_KG', 'H_LT', 'H_DH']].values
        dx = dx.reshape(1, -1)
        x[i] = dx
    X.append(x)
    y.append(dt.iloc[k:].H_KG.values)

X = np.vstack(X)
y = np.hstack(y)

run(X, y)
run(X, y, kfold=5)

# Case 3
# x: H (t-3, t-2, t-1, t) | R (t-3, t-2, t-1, t, t+1) => 27
# y: H (t+1)
print("==> Case 3 <==")
k = 4
X = []
y = []

for year in years:
    dt = df[df.Year == year]
    max_len = len(dt)-k

    x = np.empty((max_len, 27))
    for i in range(max_len):
        dx1 = dt.iloc[i:i+k][['R_KG', 'R_LT',
                              'R_DH', 'H_KG', 'H_LT', 'H_DH']].values
        dx1 = dx1.reshape(1, -1)
        dx2 = dt.iloc[i+k][['R_KG', 'R_LT', 'R_DH']].values
        dx2 = dx2.reshape(1, -1)
        x[i] = np.hstack((dx1, dx2))
    X.append(x)
    y.append(dt.iloc[k:].H_KG.values)

X = np.vstack(X)
y = np.hstack(y)

run(X, y)
run(X, y, kfold=5)
