import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor


def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def NSE(y_true, y_pred):
    return 1-(np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_pred))**2))

def NSE_scorer(model, X, y):
    y_pred = model.predict(X)
    return NSE(y, y_pred)

def calculate_model(model, X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f'NSE: {NSE(y_test, y_pred):5f}')
    print(f'R2: {r2_score(y_test, y_pred):5f}')
    print(f'MAE: {mean_absolute_error(y_test, y_pred):5f}')
    print(f'MSE: {mean_squared_error(y_test, y_pred):5f}')
    print(f'RMSE: {RMSE(y_test, y_pred):5f}')
    print('')

def calculate_kfold(model, X, y, kfold):
    print(
        f'NSE {kfold}: {np.mean(cross_val_score(model, X, y, scoring=NSE_scorer, cv=kfold)):5f}')
    print(
        f'R2 {kfold}: {np.mean(cross_val_score(model, X, y, scoring="r2", cv=kfold)):5f}')
    print(
        f'MAE {kfold}: {np.mean(-cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=kfold)):5f}')
    print(
        f'MSE {kfold}: {np.mean(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfold)):5f}')
    print(
        f'RMSE {kfold}: {np.mean(-cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=kfold)):5f}')
    print('')

def run(X, y, kfold=None):
    LR_model = LinearRegression()
    RFR_model = RandomForestRegressor()
    LGBMR_model = LGBMRegressor()

    if kfold:
        print('Linear Regression')
        calculate_kfold(LR_model, X, y, kfold)

        print('Random Forest Regression')
        calculate_kfold(RFR_model, X, y, kfold)

        print('Light Gradient Boosting Machine Regression')
        calculate_kfold(LGBMR_model, X, y, kfold)
    else:
        print('Linear Regression')
        calculate_model(LR_model, X, y)

        print('Random Forest Regression')
        calculate_model(RFR_model, X, y)

        print('Light Gradient Boosting Machine Regression')
        calculate_model(LGBMR_model, X, y)


# Load data
df = pd.read_excel('Dataset_floodseason_76_20_NEW_3.11.xlsx',
                   skiprows=range(0, 2), header=None)
columns = ['Year', 'Month', 'Day', 'MN_KG',
           'MN_LT', 'MN_DH', 'LM_KG', 'LM_LT', 'LM_DH']
df.columns = columns
years = df.Year.unique()

# Case 1
# x: MN_KG,LT,DH (t-3, t-2, t-1, t) => 3 * 4 = 12
# y: MN_KG (t+1)
print("==> Case 1 <==")
k = 4
X = []
y = []

for year in years:
    dt = df[df.Year == year]
    max_len = len(dt)-k

    x = np.empty((max_len, 12))
    for i in range(max_len):
        dx = dt.iloc[i:i+k][['MN_KG', 'MN_LT', 'MN_DH']].values
        dx = dx.reshape(1, -1)
        x[i] = dx
    X.append(x)
    y.append(dt.iloc[k:].MN_KG.values)

X = np.vstack(X)
y = np.hstack(y)

run(X, y, kfold=None)
# run(X, y, kfold=2)


# Case 2
# x: LM_KG,LT,DH (t-3, t-2, t-1, t) => 3 * 4 = 12
# y: MN_KG (t+1)
print("==> Case 2 <==")
k = 4
X = []
y = []

for year in years:
    dt = df[df.Year == year]
    max_len = len(dt)-k

    x = np.empty((max_len, 12))
    for i in range(max_len):
        dx = dt.iloc[i:i+k][['LM_KG', 'LM_LT', 'LM_DH']].values
        dx = dx.reshape(1, -1)
        x[i] = dx
    X.append(x)
    y.append(dt.iloc[k:].MN_KG.values)

X = np.vstack(X)
y = np.hstack(y)

run(X, y, kfold=None)
# run(X, y, kfold=2)

# Case 3
# x: MN_KG,LT,DH (t-3, t-2, t-1, t) | LM_KG,LT,DH (t-3, t-2, t-1, t) => 6 * 4 = 24
# y: MN_KG (t+1)
print("==> Case 3 <==")
k = 4
X = []
y = []

for year in years:
    dt = df[df.Year == year]
    max_len = len(dt)-k

    x = np.empty((max_len, 24))
    for i in range(max_len):
        dx = dt.iloc[i:i+k][['MN_KG', 'MN_LT', 'MN_DH', 
                             'LM_KG', 'LM_LT', 'LM_DH']].values
        dx = dx.reshape(1, -1)
        x[i] = dx
    X.append(x)
    y.append(dt.iloc[k:].MN_KG.values)

X = np.vstack(X)
y = np.hstack(y)

run(X, y, kfold=None)
# run(X, y, kfold=2)

# Case 4
# x: MN_KG,LT,DH (t-3, t-2, t-1, t) | LM_KG,LT,DH (t-3, t-2, t-1, t) | LM_DH (t+1) => 3 * 4 + 3 * 4 + 1 = 25
# y: MN_KG (t+1)
print("==> Case 4 <==")
k = 4
X = []
y = []

for year in years:
    dt = df[df.Year == year]
    max_len = len(dt)-k

    x = np.empty((max_len, 25))
    for i in range(max_len):
        dx1 = dt.iloc[i:i+k][['MN_KG', 'MN_LT', 'MN_DH', 
                              'LM_KG', 'LM_LT', 'LM_DH']].values
        dx1 = dx1.reshape(1, -1)
        dx2 = dt.iloc[i+k][['LM_DH']].values
        dx2 = dx2.reshape(1, -1)
        x[i] = np.hstack((dx1, dx2))
    X.append(x)
    y.append(dt.iloc[k:].MN_KG.values)

X = np.vstack(X)
y = np.hstack(y)

run(X, y, kfold=None)
# run(X, y, kfold=2)

# Case 5
# x: MN_KG,LT,DH (t-3, t-2, t-1, t) | LM_KG,LT,DH (t-3, t-2, t-1, t) | MN_LT,DH,LM_KG,LT,DH (t+1) => 3 * 4 + 3 * 4 + 5 = 29
# y: MN_KG (t+1)
print("==> Case 5 <==")
k = 4
X = []
y = []

for year in years:
    dt = df[df.Year == year]
    max_len = len(dt)-k

    x = np.empty((max_len, 29))
    for i in range(max_len):
        dx1 = dt.iloc[i:i+k][['MN_KG', 'MN_LT', 'MN_DH', 
                              'LM_KG', 'LM_LT', 'LM_DH']].values
        dx1 = dx1.reshape(1, -1)
        dx2 = dt.iloc[i+k][['MN_LT', 'MN_DH', 'LM_KG', 'LM_LT', 'LM_DH']].values
        dx2 = dx2.reshape(1, -1)
        x[i] = np.hstack((dx1, dx2))
    X.append(x)
    y.append(dt.iloc[k:].MN_KG.values)

X = np.vstack(X)
y = np.hstack(y)

run(X, y, kfold=None)
# run(X, y, kfold=2)