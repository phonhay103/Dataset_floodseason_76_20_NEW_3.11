import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


columns = ['Year', 'Month', 'Day', 'Mực nước KG', 'Mực nước LT',
           'Mực nước DH', 'Lượng mưa KG', 'Lượng mưa LT', 'Lượng mưa DH']
df = pd.read_excel('Dataset_floodseason_76_20_NEW_3.11.xlsx',
                   skiprows=range(0, 2), header=None)
df.columns = columns


def run(df=df, feature_columns=columns[4:9], label_column=columns[3], kfold=None, show=False):
    X = df[feature_columns].values
    y = df[label_column].values
    LR_model = LinearRegression()
    RFR_model = RandomForestRegressor()

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

        if show:
            df = df.copy()
            df['LR_predicted'] = LR_model.predict(X).reshape(-1, 1)
            df['RFR_predicted'] = RFR_model.predict(X).reshape(-1, 1)

            fig = sns.kdeplot(
                df[[columns[3], 'LR_predicted', 'RFR_predicted']], fill=True).get_figure()
            fig.savefig('density.png', dpi=300)
            plt.clf()

            mapping = {
                'Mực nước KG': 'mean',
                'Mực nước LT': 'mean',
                'Mực nước DH': 'mean',
                'Lượng mưa KG': 'sum',
                'Lượng mưa LT': 'sum',
                'Lượng mưa DH': 'sum',
                'LR_predicted': 'mean',
                'RFR_predicted': 'mean'
            }
            df = df.groupby(['Year']).agg(mapping)

            fig = sns.lineplot(df[[columns[3], 'LR_predicted', 'RFR_predicted']],
                               palette="tab10", linewidth=2).get_figure()
            fig.savefig('lineplot.png', dpi=300)
            plt.clf()


# run(df, kfold=10)
run(df, show=True)
