import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


@st.experimental_memo
def load_data(columns):
    df = pd.read_excel('Dataset_floodseason_76_20_NEW_3.11.xlsx',
                       skiprows=range(0, 2), header=None)
    df.columns = columns
    return df


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# App title
st.title('This is a title')

# Sidebar
columns = ['Year', 'Month', 'Day', 'Mực nước KG', 'Mực nước LT',
           'Mực nước DH', 'Lượng mưa KG', 'Lượng mưa LT', 'Lượng mưa DH']

# => Target selection <=
target_name = st.sidebar.selectbox('Target List', columns[3:])

# => Feature selection <=
feature_options = [column for column in columns[3:]
                   if column != target_name]
feature_names = st.sidebar.multiselect(
    'Feature List', feature_options)

# => Model selection <=
model_options = ['LinearRegression', 'RandomForestRegressor']
model_names = st.sidebar.multiselect(
    'Model List', model_options, model_options)

# => Time selection <=
mapping_time = {
    'Mực nước KG': 'mean',
    'Mực nước LT': 'mean',
    'Mực nước DH': 'mean',
    'Lượng mưa KG': 'sum',
    'Lượng mưa LT': 'sum',
    'Lượng mưa DH': 'sum'
}
time_option = st.sidebar.select_slider('Time', columns[:3][::-1])
if time_option == 'Day':
    time_columns = ['Year', 'Month', 'Day']
elif time_option == 'Month':
    time_columns = ['Year', 'Month']
else:
    time_columns = ['Year']

# => Model selection <=
graph_options = ['Density', 'Line']
graph_name = st.sidebar.selectbox('Graph List', graph_options)

# Load dataset
df = load_data(columns)
with st.expander('Dataset', expanded=False):
    st.dataframe(df)

# Main
clicked = st.button('Run')
if clicked:
    if not feature_names:
        st.error('Please select feature before 😢')
        st.stop()

    if not model_names:
        st.error('Please select model before 😢')
        st.stop()

    # => Training
    X = df[feature_names].values
    y = df[target_name].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    visualize_columns = [target_name]

    if model_options[0] in model_names:
        st.subheader('Linear Regression')
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        df['LR_predicted'] = model.predict(X)

        LR_MAE = mean_absolute_error(y_test, y_pred)
        LR_MSE = mean_squared_error(y_test, y_pred)
        LR_RMSE = root_mean_squared_error(y_test, y_pred)
        col1, col2, col3 = st.columns(3)
        col1.metric('MAE', f'{LR_MAE:6f}')
        col2.metric('MSE', f'{LR_MSE:6f}')
        col3.metric('RMSE', f'{LR_RMSE:6f}')

        mapping_time['LR_predicted'] = 'mean'
        visualize_columns.append('LR_predicted')

    if model_options[1] in model_names:
        st.subheader('Random Forest Regression')
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        df['RFR_predicted'] = model.predict(X)

        RFR_MAE = mean_absolute_error(y_test, y_pred)
        RFR_MSE = mean_squared_error(y_test, y_pred)
        RFR_RMSE = root_mean_squared_error(y_test, y_pred)
        col1, col2, col3 = st.columns(3)
        col1.metric('MAE', f'{RFR_MAE:6f}')
        col2.metric('MSE', f'{RFR_MSE:6f}')
        col3.metric('RMSE', f'{RFR_RMSE:6f}')

        mapping_time['RFR_predicted'] = 'mean'
        visualize_columns.append('RFR_predicted')

    # => Visualization
    df = df.groupby(time_columns, as_index=False).agg(mapping_time)
    if graph_name == graph_options[0]:
        fig = sns.kdeplot(df[visualize_columns], fill=True).get_figure()
        fig.savefig(f'density_{time_option}.png', dpi=300)
        st.pyplot(fig, True)
    else:
        fig = sns.lineplot(df[visualize_columns],
                           palette="tab10", linewidth=2).get_figure()
        fig.savefig(f'lineplot_{time_option}.png', dpi=300)
        st.pyplot(fig, True)
