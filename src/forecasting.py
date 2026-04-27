from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def prepare_sales(df):
    sales = df.groupby(['YEAR','MONTH'])['SALES'].sum().reset_index()
    sales['TimeIndex'] = range(len(sales))
    return sales

def train_model(sales):
    X = sales[['TimeIndex']]
    y = sales['SALES']

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)

    return model, mae

def predict_future(model, last_idx, steps=6):
    return [model.predict([[last_idx+i]])[0] for i in range(1, steps+1)]