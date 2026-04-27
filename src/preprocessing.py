import pandas as pd

def load_data(path):
    df = pd.read_excel(path)
    return df


def clean_data(df):
    # Drop only important missing values
    df = df.dropna(subset=['InvoiceDate', 'CustomerID'])

    # Rename columns
    df = df.rename(columns={
        'InvoiceDate': 'ORDERDATE',
        'InvoiceNo': 'ORDERNUMBER',
        'CustomerID': 'CUSTOMERNAME',
        'Quantity': 'QUANTITYORDERED',
        'Country': 'COUNTRY',
        'UnitPrice': 'PRICE',
        'Description': 'PRODUCTLINE'
    })

    # Remove invalid values
    df = df[df['QUANTITYORDERED'] > 0]
    df = df[df['PRICE'] > 0]

    # Convert types
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df['CUSTOMERNAME'] = df['CUSTOMERNAME'].astype(str)

    # Create SALES column
    df['SALES'] = df['QUANTITYORDERED'] * df['PRICE']

    # Time features
    df['YEAR'] = df['ORDERDATE'].dt.year
    df['MONTH'] = df['ORDERDATE'].dt.month

    return df