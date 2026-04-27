import pandas as pd

def calculate_clv(df):
    """
    Simple CLV calculation using RFM logic
    """

    clv = df.groupby('CUSTOMERNAME').agg({
        'SALES': ['sum', 'mean', 'count']
    })

    clv.columns = ['TotalRevenue', 'AvgOrderValue', 'PurchaseFrequency']

    # Simple CLV formula
    clv['CLV'] = clv['AvgOrderValue'] * clv['PurchaseFrequency']

    return clv.reset_index()