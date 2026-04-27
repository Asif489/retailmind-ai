from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

def create_basket(df):
    basket = df.groupby(['ORDERNUMBER', 'PRODUCTLINE'])['QUANTITYORDERED'] \
        .sum().unstack().fillna(0)

    basket = (basket > 0).astype(int)
    return basket

def generate_rules(basket):
    freq = apriori(basket, min_support=0.05, use_colnames=True)

    if freq.empty:
        return pd.DataFrame()

    rules = association_rules(freq, metric='lift', min_threshold=1)

    if rules.empty:
        return pd.DataFrame()

    rules = rules[['antecedents','consequents','support','confidence','lift']]

    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    return rules.sort_values(by='lift', ascending=False)