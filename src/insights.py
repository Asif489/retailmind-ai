def generate_insights(df, rfm, sales, churn_df=None):
    insights = []

    # Revenue insight
    total_revenue = df['SALES'].sum()
    insights.append(f"Total revenue generated is {total_revenue:,.0f}.")

    # Best customer
    top_customer = df.groupby('CUSTOMERNAME')['SALES'].sum().idxmax()
    insights.append(f"Top performing customer is {top_customer}.")

    # Best product
    top_product = df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
    insights.append(f"Best selling product category is {top_product}.")

    # Sales trend
    if sales['SALES'].iloc[-1] > sales['SALES'].iloc[0]:
        insights.append("Sales show an increasing trend over time.")
    else:
        insights.append("Sales show a declining trend over time.")

    # Churn insight
    if churn_df is not None:
        high_risk = (churn_df['Churn Probability'] > 0.7).sum()
        insights.append(f"There are {high_risk} high-risk churn customers.")

    return insights