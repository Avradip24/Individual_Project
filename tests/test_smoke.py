import pandas as pd
df = pd.read_csv('data/processed/train.csv')
for cluster_id in df['merchant_cluster'].value_counts().head(15).index:
    sample = df[df['merchant_cluster'] == cluster_id]['transaction_description'].unique()[
        :8]
    print(
        f'\n=== Cluster {cluster_id} ({len(df[df['merchant_cluster'] == cluster_id])} rows) ===')
    for s in sample:
        print(' ', s)
