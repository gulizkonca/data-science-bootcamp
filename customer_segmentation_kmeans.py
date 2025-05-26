Python 3.13.2 (v3.13.2:4f8bb3947cf, Feb  4 2025, 11:51:10) [Clang 15.0.0 (clang-1500.3.9.4)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> #!/usr/bin/env python
... # pip install pandas scikit-learn matplotlib seaborn
... import pandas as pd
... from sklearn.preprocessing import StandardScaler
... from sklearn.cluster import KMeans
... from sklearn.metrics import silhouette_score
... import matplotlib.pyplot as plt
... import seaborn as sns
... 
... DATA_PATH = "ecommerce_customers.csv"
... 
... def main():
...     df = pd.read_csv(DATA_PATH)
...     num_df = df.select_dtypes(include="number")
...     X = StandardScaler().fit_transform(num_df)
... 
...     k = 3
...     km = KMeans(n_clusters=k, random_state=42, n_init=10)
...     labels = km.fit_predict(X)
...     sil = silhouette_score(X, labels)
...     print(f"Silhouette Score (k={k}):", round(sil, 3))
... 
...     df["Cluster"] = labels
...     sns.pairplot(df, hue="Cluster", vars=num_df.columns[:4])
...     plt.tight_layout(); plt.savefig("clusters.png")
...     print("Plot saved → clusters.png")
...     df.to_csv("clustered_customers.csv", index=False)
... 
... if __name__ == "__main__":
...     main()
