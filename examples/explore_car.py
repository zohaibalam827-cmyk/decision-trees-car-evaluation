import pandas as pd

cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv("data/car.data", names=cols)

print(df.shape)
print(df.head())
print(df.dtypes)
print("missing values:", df.isnull().sum().sum())
print(df["class"].value_counts())
