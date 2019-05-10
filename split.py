import pandas as pd
import glob, os
df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "data/train*.csv"))))
df = df.dropna()

df["Text"] = df['Text'].map(lambda x: x.encode('').decode('utf-8'))
df
s = df.shape[0]
train_index = int(s * 0.75)
dev_index = int(s * 0.8)
df

with open("train.csv", "w", encoding="utf-8") as f:
    df.iloc[0:train_index].to_csv(f, encoding="utf-8")

with open("dev.csv", "w", encoding="utf-8") as f:
    df.iloc[train_index:dev_index].to_csv(f)

with open("test.csv", "w", encoding="utf-8") as f:
    df.iloc[dev_index:].to_csv(f)
