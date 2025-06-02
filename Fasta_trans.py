import pandas as pd
# 讀取 CSV 檔
df = pd.read_csv("AMP1000.csv")

# 重新編號
df = df.reset_index(drop=True)
df["id"] = ["Sequence_{}".format(i + 1) for i in range(len(df))]

# 輸出成 FASTA 格式
with open("output_AMP1000.fasta", "w") as f:
    for i, row in df.iterrows():
        f.write(f">{row['id']}\n{row['sequence']}\n")

df.to_csv("processed_AMP1000_data.csv", index=False)

