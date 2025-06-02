import os
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from Bio import SeqIO
from ESM_2 import all_seq_embeddings
import torch
from torch_geometric.loader import DataLoader


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 讀取前處理 CSV
df = pd.read_csv("processed_AMP1000_data.csv")

# 嵌入資料 (每筆是 [L, 320]，共有 1000 筆)
all_embeddings = all_seq_embeddings


# 預設 adjacency matrix 路徑
adj_dir = "distance_AMP1000_matrices"

graph_list = []

for i, row in df.iterrows():
    seq_id = row["id"]
    sequence = row["sequence"]
    label = int(row["activity"])

    emb = all_embeddings[i]  # shape: [L, 320]
    emb = torch.tensor(emb, dtype=torch.float)

    adj_path = os.path.join(adj_dir, f"{seq_id}.npy")
    if not os.path.exists(adj_path):
        print(f"Missing adjacency for {seq_id}")
        continue

    adj = np.load(adj_path)
    edge_index = []
    for i in range(len(sequence)):
        for j in range(len(sequence)):
            if adj[i, j] != 0 and i != j:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=emb, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))

    data.seq_id = seq_id  
    graph_list.append(data)

train_loader = DataLoader(graph_list, batch_size=32, shuffle=True)