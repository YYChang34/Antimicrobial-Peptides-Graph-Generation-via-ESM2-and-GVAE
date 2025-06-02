import torch
import esm
from Bio import SeqIO  # 用來讀 fasta 檔案

# 讀取 fasta 並轉成 List[Tuple[str, str]]
fasta_path = "output_AMP1000.fasta"
data = [(record.id, str(record.seq)) for record in SeqIO.parse(fasta_path, "fasta")]

# 載入模型
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

# 取得 batch_converter
batch_converter = alphabet.get_batch_converter()
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# 取得 embeddingD
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[6], return_contacts=False)

token_embeddings = results["representations"][6] 

padding_mask = (batch_tokens != alphabet.padding_idx)
all_seq_embeddings = []

for i in range(len(data)):
    seq_len = padding_mask[i].sum().item()
    embedding = token_embeddings[i, 1:seq_len-1, :]  
    all_seq_embeddings.append(embedding)

