import os
import torch
from Bio import SeqIO
from transformers import EsmForProteinFolding

# 參數
fasta_file = "output_AMP1000.fasta"
output_dir = "esmfold_AMP1000_outputs"
os.makedirs(output_dir, exist_ok=True)

# 載入模型
model = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1", low_cpu_mem_usage=True
).to("cuda")

# 批次預測
for record in SeqIO.parse(fasta_file, "fasta"):
    seq_id   = record.id
    sequence = str(record.seq)
    print(f"🔄 處理：{seq_id}")

    # 直接得到 PDB 字串
    with torch.no_grad():
        pdb_str = model.infer_pdb(sequence)

    # 寫檔
    out_path = os.path.join(output_dir, f"{seq_id}.pdb")
    with open(out_path, "w") as f:
        f.write(pdb_str)

    print(f"✅ 已儲存：{out_path}")
