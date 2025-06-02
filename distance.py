import os
import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial.distance import pdist, squareform

# 設定參數
folder_path = 'esmfold_AMP1000_outputs'
threshold = 10.0  # Å
output_folder = 'distance_AMP1000_matrices'

# 建立輸出資料夾
os.makedirs(output_folder, exist_ok=True)

# 初始化 PDB parser
parser = PDBParser(QUIET=True)

def extract_ca_coordinates(pdb_file):
    structure = parser.get_structure('protein', pdb_file)
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].get_coord())
    return np.array(ca_coords)

# 遍歷資料夾中的所有 PDB 檔案
for filename in os.listdir(folder_path):
    if filename.endswith('.pdb'):
        pdb_path = os.path.join(folder_path, filename)
        try:
            ca_coords = extract_ca_coordinates(pdb_path)
            
            # 計算距離矩陣並轉為二值化矩陣
            dist_matrix = squareform(pdist(ca_coords))  # L x L
            binary_matrix = (dist_matrix < threshold).astype(int)

            # 儲存成 .npy 檔案
            out_filename = os.path.splitext(filename)[0] + '.npy'
            np.save(os.path.join(output_folder, out_filename), binary_matrix)

            print(f"處理完成：{filename}")
        except Exception as e:
            print(f"處理 {filename} 發生錯誤：{e}")