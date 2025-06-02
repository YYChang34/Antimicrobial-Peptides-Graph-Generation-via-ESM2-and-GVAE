# Antimicrobial Peptides Graph Generation via ESM2 and GVAE

## System Requirements

* **Operating System**: Linux Ubuntu 22.04 or Windows 10+
* **Python Version**: 3.10+
* **Anaconda**: Installed
* **PyTorch Version**: 2.0+
* **CUDA Version**: 11.7 or higher
* **Other Packages**: pandas, numpy, biopython, torch-geometric, einops, matplotlib, transformers, scipy, tqdm, esm

## Directory Structure

```
project_directory/
├── AMP1000.csv
├── Fasta_trans.py
├── ESM_2.py
├── run_ESMfold.py
├── distance.py
├── construct_graph.py
├── GVAE.py
├── output_AMP1000.fasta
├── processed_AMP1000_data.csv
├── loss_plot.png
└── generated_graphs/
```

## How to Run

### 1. Create and Activate Conda Environment

```bash
conda create -n protein_graph python=3.10
conda activate protein_graph
```

### 2. Install Required Packages

```bash
conda install pandas numpy biopython matplotlib scipy tqdm -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric einops transformers esm
```

> **Note**: If using Apple Silicon (M1/M2), follow the official [PyTorch installation guide](https://pytorch.org/) for compatibility.

### 3. Dataset Preparation

Place the raw file `AMP1000.csv` in the project root directory.

* This file contains 100 AMP sequences, with some ID inconsistencies that need to be fixed.

### 4. Data Preprocessing

Execute the following scripts in order to process the data:

```bash
# Step 1: Convert CSV to FASTA and fix ID issues
python Fasta_trans.py

# Step 2: Generate sequence embeddings using ESM2
python ESM_2.py

# Step 3: Predict PDB structure files using ESMFold
python run_ESMfold.py

# Step 4: Extract alpha carbon distances from PDB files
python distance.py

# Step 5: Construct graphs with node features and adjacency
python construct_graph.py
```

The following output files will be generated:

* `output_AMP1000.fasta`: FASTA format sequences with fixed IDs
* `processed_AMP1000_data.csv`: Tabular data with continuous IDs
* `generated_graphs/`: Folder containing graph objects

### 5. Model Training

Run the GVAE model training:

```bash
python GVAE.py
```

* Training loss plot will be saved to `loss_plot.png`
* The GVAE model will learn latent representations of protein graphs

![image](https://github.com/YYChang34/Antimicrobial-Peptides-Graph-Generation-via-ESM2-and-GVAE/blob/main/loss_plot.png)

![image](https://github.com/YYChang34/Antimicrobial-Peptides-Graph-Generation-via-ESM2-and-GVAE/blob/main/generative_graph.png)

### Notes

* ESM2 and ESMFold models are provided by Meta AI's [ESM repository](https://github.com/facebookresearch/esm)
* If ESMFold fails on some sequences, consider reducing batch size or using fewer samples
* Ensure your CUDA and PyTorch versions are compatible to avoid runtime errors

---

This project constructs protein structural graphs from AMP sequences using ESM2 embeddings and spatial structure information, then trains a Graph Variational Autoencoder (GVAE) for graph-level representation learning. Feel free to extend this framework for classification, clustering, or graph generation tasks.
