import os
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from construct_graph import train_loader  # 你已有的 train_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# === VAE Encoder ===
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin_mu = Linear(hidden_channels, latent_dim)
        self.lin_logvar = Linear(hidden_channels, latent_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        mu = self.lin_mu(x)
        logvar = self.lin_logvar(x)
        return mu, logvar

# === VAE Decoder ===
max_num_nodes = 32  # 你資料最大節點數
node_feature_dim = 320

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_channels, max_num_nodes, node_feature_dim):
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.node_feature_dim = node_feature_dim

        self.lin1 = Linear(latent_dim, hidden_channels)
        self.lin_node = Linear(hidden_channels, max_num_nodes * node_feature_dim)
        self.lin_adj = Linear(hidden_channels, max_num_nodes * max_num_nodes)

    def forward(self, z):
        h = F.relu(self.lin1(z))
        x = self.lin_node(h)
        adj_logits = self.lin_adj(h)
        x = x.view(-1, self.max_num_nodes, self.node_feature_dim)
        adj = adj_logits.view(-1, self.max_num_nodes, self.max_num_nodes)
        adj = torch.sigmoid(adj)
        # 去除自環
        adj = adj * (1 - torch.eye(self.max_num_nodes, device=adj.device).unsqueeze(0))
        return x, adj

# === GraphVAE Model ===
class GraphVAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, num_nodes):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_channels, num_nodes, in_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        mu, logvar = self.encoder(data.x, data.edge_index, data.batch)
        z = self.reparameterize(mu, logvar)
        x_recon, adj_recon = self.decoder(z)
        return x_recon, adj_recon, mu, logvar

num_nodes = max_num_nodes

model = GraphVAE(in_channels=node_feature_dim, hidden_channels=64, latent_dim=16, num_nodes=num_nodes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x_recon, adj_recon, mu, logvar = model(data)

        batch_size = data.num_graphs
        recon_node_loss = 0
        recon_adj_loss = 0

        start = 0
        for i in range(batch_size):
            num_nodes_i = (data.batch == i).sum().item()
            x_true = data.x[start:start + num_nodes_i]  # shape: [Ni, node_feature_dim]
            x_pred = x_recon[i, :num_nodes_i, :]        # shape: [Ni, node_feature_dim]

            # 節點 loss
            recon_node_loss += F.mse_loss(x_pred, x_true)

            # 鄰接矩陣 loss
            adj_true = torch.zeros(num_nodes, num_nodes, device=device)
            mask = data.batch == i
            edge_idx = data.edge_index[:, mask[data.edge_index[0]] & mask[data.edge_index[1]]]
            # 映射到局部索引（0 ~ num_nodes_i-1）
            node_idx = torch.nonzero(mask, as_tuple=False).view(-1)
            node_map = {idx.item(): j for j, idx in enumerate(node_idx)}
            if edge_idx.numel() > 0:
                edge_idx_mapped = torch.tensor([[node_map[e.item()] for e in edge] for edge in edge_idx.t()],
                                               dtype=torch.long, device=device).t()
                adj_true[edge_idx_mapped[0], edge_idx_mapped[1]] = 1.0

            adj_pred = adj_recon[i, :num_nodes_i, :num_nodes_i]
            recon_adj_loss += F.binary_cross_entropy(adj_pred, adj_true[:num_nodes_i, :num_nodes_i])

            start += num_nodes_i

        recon_node_loss /= batch_size
        recon_adj_loss /= batch_size
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_node_loss + recon_adj_loss + kl_loss

        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    print(f'Epoch {epoch}, Loss: {loss_all:.4f}')


# 生成新圖並存檔
model.eval()
os.makedirs('generated_graphs', exist_ok=True)
num_generate = 10

for i in range(num_generate):
    with torch.no_grad():
        z = torch.randn(1, 16).to(device)
        x_gen, adj_gen = model.decoder(z)

        adj_bin = (adj_gen > 0.5).float().squeeze(0)
        edge_index = dense_to_sparse(adj_bin)[0]
        generated_graph = Data(x=x_gen.squeeze(0), edge_index=edge_index)

        save_path = f'generated_graphs/graph_{i}.pt'
        torch.save(generated_graph, save_path)
        print(f'Saved generated graph {i} to {save_path}')
