import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class CustomTGN(nn.Module):
    def __init__(self, node_features: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_dim))
        self.time_encoder = nn.Linear(time_dim, memory_dim)
        self.node_encoder = nn.Linear(node_features, memory_dim)
        self.update_gate = nn.Linear(memory_dim * 2, memory_dim)
        self.reset_gate = nn.Linear(memory_dim * 2, memory_dim)

    def forward(self, nodes: torch.Tensor, memory: torch.Tensor, time: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        time_enc = torch.tanh(self.time_encoder(time))
        node_enc = torch.tanh(self.node_encoder(nodes))
        combined = torch.cat([memory, node_enc], dim=-1)
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        candidate = torch.tanh(node_enc + reset * memory)
        new_memory = (1 - update) * memory + update * candidate
        return new_memory

class UltimateJAIDEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = 32000
        self.hds_dim = config.get('native_config', 'hds_dim', 9216)
        self.node_features = config.get('native_config', 'tgn_node_features', 2048)
        self.memory_dim = config.get('native_config', 'tgn_memory_dim', 2048)
        self.time_dim = config.get('native_config', 'tgn_time_dim', 128)
        self.embed = nn.Embedding(self.vocab_size, self.hds_dim)
        self.tgn = CustomTGN(self.node_features, self.memory_dim, self.time_dim)
        self.hds = nn.Linear(self.hds_dim, self.hds_dim)
        self.graph_conv = GCNConv(self.hds_dim, self.hds_dim)
        self.quantum = nn.Linear(self.hds_dim, self.hds_dim)
        self.output_head = nn.Linear(self.hds_dim, self.vocab_size)
        self.memory = nn.Parameter(torch.randn(self.memory_dim))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.get('optimizer_config', 'lr', 0.00014625))

    def forward(self, tokens: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        x = self.embed(tokens)
        nodes = x.mean(dim=1).unsqueeze(1).repeat(1, self.node_features, 1)
        time = torch.rand(1, self.time_dim)
        tgn_out = self.tgn(nodes, self.memory, time, edge_index)
        hds_out = torch.relu(self.hds(tgn_out))
        if edge_index is not None:
            hds_out = self.graph_conv(hds_out, edge_index)
        quantum_out = self.quantum(hds_out)
        logits = self.output_head(quantum_out)
        return logits
