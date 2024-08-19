import torch
import scipy.io
import numpy as np
from torch_geometric.data import Data
from pretrain_gnns.bio.model import GNN
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch_geometric.transforms as T
from torch.utils.checkpoint import checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_graph_data(filepath):
    data = scipy.io.loadmat(filepath)
    x = torch.tensor(data['attrb'].todense(), dtype=torch.float32)
    edge_index = torch.tensor(np.array(data['network'].nonzero()), dtype=torch.long)
    edge_attr = torch.ones(edge_index.shape[1], 9)
    y = torch.tensor(data['group'].argmax(axis=1).squeeze(), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

print("Loading datasets...")
train_data = load_graph_data('acmv9.mat')
test_data = load_graph_data('citationv1.mat')

transform = T.NormalizeFeatures()
train_data = transform(train_data)
test_data = transform(test_data)

class CPUOffloadGraphSampler:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.num_nodes = data.x.size(0)

    def __iter__(self):
        node_indices = torch.randperm(self.num_nodes)
        for i in range(0, self.num_nodes, self.batch_size):
            batch_indices = node_indices[i:i+self.batch_size]
            edge_mask = (self.data.edge_index[0].unsqueeze(1) == batch_indices.unsqueeze(0)).any(1)
            batch_edge_index = self.data.edge_index[:, edge_mask]
            batch_edge_attr = self.data.edge_attr[edge_mask]
            
            node_map = {int(idx.item()): i for i, idx in enumerate(batch_indices)}
            batch_edge_index = torch.tensor([[node_map.get(int(idx.item()), -1) for idx in batch_edge_index[0]],
                                             [node_map.get(int(idx.item()), -1) for idx in batch_edge_index[1]]], 
                                            dtype=torch.long)
            
            valid_edges = (batch_edge_index[0] != -1) & (batch_edge_index[1] != -1)
            batch_edge_index = batch_edge_index[:, valid_edges]
            batch_edge_attr = batch_edge_attr[valid_edges]
            
            batch_x = self.data.x[batch_indices]
            batch_y = self.data.y[batch_indices]
            
            yield batch_indices, batch_x, batch_edge_index, batch_edge_attr, batch_y

    def __len__(self):
        return (self.num_nodes + self.batch_size - 1) // self.batch_size

class MemoryEfficientGNN(GNN):
    def forward(self, x, edge_index, edge_attr, batch_indices):
        h_list = [x]
        for layer in range(self.num_layer):
            h = checkpoint(self.gnns[layer], h_list[layer], edge_index, edge_attr, use_reentrant=False)
            if layer == self.num_layer - 1:
                h = torch.nn.functional.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = torch.nn.functional.dropout(torch.nn.functional.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = sum(h_list[1:])

        return node_representation[batch_indices]

print("Loading pre-trained model...")
num_node_features = train_data.num_node_features
num_classes = train_data.y.max().item() + 1
model = MemoryEfficientGNN(num_layer=5, emb_dim=300, JK="last", drop_ratio=0.5, gnn_type='gin')
model.load_state_dict(torch.load('pretrain_gnns/bio/model_gin/supervised.pth', map_location=device))
model = model.to(device)

model.classifier = torch.nn.Linear(300, num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()
scaler = GradScaler()

print("Starting training...")
model.train()
num_epochs = 20
batch_size = 32
accumulation_steps = 256

for epoch in range(num_epochs):
    total_loss = 0
    sampler = CPUOffloadGraphSampler(train_data, batch_size)
    optimizer.zero_grad()
    
    for i, (batch_indices, batch_x, batch_edge_index, batch_edge_attr, batch_y) in enumerate(tqdm(sampler, desc=f"Epoch {epoch+1}/{num_epochs}")):
        batch_x = batch_x.to(device)
        batch_edge_index = batch_edge_index.to(device)
        batch_edge_attr = batch_edge_attr.to(device)
        batch_y = batch_y.to(device)
        
        with autocast():
            node_representation = model(batch_x, batch_edge_index, batch_edge_attr, torch.arange(len(batch_indices)))
            output = model.classifier(node_representation)
            loss = criterion(output, batch_y)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * accumulation_steps

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        del batch_x, batch_edge_index, batch_edge_attr, batch_y, node_representation, output
        torch.cuda.empty_cache()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(sampler):.4f}')

print("Evaluating model...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    sampler = CPUOffloadGraphSampler(test_data, batch_size)
    for batch_indices, batch_x, batch_edge_index, batch_edge_attr, batch_y in tqdm(sampler, desc="Evaluation"):
        batch_x = batch_x.to(device)
        batch_edge_index = batch_edge_index.to(device)
        batch_edge_attr = batch_edge_attr.to(device)
        
        with autocast():
            node_representation = model(batch_x, batch_edge_index, batch_edge_attr, torch.arange(len(batch_indices)))
            output = model.classifier(node_representation)
        
        predictions = torch.argmax(output, dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch_y.numpy())

        del batch_x, batch_edge_index, batch_edge_attr, node_representation, output
        torch.cuda.empty_cache()

accuracy = accuracy_score(all_labels, all_preds)
micro_f1 = f1_score(all_labels, all_preds, average='micro')
print(f'Accuracy: {accuracy:.4f}')
print(f'Micro F1 Score: {micro_f1:.4f}')