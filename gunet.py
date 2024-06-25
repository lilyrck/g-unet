import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

class gPool(nn.Module):
    def __init__(self, in_features, k):
        """
        Initialize the gPool layer.
        
        Args:
        - in_features: The dimensionality of input features
        - k: The numbe of nodes to select for the pooling
        """
        super(gPool, self).__init__()
        self.proj = nn.Linear(in_features, 1)  # Projection to 1D, helps in computing scalar projections of node features
        self.sigmoid = nn.Sigmoid()
        self.k = k # Number of nodes to select for the pooled graph

    def forward(self, A, X):
        """
        Forward pass of the gPool layer.

        Args:
        - A: adjacency matrix (N x N)
        - X: node feature matrix (N x C)

        Returns:
        - A_pooled: pooled adjacency matrix (k x k)
        - X_pooled: pooled node feature matrix (k x C)
        - indices: indices of selected nodes in the original graph (k)
        """
        # Compute the second power of the adjacency matrix
        A2 = torch.matmul(A, A)

        # Project node features to 1D using trainable projection vector p
        projections = self.proj(X)  # (N x 1)

        # Perform k-max pooling to select top-k nodes based on projections
        _, indices = torch.topk(projections.view(-1), self.k, largest=True)
        indices = indices.unique()  # Ensure unique indices

        # Construct new feature matrix X'
        X_pooled = X[indices]  # (k x C)

        # Construct new adjacency matrix A' using the second power adjacency matrix
        A_pooled = A2[indices][:, indices]  # (k x k)

        # Calculate selection mask y'
        y = self.sigmoid(projections)  # (N x 1)
        y_pooled = y[indices]  # (k x 1)

        # Scale pooled features by selection mask
        X_pooled = X_pooled * y_pooled.view(-1, 1)  # (k x C)

        return A_pooled, X_pooled, indices
    

# # Test for the gPool
# # Dummy input node features and adjacency matrix
# num_nodes = 10
# in_features = 5
# k = 3  # Number of nodes to select

# A = torch.abs(torch.randn(num_nodes, num_nodes))  # Dummy adjacency matrix
# X = torch.randn(num_nodes, in_features)  # Dummy node features

# # Instantiate gPool layer
# gpool_layer = gPool(in_features=in_features, k=k)

# # Perform pooling operation
# A_pooled, X_pooled, indices = gpool_layer(A, X)

# # Print results
# print("Original A shape:", A)
# print("Pooled A shape:", A_pooled)
# print("Original X shape:", X)
# print("Pooled X shape:", X_pooled.shape)
# print("Indices of selected nodes:", indices)
    

class attnPool(nn.Module):
    def __init__(self, in_features, k):
        """
        Initialize the attnPool layer.
        
        Args:
        - in_features: The dimensionality of input features
        - k: The numbe of nodes to select for the pooling
        """
        super(attnPool, self).__init__()
        self.W_k = nn.Linear(in_features, in_features) # Linear layer for key matrix
        self.W_v = nn.Linear(in_features, 1) # Linear layer for value matrix
        self.k = k # nb of nodes to selected for the pooling

    def forward(self, A, X):
        """
        Forward pass of the attnPool layer.

        Args:
        - A: adjacency matrix (N x N)
        - X: node feature matrix (N x C)

        Returns:
        - A_pooled: pooled adjacency matrix (k x k)
        - X_pooled: pooled node feature matrix (k x C)
        - indices: indices of selected nodes in the original graph (k)
        """
        # Compute key matrix K and value matrix V
        K = self.W_k(X) # (N x C)
        V = self.W_v(X).squeeze(1) # (N,)

        # Compute attention coefficients E
        E = torch.matmul(X, K.transpose(0, 1))  # (N x N)

        # Normalize attention coefficients using softmax
        E_normalized = F.softmax(E, dim=1)  # (N x N)

        # Compute ranking scores y
        y = torch.matmul(E_normalized, V)  # (N,)

         # Select top-k nodes based on ranking scores
        _, indices = torch.topk(y, self.k, largest=True)
        indices = indices.unique()  # Ensure unique indices

        # Construct new feature matrix X'
        X_pooled = X[indices]  # (k x C)

        # Construct new adjacency matrix A' based on selected nodes
        A_pooled = A[indices][:, indices]  # (k x k)

        return A_pooled, X_pooled, indices


# # Example usage and testing of attnPool layer

# # Dummy input node features and adjacency matrix
# num_nodes = 10
# in_features = 5
# k = 3  # Number of nodes to select

# A = torch.abs(torch.randn(num_nodes, num_nodes))  # Dummy adjacency matrix
# X = torch.randn(num_nodes, in_features)  # Dummy node features

# # Instantiate attnPool layer
# attn_pool_layer = attnPool(in_features=in_features, k=k)

# # Perform attention-based pooling operation
# A_pooled, X_pooled, indices = attn_pool_layer(A, X)

# # Print results
# print("Original A shape:", A.shape)
# print("Pooled A shape:", A_pooled.shape)
# print("Original X shape:", X.shape)
# print("Pooled X shape:", X_pooled.shape)
# print("Indices of selected nodes:", indices)

class gUnpool(nn.Module):
    def __init__(self, in_features):
        """
        Initialize the gUnpool layer.
        
        Args:
        - in_features: The dimensionality of input features
        """
        super(gUnpool, self).__init__()
        self.in_features = in_features

    def forward(self, A, X, indices):
        """
        Forward pass of the gUnpool layer.

        Args:
        - A: adjacency matrix of the previous level (k x k)
        - X: pooled node feature matrix (k x C)
        - indices: indices of selected nodes in the original graph (tensor of shape (k,))

        Returns:
        - X_unpooled: unpooled node feature matrix (N x C)
        - A: adjacency matrix of the previous level (k x k)
        """

        # Initialize new feature matrix with zeros
        X_unpooled = X.new_zeros([A.shape[0], X.shape[1]])

        # Distribute pooled features back to the original nodes
        X_unpooled[indices] = X

        return X_unpooled, A
    

# # Example usage and testing of gUnpool

# A_test = torch.abs(torch.randn(7, 7))
# print(A_test.shape)
# X_test = torch.randn(4,3)
# print(X_test.shape)

# indices = torch.tensor([1, 2, 4, 6])  # Indices of selected nodes in the original graph

# # Initialize gUnpool layer
# unpool_layer = gUnpool(in_features=X_test.shape[1])  # in_features is the number of features in X_pooled

# # Perform graph unpooling
# X_unpooled = unpool_layer(A_test, X_test, indices)

# # Print results
# print("Shape of X_pooled:", X_test.shape)
# print("Pooled node features:")
# print(X_test)
# print("\nIndices of selected nodes:", indices)
# print("\nShape of X_unpooled after unpooling:", X_unpooled.shape)
# print("Unpooled node features:")
# print(X_unpooled)

# The GCN layer, here A stays the same but the dimentionality of X is increased or decreased 
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, act):
        """
        Initialize the GCN layer.
        
        Args:
        - input_dim: The dimensionality of input features
        - output_dim: The dimensionality of output features
        - act: The activation function
        
        """
        super(GCN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.act = act

    def forward(self, X, A):
        """
        Perform the forward pass of the GCN layer.
        
        Args:
        - X: Node features of shape (N, input_dim)
        - A: Adjacency matrix of shape (N, N)
        
        Returns:
        - out: Output node features of shape (N, output_dim)
        
        """
        # A' = A + 2I (adding 2 times the identity matrix for stronger self-loops)
        A = A + 2 * torch.eye(A.size(0))  # Adding self-loops to the adjacency matrix

        # Compute degree matrix D'
        D = torch.diag(torch.sum(A, dim=1))  # Degree matrix
        D_inv_sqrt = torch.sqrt(torch.inverse(D))  # D'^-1/2

        # Normalize adjacency matrix: A' = D'^-1/2 * A' * D'^-1/2
        A_normalized = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)

        # Perform graph convolution: X_l+1 = sigma((D'^-1/2) * A' * (D'^-1/2) * X_l * W_l)
        AX = torch.matmul(A_normalized, X)  # A' * X
        out = self.linear(AX)  # Linear transformation

        # Apply sigma
        out = self.act(out)

        return out

# # Test for the GCN 
# # Dummy input node features and adjacency matrix
# num_nodes = 4
# in_features = 2
# out_features = 3

# X = torch.randn(num_nodes, in_features)  # Dummy node features
# A = torch.abs(torch.randn(num_nodes, num_nodes))  # Dummy adjacency matrix

# # Instantiate GCN layer
# gcn_layer = GCN(input_dim=in_features, output_dim=out_features, act=nn.ReLU())

# # Perform graph convolution
# output = gcn_layer(X, A)

# # Print results
# print("Original X shape:", X.shape)
# print("Original X tensor:")
# print(X)
# print("Output shape after GCN:", output.shape)
# print("Output tensor:")
# print(output)
    

class GraphUnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim, act):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.down_gcn1 = GCN(in_dim, 3, act)
        self.poll1 = gPool(3, ks[0])
        self.down_gcn2 = GCN(3, 3, act)
        self.poll2 = gPool(3, ks[1])
        self.bottom_gcn = GCN(3, 3, act)  # Bottom GCN layer
        self.unpool1 = gUnpool(3)
        self.up_gcn1 = GCN(3, 3, act)
        self.unpool2 = gUnpool(3)
        self.up_gcn2 = GCN(out_dim, out_dim, act)
        self.depth = len(ks)

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []

        # First down
        h = self.down_gcn1(h, g)
        adj_ms.append(g)
        down_outs.append(h)
        g, h, idx = self.poll1(g, h)
        indices_list.append(idx)

        # Second down
        
        h = self.down_gcn2(h, g)
        adj_ms.append(g)
        down_outs.append(h)
        g, h, idx = self.poll2(g, h)
        indices_list.append(idx)

        h = self.bottom_gcn(h, g)  # Bottom GCN layer

        # First Up
        up_idx = self.depth - 1
        g, idx = adj_ms[up_idx], indices_list[up_idx]
        h, g = self.unpool1(g, h, idx)
        h = self.up_gcn1(h, g)
        h = h.add(down_outs[up_idx])  # Skip connection

        # Second up
        up_idx = self.depth - 2
        g, idx = adj_ms[up_idx], indices_list[up_idx]
        h, g = self.unpool2(g, h, idx)
        h = self.up_gcn2(h, g)
        h = h.add(down_outs[up_idx])  # Skip connection


        return h, g

# Define a dummy dataset (you need to replace this with your actual dataset)
class DummyGraphDataset(Dataset):
    def __init__(self, num_graphs, num_nodes, in_dim):
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.data = []
        for _ in range(num_graphs):
            A = torch.abs(torch.randn(num_nodes, num_nodes))  # Random adjacency matrix
            X = torch.randn(num_nodes, in_dim)  # Random node features
            self.data.append((A, X))

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        return self.data[idx]

# Example usage
k_pool = [4, 2]  # The number of nodes we have to select at each pooling
in_dim = 2  # Input dimensionality of features
out_dim = 3  # Output dimensionality of features
dim = 7  # Dimensionality for GCN layers
act = nn.ReLU()  # Activation function


# Initialize GraphUnet
model = GraphUnet(k_pool, in_dim, out_dim, dim, act)

# Example inputs
g = torch.abs(torch.rand(dim, dim))  # Example adjacency matrix
h = torch.rand(dim, in_dim)  # Example node features
print(f"X input shape:", h.shape)
print(f"X input:", h)
print(f"A input shape:", g.shape)
print(f"A input:", g)

# Forward pass
h_new, g_new = model(g, h)

# Print shapes of outputs
print(f"X output shape:", h_new.shape)
print(f"X output:", h_new)
print(f"A output shape:", g_new.shape)
print(f"A output:", g_new)

