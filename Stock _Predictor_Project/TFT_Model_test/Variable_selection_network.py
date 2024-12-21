class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, n_vars, hidden_dim, dropout=0.1, context_dim=None):
        super().__init__()
        # Flattened inputs to produce weights for each variable
        self.flattened_inputs = GRN(n_vars * input_dim, hidden_dim, output_dim=n_vars, dropout=dropout, context_dim=context_dim)

        # GRN for each variable
        self.transformed_inputs = nn.ModuleList([
            GRN(input_dim, hidden_dim, hidden_dim, dropout=dropout, context_dim=context_dim)
            for _ in range(n_vars)
        ])

        self.softmax = nn.Softmax(dim=-1)
        self.n_vars = n_vars
        self.input_dim = input_dim

    def forward(self, embedding, context=None):
        # embedding: [batch, time, n_vars*input_dim] if temporal
        # Get sparse weights for variable selection
        sparse_weights = self.flattened_inputs(embedding, context=context) # [batch, time, n_vars]
        sparse_weights = self.softmax(sparse_weights) # [batch, time, n_vars]

        batch_size = embedding.size(0)
        seq_len = embedding.size(1)
        embedding_reshaped = embedding.view(batch_size, seq_len, self.n_vars, self.input_dim)

        transformed = []
        for i, grn in enumerate(self.transformed_inputs):
            var_emb = embedding_reshaped[:, :, i, :] # [batch, time, input_dim]
            transformed.append(grn(var_emb, context=context)) # [batch, time, hidden_dim]

        transformed = torch.stack(transformed, dim=-1) # [batch, time, hidden_dim, n_vars]

        # Weighted combination of variables
        sparse_weights = sparse_weights.unsqueeze(-2) # [batch, time, 1, n_vars]
        combined = (transformed * sparse_weights).sum(dim=-1) # [batch, time, hidden_dim]

        return combined, sparse_weights
