import torch.nn as nn
import torch

USER_VOCAB_SIZE = 30000
USER_EMBEDDING_SIZE = 256
JOB_VOCAB_SIZE = 30000
JOB_EMBEDDING_SIZE = 256

class UserEncoder(nn.Module):
    """Encodes resume token sequences using Transformers."""
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=3,
        max_seq_len=900,
        dim_feedforward=512,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(USER_VOCAB_SIZE, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection to preference vector
        self.projection = nn.Linear(d_model, 256)

    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len)
        batch_size, seq_len = token_ids.shape
        
        # Create mask for padding tokens (assuming padding_idx=0)
        padding_mask = (token_ids == 0)  # True for padding
        
        # Embed tokens and add positional encoding
        token_emb = self.token_embedding(token_ids)  # (batch_size, seq_len, d_model)
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        pos_emb = self.positional_embedding(positions) 
        x = token_emb + pos_emb
        
        # Transformer processing (mask padding tokens)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Aggregate sequence: Mean pooling (ignore padding)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        seq_lengths = (~padding_mask).sum(dim=1).unsqueeze(-1)
        pooled = x.sum(dim=1) / seq_lengths
        
        # Project to preference vector
        return self.projection(pooled) 

class JobEncoder(nn.Module):
    """Encodes resume token sequences using Transformers."""
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=3,
        max_seq_len=900,
        dim_feedforward=512,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(JOB_VOCAB_SIZE, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection to preference vector
        self.projection = nn.Linear(d_model, 256)

    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len)
        batch_size, seq_len = token_ids.shape
        
        # Create mask for padding tokens (assuming padding_idx=0)
        padding_mask = (token_ids == 0)  # True for padding
        
        # Embed tokens and add positional encoding
        token_emb = self.token_embedding(token_ids)  # (batch_size, seq_len, d_model)
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        pos_emb = self.positional_embedding(positions) 
        x = token_emb + pos_emb
        
        # Transformer processing (mask padding tokens)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Aggregate sequence: Mean pooling (ignore padding)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        seq_lengths = (~padding_mask).sum(dim=1).unsqueeze(-1)
        pooled = x.sum(dim=1) / seq_lengths
        
        # Project to preference vector
        return self.projection(pooled) 

class CollaborativeFiltering(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, user_pref, job_pref):
        combined = torch.cat([user_pref, job_pref], dim=1)
        return self.mlp(combined).squeeze()

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_encoder = UserEncoder()
        self.job_encoder = JobEncoder()
        self.cf = CollaborativeFiltering()
    
    def forward(self, user_vec, job_vec):
        user_pref = self.user_encoder(user_vec)
        job_pref = self.job_encoder(job_vec)
        rating = self.cf(user_pref, job_pref)
        return rating