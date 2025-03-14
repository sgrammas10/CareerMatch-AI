import torch.nn as nn
import torch
import sentencepiece as spm

# Train initial model, uncomment if m.model and m.vocab get deleted
# spm.SentencePieceTrainer.train('--input=initText.txt --model_prefix=m --vocab_size=5000')

# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('m.model')
# #encoding = sp.EncodeAsIds("The quick sly fox jumped over the lazy rabbit")
# #print(encoding)

USER_VOCAB_SIZE = 5000
USER_EMBEDDING_SIZE = 256
JOB_VOCAB_SIZE = 5000
JOB_EMBEDDING_SIZE = 256
USE_MLP = True

class UserEncoder(nn.Module):
    """Encodes resume token sequences using Transformers."""
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=3,
        max_seq_len=512,
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
        max_seq_len=512,
        dim_feedforward=512,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(JOB_EMBEDDING_SIZE, d_model)
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

# class CollaborativeFiltering(nn.Module):
#     def __init__(self, use_mlp=False):
#         super().__init__()
#         self.use_mlp = use_mlp
#         if self.use_mlp:
#             self.mlp = nn.Sequential(
#                 nn.Linear(256 * 2, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 1)
#             )
    
#     def forward(self, user_pref, job_pref):
#         if self.use_mlp:
#             combined = torch.cat([user_pref, job_pref], dim=1)
#             return self.mlp(combined).squeeze()
#         else:
#             # Dot product similarity
#             return (user_pref * job_pref).sum(dim=1)

# class FullModel(nn.Module):
#     def __init__(self, use_mlp=False):
#         super().__init__()
#         self.user_encoder = UserEncoder()
#         self.job_encoder = JobEncoder()
#         self.cf = CollaborativeFiltering(use_mlp=use_mlp)
    
#     def forward(self, user_vec, job_vec):
#         user_pref = self.user_encoder(user_vec)
#         job_pref = self.job_encoder(job_vec)
#         rating = self.cf(user_pref, job_pref)
#         return rating

# model = FullModel(use_mlp=USE_MLP)
# criterion = nn.MSELoss()  # For regression tasks
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Example training loop
# def train(model, dataloader, epochs):
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for user_vec, job_vec, target_rating in dataloader:
#             optimizer.zero_grad()
#             pred_rating = model(user_vec, job_vec)
#             loss = criterion(pred_rating, target_rating)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# user_vec_test = torch.randn(1, USER_EMBEDDING_SIZE)  # Simulated resume vector
# job_vec_test = torch.randn(1, JOB_EMBEDDING_SIZE)    # Simulated job vector
# model.eval()
# with torch.no_grad():
#     rating = model(user_vec_test, job_vec_test)
# print(f"Predicted alignment rating: {rating.item()}")

# use dataset class as a wrapper for inputted data and then use DataLoader function to train model

# input the text, covert to numbers using sentencepiece, generate a 1x100 encoding and pair with some job postings they are interested with

# generate encodings for the text using Transformers, train based on the distance to the existing encodings provided as similar

# Figure out how to make initial encodings

# Then implement collaborative filtering based on these models

# still implement the random epsilon technique, but the ranking will be based on the distance from the current encoding


#input: resume
#output: preference matrix

