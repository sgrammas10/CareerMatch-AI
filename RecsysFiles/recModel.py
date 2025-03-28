import torch.nn as nn
import torch

USER_VOCAB_SIZE = 5000
USER_EMBEDDING_SIZE = 256
JOB_VOCAB_SIZE = 5000
JOB_EMBEDDING_SIZE = 256

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

# model = FullModel()

# def load_user_encoder(weight_path, device="cpu"):
#     encoder = UserEncoder().to(device)
#     encoder.load_state_dict(torch.load(weight_path, map_location=device))
#     encoder.eval()
#     return encoder

# def load_job_encoder(weight_path, device="cpu"):
#     encoder = JobEncoder().to(device)
#     encoder.load_state_dict(torch.load(weight_path, map_location=device))
#     encoder.eval()
#     return encoder

# def load_cf_model(weight_path, device="cpu"):
#     cf = CollaborativeFiltering().to(device)
#     cf.load_state_dict(torch.load(weight_path, map_location=device))
#     cf.eval()
#     return cf

# # Initialize components
# user_encoder = load_user_encoder(
#     weight_path="models/user_encoder_final.pth",
#     device="cuda"
# )

# job_encoder = load_job_encoder(
#     weight_path="models/job_encoder_final.pth",
#     device="cuda"
# )

# cf_model = load_cf_model(
#     weight_path="models/cf_final.pth",
#     device="cuda"
# )

# # Inference pipeline (one by one)
# def predict_rating(job_tokens, resume_tokens):
#     with torch.no_grad():
#         job_pref = job_encoder(job_tokens)
#         user_pref = user_encoder(resume_tokens)
#         return cf_model(user_pref, job_pref)

# # Example usage with new data
# def process_new_input(text, max_seq_len=512):
#     sp = spm.SentencePieceProcessor()
#     sp.load('m.model')
#     tokens = sp.EncodeAsIds(text)
#     tokens += [0] * (max_seq_len - len(tokens))
#     return torch.tensor(tokens).unsqueeze(0).to("cuda")

# # Process new job and resume
# new_job = process_new_input("Senior Python Developer")
# new_resume = process_new_input("5 years Python experience...")

# # Get rating
# rating = predict_rating(new_job, new_resume)
# print(f"Predicted alignment score: {rating.item():.4f}")

# def batch_process_texts(texts, device='cpu', max_seq_len=512):
#     """Process a list of texts into batched tokens"""
#     sp = spm.SentencePieceProcessor()
#     sp.Load('m.model')
#     # tokens = sp.EncodeAsIds(text)
#     batch_tokens = []
#     for text in texts:
#         tokens = sp.EncodeAsIds(text)
#         tokens += [sp.pad_id()] * (max_seq_len - len(tokens))
#         batch_tokens.append(tokens)
#     return torch.tensor(batch_tokens, dtype=torch.long, device=device)

# def batch_predict(job_texts, resume_texts, models, device='cpu'):
#     """
#     Args:
#         job_texts: List[str] - Batch of job descriptions
#         resume_texts: List[str] - Batch of resumes
#         models: Dict - {'user_encoder', 'job_encoder', 'cf'}
    
#     Returns:
#         ratings: Tensor - (batch_size,) similarity scores
#     """
#     # Process texts to tokens
#     job_tokens = batch_process_texts(job_texts, device)
#     resume_tokens = batch_process_texts(resume_texts, device)
    
#     # Get preferences
#     with torch.no_grad():
#         job_prefs = models['job_encoder'](job_tokens)
#         user_prefs = models['user_encoder'](resume_tokens)
#         ratings = models['cf'](user_prefs, job_prefs)
    
#     return ratings

# # Example usage
# models = {
#     'user_encoder': load_user_encoder(...),
#     'job_encoder': load_job_encoder(...),
#     'cf': load_cf_model(...)
# }

# job_batch = ["Senior Python Developer", "ML Engineer"]
# resume_batch = ["5+ years Python...", "TensorFlow experience..."]

# ratings = batch_predict(job_batch, resume_batch, models, device='cuda')
# print(f"Batch ratings: {ratings.cpu().numpy().tolist()}")