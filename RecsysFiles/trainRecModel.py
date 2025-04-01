from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from torch import nn, optim
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from recModel import FullModel
import os

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.set_flush_denormal(True)

def precompute_targets(input_csv, output_csv):
    """Add SentenceTransformer similarity scores to dataset"""
    df = pd.read_csv(input_csv)
    bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Batch process for efficiency
    batch_size = 32
    targets = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        job_embs = bi_encoder.encode(batch['job_description'].tolist(), 
                                   convert_to_tensor=True, normalize_embeddings=True)
        resume_embs = bi_encoder.encode(batch['resume'].tolist(),
                                      convert_to_tensor=True, normalize_embeddings=True)
        batch_targets = (job_embs * resume_embs).sum(dim=1)
        targets.extend(batch_targets.cpu().numpy())
    
    df['target'] = targets
    df.to_csv(output_csv, index=False)

class JobResumeDataset(Dataset):
    def __init__(self, csv_path, max_seq_len=900):
        self.df = pd.read_csv(csv_path)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tokenize with clamping
        def process_text(text):
            sp = spm.SentencePieceProcessor()
            sp.Load('m.model')
            tokens = sp.EncodeAsIds(text)[:self.max_seq_len]
            tokens = [min(t, 30000-1) for t in tokens]
            tokens += [0] * (self.max_seq_len - len(tokens))
            return torch.tensor(tokens, dtype=torch.long)
        
        return {
            'job_tokens': process_text(row['job_description']),
            'resume_tokens': process_text(row['resume']),
            'target': torch.tensor(row['target'], dtype=torch.float32)
        }

def train():
    # Precompute data
    precompute_targets('raw.csv', 'preprocessed_data.csv')

    # Load data
    dataset = JobResumeDataset('preprocessed_data.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model setup
    model = FullModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            job_tokens = batch['job_tokens'].to(device)
            resume_tokens = batch['resume_tokens'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            job_prefs = model.job_encoder(job_tokens)
            user_prefs = model.user_encoder(resume_tokens)
            preds = model.cf(user_prefs, job_prefs)
            
            # Compute loss
            loss = criterion(preds, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "full_model.pth")
    torch.save(model.user_encoder.state_dict(), "user_encoder_final.pth")
    torch.save(model.job_encoder.state_dict(), "job_encoder_final.pth")
    torch.save(model.cf.state_dict(), "cf_final.pth")
    print("Model weights saved")

if __name__ == "__main__":
    train()