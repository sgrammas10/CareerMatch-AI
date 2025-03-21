import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from torch import nn, optim
from recModel import FullModel
import sentencepiece as spm

# Train initial model, uncomment if m.model and m.vocab get deleted
# spm.SentencePieceTrainer.train('--input=combined_csvs.txt --model_prefix=m --vocab_size=5000')

# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('m.model')
# #encoding = sp.EncodeAsIds("The quick sly fox jumped over the lazy rabbit")
# #print(encoding)

# 1. Load the CSV
class JobResumeDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_seq_len=10000):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        job_text = self.df.iloc[idx]['RoleDescription']
        resume_text = self.df.iloc[idx]['Resume']
        
        # Tokenize using SentencePiece (pad/truncate to max_seq_len)
        job_tokens = self.tokenizer.encode(job_text)[:self.max_seq_len] + \
                    [0] * (self.max_seq_len - len(self.tokenizer.encode(job_text)))
        resume_tokens = self.tokenizer.encode(resume_text)[:self.max_seq_len] + \
                       [0] * (self.max_seq_len - len(self.tokenizer.encode(resume_text)))
        
        return {
            'job_tokens': torch.tensor(job_tokens, dtype=torch.long),
            'resume_tokens': torch.tensor(resume_tokens, dtype=torch.long),
            'job_text': job_text,
            'resume_text': resume_text
        }

# 2. Custom collate function to handle text
def collate_fn(batch):
    return {
        'job_tokens': torch.stack([item['job_tokens'] for item in batch]),
        'resume_tokens': torch.stack([item['resume_tokens'] for item in batch]),
        'job_texts': [item['job_text'] for item in batch],
        'resume_texts': [item['resume_text'] for item in batch]
    }

# 3. Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FullModel().to(device)
bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # On CPU

# 4. Training loop
def train(csv_path, save_dir, epochs=10, batch_size=32):
    dataset = JobResumeDataset(csv_path, tokenizer=sp)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Move tokenized inputs to device
            job_tokens = batch['job_tokens'].to(device)
            resume_tokens = batch['resume_tokens'].to(device)
            
            # Generate Bi-Encoder ratings for the batch (on CPU)
            with torch.no_grad():
                job_embeddings = bi_encoder.encode(
                    batch['job_texts'], 
                    convert_to_tensor=True, 
                    normalize_embeddings=True
                )
                resume_embeddings = bi_encoder.encode(
                    batch['resume_texts'], 
                    convert_to_tensor=True, 
                    normalize_embeddings=True
                )
                targets = (job_embeddings * resume_embeddings).sum(dim=1).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            preds = model(job_tokens, resume_tokens)
            loss = criterion(preds, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Save weights after each epoch
            torch.save(model.user_encoder.state_dict(), f"{save_dir}/user_encoder_epoch{epoch+1}.pth")
            torch.save(model.job_encoder.state_dict(), f"{save_dir}/job_encoder_epoch{epoch+1}.pth")
            torch.save(model.cf.state_dict(), f"{save_dir}/cf_epoch{epoch+1}.pth")
        
        # Save final weights
        torch.save(model.user_encoder.state_dict(), f"{save_dir}/user_encoder_final.pth")
        torch.save(model.job_encoder.state_dict(), f"{save_dir}/job_encoder_final.pth")
        torch.save(model.cf.state_dict(), f"{save_dir}/cf_final.pth")
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# 5. Run training
train('your_data.csv', './', epochs=10)