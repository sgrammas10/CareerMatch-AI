from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from torch import nn, optim
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from recModel import FullModel
from langdetect import detect, LangDetectException
import logging
import chardet
import re
import csv
import os

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.set_flush_denormal(True)

# Configure detailed logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding'] if result['confidence'] > 0.7 else 'latin-1'

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False
    except:
        return True  # Give benefit of doubt

def read_csv_with_fallback(file_path):
    """Read CSV without pandas, returns (headers, rows)"""
    encoding = detect_encoding(file_path)
    logger.info(f"Reading {file_path} with encoding: {encoding}")
    
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
            rows = list(reader)
        except StopIteration:
            raise ValueError("Empty CSV file")
    
    # Find target columns
    job_col = next(i for i,h in enumerate(headers) if 'role' in h.lower())
    resume_col = next(i for i,h in enumerate(headers) if 'resume' in h.lower())
    
    return headers, rows, job_col, resume_col

def clean_csv(input_path, output_path, force_encoding=None):
    """
    Cleans a CSV file by:
    1. Removing all non-alphanumeric characters except basic punctuation
    2. Normalizing line endings
    3. Preserving CSV structure
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Detect encoding if not forced
    if not force_encoding:
        with open(input_path, 'rb') as f:
            rawdata = f.read(100000)  # Read first 100KB for detection
            result = chardet.detect(rawdata)
            encoding = result['encoding'] if result['confidence'] > 0.7 else 'latin-1'
    else:
        encoding = force_encoding
    
    logger.info(f"Cleaning {input_path} with detected encoding: {encoding}")
    
    # Strict allowed characters regex (alphanumerics, space, and basic punctuation)
    allowed_chars = re.compile(
        b'[^'
        b'a-zA-Z0-9'    # Alphanumerics
        b'\x20'         # Space
        b'!@#$%^&*()_+-=[]{}|;:,./<>?~`\'"\\'  # Common punctuation
        b'\n'           # Newline
        b']'
    )
    
    # Counters for statistics
    bad_lines = 0
    bad_chars = 0
    
    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        for line_num, line in enumerate(fin, 1):
            try:
                # Normalize line endings to Unix-style
                cleaned = line.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
                
                # Remove non-allowed characters
                cleaned = allowed_chars.sub(b'', cleaned)
                
                # Detect character replacement
                if cleaned != line:
                    bad_chars += len(line) - len(cleaned)
                
                # Write cleaned line
                fout.write(cleaned)
                
            except Exception as e:
                bad_lines += 1
                logger.warning(f"Skipping line {line_num}: {str(e)}")
                continue
    
    logger.info(f"Cleaning complete. Removed {bad_chars} bad characters, skipped {bad_lines} lines")
    logger.info(f"Clean file saved to {output_path}")
    return output_path

def precompute_targets(input_csv, output_csv):
    # Read CSV data
    try:
        headers, rows, job_idx, resume_idx = read_csv_with_fallback(input_csv)
    except Exception as e:
        logger.error(f"CSV read failed: {e}")
        raise

    # Process valid entries
    jobs = []
    resumes = []
    valid_rows = 0
    
    for row_idx, row in enumerate(rows):
        try:
            # Ensure row has enough columns
            if len(row) < max(job_idx, resume_idx) + 1:
                logger.warning(f"Skipping row {row_idx}: Not enough columns")
                continue
                
            job_text = row[job_idx].strip()[:10000]
            resume_text = row[resume_idx].strip()[:10000]
            
            if len(job_text) < 10 or len(resume_text) < 10:
                logger.debug(f"Skipping row {row_idx}: Insufficient text")
                continue
                
            jobs.append(job_text)
            resumes.append(resume_text)
            valid_rows += 1
            
        except Exception as e:
            logger.warning(f"Skipping row {row_idx}: {str(e)}")
            continue

    logger.info(f"Found {valid_rows} valid entries")
    
    if valid_rows == 0:
        raise ValueError("No valid data found after processing")

    # Generate ALL combinations
    total_pairs = len(jobs) * len(resumes)
    logger.info(f"Generating {total_pairs} pairs from {len(jobs)} jobs and {len(resumes)} resumes")

    # Write output with progress
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['RoleDescription', 'Resume', 'target'])
        
        bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        batch_size = 32
        
        # Process in sequential batches
        for batch_idx in range(0, total_pairs, batch_size):
            # Get current batch range
            batch_end = min(batch_idx + batch_size, total_pairs)
            
            # Collect job-resume pairs for this batch
            batch_pairs = []
            for pair_num in range(batch_idx, batch_end):
                i = pair_num // len(resumes)  # Job index
                j = pair_num % len(resumes)   # Resume index
                batch_pairs.append((jobs[i], resumes[j]))
            
            # Encode entire batch
            job_texts = [p[0] for p in batch_pairs]
            resume_texts = [p[1] for p in batch_pairs]
            
            job_embs = bi_encoder.encode(job_texts, normalize_embeddings=True)
            resume_embs = bi_encoder.encode(resume_texts, normalize_embeddings=True)
            targets = (job_embs * resume_embs).sum(axis=1)
            
            # Write batch to CSV
            for (job, resume), target in zip(batch_pairs, targets):
                writer.writerow([job, resume, float(target)])
            
            logger.info(f"Processed {batch_end}/{total_pairs} pairs")

    logger.info(f"Successfully generated {total_pairs} pairs in {output_csv}")

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
            'job_tokens': process_text(row['RoleDescription']),
            'resume_tokens': process_text(row['Resume']),
            'target': torch.tensor(row['target'], dtype=torch.float32),
            'job_text': row['RoleDescription'],  # Include raw text
            'resume_text': row['Resume']  # Include raw text
        }

def train():
    #Clean the CSV
    #clean_csv('RawTextData.csv', 'cleaned_data.csv')

    # Precompute data
    precompute_targets('cleaned_data.csv', 'preprocessed_data.csv')

    # Load data
    dataset = JobResumeDataset('preprocessed_data.csv')
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    # Model setup
    model = FullModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    criterion = nn.MSELoss()

    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        batch_idx = 1
        
        for batch in dataloader:
            job_tokens = batch['job_tokens'].to(device)
            resume_tokens = batch['resume_tokens'].to(device)
            targets = batch['target'].to(device)

            if batch_idx % 10 == 0:  # Adjust frequency as needed
                print(f"\n--- Batch {batch_idx} Samples ---")
                for i in range(3):
                    if i >= len(batch['job_text']):
                        break
                    job_sample = ' '.join(batch['job_text'][i].split()[:15])
                    resume_sample = ' '.join(batch['resume_text'][i].split()[:15])
                    rating = batch['target'][i].item()
                    print(f"Sample {i+1}:")
                    print(f"Job Posting: {job_sample}...")
                    print(f"Resume: {resume_sample}...")
                    print(f"Rating: {rating:.4f}\n")
            
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

            batch_idx += 1
        
        epoch_loss_avg = epoch_loss/len(dataloader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss_avg:.4f}")

        loss_str = f"{epoch_loss_avg:.4f}".replace('.', '_')
        
        # Save checkpoints with loss in filename
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}_loss_{loss_str}_full_model.pth")
        torch.save(model.user_encoder.state_dict(), f"checkpoints/epoch_{epoch+1}_loss_{loss_str}_user_encoder.pth")
        torch.save(model.job_encoder.state_dict(), f"checkpoints/epoch_{epoch+1}_loss_{loss_str}_job_encoder.pth")
        torch.save(model.cf.state_dict(), f"checkpoints/epoch_{epoch+1}_loss_{loss_str}_cf.pth")
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.4f}")

    # torch.save(model.state_dict(), "full_model.pth")
    # torch.save(model.user_encoder.state_dict(), "user_encoder_final.pth")
    # torch.save(model.job_encoder.state_dict(), "job_encoder_final.pth")
    # torch.save(model.cf.state_dict(), "cf_final.pth")
    print("Model weights saved")

if __name__ == "__main__":
    train()