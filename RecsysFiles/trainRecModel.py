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

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.set_flush_denormal(True)

# Configure detailed logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
        result = chardet.detect(rawdata)
    # Force fallback to latin-1 if confidence is low
    if result['confidence'] < 0.9:
        return 'latin-1'
    return result['encoding']

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False
    except:
        return True  # Give benefit of doubt

import chardet
import logging
import re

def clean_csv(input_path, output_path, force_encoding=None):
    """
    Cleans a CSV file by:
    1. Detecting encoding
    2. Removing invalid characters
    3. Normalizing line endings
    4. Preserving CSV structure
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
    
    # Define pattern for non-printable ASCII characters
    non_printable = re.compile(b'[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]')
    
    # Counters for statistics
    bad_lines = 0
    bad_chars = 0
    
    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        for line_num, line in enumerate(fin, 1):
            try:
                # Clean line endings and non-printable characters
                cleaned = line.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
                cleaned = non_printable.sub(b'', cleaned)
                
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
    """Robust CSV processing with multiple encoding fallbacks"""
    # 1. Detect encoding with fallback
    try:
        encoding = detect_encoding(input_csv)
    except Exception as e:
        logger.warning(f"Encoding detection failed: {e}, using latin-1")
        encoding = 'latin-1'

    logger.info(f"Using encoding: {encoding}")

    # 2. Read CSV with error-tolerant decoding
    try:
        df = pd.read_csv(
            input_csv,
            encoding=encoding,
            engine='python',
            on_bad_lines=lambda bad: logger.warning(f"Skipping bad line: {bad}"),
            dtype='object',
            header=0,
            quoting=3,
            error_bad_lines=False,
            warn_bad_lines=True,
            sep=None,  # Auto-detect separator
            encoding_errors='replace'  # Replace invalid chars with �
        )
    except UnicodeDecodeError:
        # Final fallback to latin-1 with error replacement
        df = pd.read_csv(
            input_csv,
            encoding='latin-1',
            engine='python',
            error_bad_lines=False,
            warn_bad_lines=True,
            sep=None,
            encoding_errors='replace'
        )
    
    logger.info(f"Raw CSV contains {len(df)} rows")
    logger.debug(f"Sample raw data:\n{df.head(2)}")

    # 3. Flexible column name matching
    job_col = next((col for col in df.columns if 'role' in col.lower()), None)
    resume_col = next((col for col in df.columns if 'resume' in col.lower()), None)
    
    if not job_col or not resume_col:
        raise ValueError("Could not find required columns in CSV")

    # 4. Relaxed data collection
    jobs = []
    resumes = []
    valid_rows = 0
    
    for idx, row in df.iterrows():
        try:
            # Basic cleaning
            job_text = str(row[job_col]).strip()[:10000]  # Limit text length
            resume_text = str(row[resume_col]).strip()[:10000]
            
            # Relaxed validation
            if len(job_text) < 10 or len(resume_text) < 10:
                logger.debug(f"Skipping row {idx}: Insufficient text")
                continue
                
            # Optional: Comment out English check if problematic
            # if not is_english(job_text) or not is_english(resume_text):
            #     continue
                
            jobs.append(job_text)
            resumes.append(resume_text)
            valid_rows += 1
            
        except Exception as e:
            logger.warning(f"Row {idx} error: {str(e)}")
            continue

    logger.info(f"Found {valid_rows} valid entries (from {len(df)} raw rows)")
    
    if valid_rows == 0:
        raise ValueError("No valid data found after processing")

    # 5. Create combinations
    product_df = pd.DataFrame(
        [(j, r) for j in jobs for r in resumes],
        columns=['RoleDescription', 'Resume']
    )
    
    logger.info(f"Generated {len(product_df)} pairs")

    # 6. Process embeddings with error handling
    bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    batch_size = 16  # Smaller batches for stability
    targets = []
    
    for i in range(0, len(product_df), batch_size):
        try:
            batch = product_df.iloc[i:i+batch_size]
            
            job_embs = bi_encoder.encode(
                batch['RoleDescription'].tolist(),
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            resume_embs = bi_encoder.encode(
                batch['Resume'].tolist(),
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            batch_targets = (job_embs * resume_embs).sum(dim=1)
            targets.extend(batch_targets.cpu().numpy())
            
        except Exception as e:
            logger.error(f"Batch {i//batch_size} failed: {str(e)}")
            # Fallback: Use neutral similarity score
            targets.extend([0.5] * len(batch))
            continue

    product_df['target'] = targets
    product_df.to_csv(output_csv, index=False)
    logger.info(f"Successfully saved {len(product_df)} pairs to {output_csv}")

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
            'target': torch.tensor(row['target'], dtype=torch.float32)
        }

def train():
    #Clean the CSV
    clean_csv('RawTextData.csv', 'cleaned_data.csv')

    # Precompute data
    precompute_targets('cleaned_data.csv', 'preprocessed_data.csv')

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