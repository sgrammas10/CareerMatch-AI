import torch
from recModel import UserEncoder, JobEncoder, CollaborativeFiltering
import sentencepiece as spm

def load_user_encoder(weight_path, device="cpu"):
    encoder = UserEncoder().to(device)
    encoder.load_state_dict(torch.load(weight_path, map_location=device))
    encoder.eval()
    return encoder

# user_encoder = load_user_encoder(
#     weight_path="models/user_encoder_final.pth",
#     device="cuda"
# )

def load_job_encoder(weight_path, device="cpu"):
    encoder = JobEncoder().to(device)
    encoder.load_state_dict(torch.load(weight_path, map_location=device))
    encoder.eval()
    return encoder

# job_encoder = load_job_encoder(
#     weight_path="models/job_encoder_final.pth",
#     device="cuda"
# )

def load_cf_model(weight_path, device="cpu"):
    cf = CollaborativeFiltering().to(device)
    cf.load_state_dict(torch.load(weight_path, map_location=device))
    cf.eval()
    return cf

# cf_model = load_cf_model(
#     weight_path="models/cf_final.pth",
#     device="cuda"
# )

# Initialize components
sp = spm.SentencePieceProcessor()
sp.Load('m.model')
user_encoder = UserEncoder().to("cpu")
job_encoder = JobEncoder().to("cpu")
cf_model = CollaborativeFiltering().to("cpu")

# Inference pipeline (one by one)
def predict_rating(job_pref, user_pref):
    with torch.no_grad():
        return cf_model(user_pref, job_pref)

# Example usage with new data
def process_new_job(text, max_seq_len=5000):
    tokens = sp.EncodeAsIds(text)
    tokens = [min(t, 30000-1) for t in tokens]  # Clamp tokens
    tokens += [0] * (max_seq_len - len(tokens))
    tokens = torch.tensor(tokens).unsqueeze(0).to("cpu")
    return job_encoder(tokens)

def process_new_user(text, max_seq_len=5000):
    tokens = sp.EncodeAsIds(text)
    tokens = [min(t, 30000-1) for t in tokens]  # Clamp tokens
    tokens += [0] * (max_seq_len - len(tokens))
    tokens = torch.tensor(tokens).unsqueeze(0).to("cpu")
    return user_encoder(tokens)

# Process new job and resume
new_job = process_new_job("Senior Python Developer")

new_resume = process_new_user("5 years Python experience...")

# Get rating
rating = predict_rating(new_job, new_resume)
# print(f"Predicted alignment score: {rating.item():.4f}")


def batch_process_jobs(texts, device='cpu', max_seq_len=5000):
    """Process batch of job texts to preference vectors"""
    batch_tokens = []
    for text in texts:
        tokens = sp.EncodeAsIds(text)[:max_seq_len]
        tokens = [min(t, 30000-1) for t in tokens]  # Use actual vocab size
        tokens += [0] * (max_seq_len - len(tokens))
        batch_tokens.append(tokens)
    
    # Convert to tensor and encode
    token_tensor = torch.tensor(batch_tokens, device=device)
    with torch.no_grad():
        return job_encoder(token_tensor)  # Returns (batch_size, 256)

def batch_process_users(texts, device='cpu', max_seq_len=5000):
    """Process batch of user texts to preference vectors"""
    batch_tokens = []
    for text in texts:
        tokens = sp.EncodeAsIds(text)[:max_seq_len]
        tokens = [min(t, 30000-1) for t in tokens]  # Use actual vocab size
        tokens += [0] * (max_seq_len - len(tokens))
        batch_tokens.append(tokens)
    
    # Convert to tensor and encode
    token_tensor = torch.tensor(batch_tokens, device=device)
    with torch.no_grad():
        return user_encoder(token_tensor)

def batch_predict(job_prefs, user_prefs):
    """
    Args:
        job_prefs: List[Tensor] - Batch of job preferences
        user_prefs: List[Tensor] - Batch of user preferences
    
    Returns:
        ratings: Tensor - (batch_size,) similarity scores
    """
    # Get preferences
    with torch.no_grad():
        return cf_model(user_prefs, job_prefs)
    

# Example usage
job_batch = ["Senior Python Developer", "ML Engineer"]
resume_batch = ["5+ years Python...", "TensorFlow experience..."]

jobs_pref = batch_process_jobs(job_batch)
users_pref = batch_process_users(resume_batch)

ratings = batch_predict(jobs_pref, users_pref)
# print(f"Batch ratings: {ratings.cpu().numpy().tolist()}")


def calculate_adjustment(current_vec: torch.Tensor, 
                        target_vec: torch.Tensor, 
                        multiplier: float,
                        reverse: bool = False) -> torch.Tensor:
    """
    Core adjustment calculation used by all functions
    - multiplier: percentage of distance to move (0.0-1.0)
    - reverse: move away from target if True
    """
    direction = target_vec - current_vec
    distance = torch.norm(direction)
    
    # Handle zero distance case
    if distance < 1e-6:  # Prevent division by zero
        return current_vec
    
    normalized_dir = direction / distance
    adjustment = multiplier * distance * normalized_dir
    return current_vec + (-adjustment if reverse else adjustment)

# ========================
# USER ACTION FUNCTIONS
# ========================

def user_save_for_later(user_pref: torch.Tensor, company_pref: torch.Tensor) -> torch.Tensor:
    multiplier: float = 0.05 # Move 5% of distance closer

    return calculate_adjustment(user_pref, company_pref, multiplier)

def user_apply(user_pref: torch.Tensor, company_pref: torch.Tensor) -> torch.Tensor:
    multiplier: float = 0.2 # Move 20% of distance closer

    return calculate_adjustment(user_pref, company_pref, multiplier)

def user_reject(user_pref: torch.Tensor, company_pref: torch.Tensor) -> torch.Tensor:
    multiplier: float = 0.2 # Move 20% of distance away

    return calculate_adjustment(user_pref, company_pref, multiplier, reverse=True)

# ========================
# COMPANY ACTION FUNCTIONS
# ========================

def company_reject(company_pref: torch.Tensor, user_pref: torch.Tensor) -> torch.Tensor:
    multiplier: float = 0.2 # Move 20% of distance away

    return calculate_adjustment(company_pref, user_pref, multiplier, reverse=True)

def company_accept(company_pref: torch.Tensor, user_pref: torch.Tensor) -> torch.Tensor:
    multiplier: float = 0.2 # Move 20% of distance closer

    return calculate_adjustment(company_pref, user_pref, multiplier)

def company_resume_viewed(company_pref: torch.Tensor, user_pref: torch.Tensor) -> torch.Tensor:
    multiplier: float = 0.05 # Move 5% of distance closer

    return calculate_adjustment(company_pref, user_pref, multiplier)

# ========================
# USAGE EXAMPLE
# ========================

# if __name__ == "__main__":
#     # Initialize sample vectors
#     user_pref = torch.randn(256)  # User preference vector
#     company_pref = torch.randn(256)  # Company preference vector
    
#     print("Original cosine similarity:", torch.cosine_similarity(user_pref, company_pref, dim=0).item())
    
#     # User saves for later
#     updated_user = user_save_for_later(user_pref, company_pref)
#     print("After save_for_later similarity:", torch.cosine_similarity(updated_user, company_pref, dim=0).item())
    
#     # Company views resume
#     updated_company = company_resume_viewed(company_pref, user_pref)
#     print("After resume_viewed similarity:", torch.cosine_similarity(user_pref, updated_company, dim=0).item())
    
#     # User applies
#     updated_user = user_apply(updated_user, updated_company)
#     print("After apply similarity:", torch.cosine_similarity(updated_user, updated_company, dim=0).item())
    
#     # Company rejects
#     updated_company = company_reject(updated_company, updated_user)
#     print("After rejection similarity:", torch.cosine_similarity(updated_user, updated_company, dim=0).item())