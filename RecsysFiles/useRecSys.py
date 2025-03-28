import torch
from recModel import UserEncoder, JobEncoder, CollaborativeFiltering
import sentencepiece as spm

# # user_encoder = load_user_encoder(
# #     weight_path="models/user_encoder_final.pth",
# #     device="cuda"
# # )

# def load_user_encoder(weight_path, device="cpu"):
#     encoder = UserEncoder().to(device)
#     encoder.load_state_dict(torch.load(weight_path, map_location=device))
#     encoder.eval()
#     return encoder

# # job_encoder = load_job_encoder(
# #     weight_path="models/job_encoder_final.pth",
# #     device="cuda"
# # )

# def load_job_encoder(weight_path, device="cpu"):
#     encoder = JobEncoder().to(device)
#     encoder.load_state_dict(torch.load(weight_path, map_location=device))
#     encoder.eval()
#     return encoder

# # cf_model = load_cf_model(
# #     weight_path="models/cf_final.pth",
# #     device="cuda"
# # )

# def load_cf_model(weight_path, device="cpu"):
#     cf = CollaborativeFiltering().to(device)
#     cf.load_state_dict(torch.load(weight_path, map_location=device))
#     cf.eval()
#     return cf

# # Initialize components
# user_encoder = UserEncoder().to("cpu")

# job_encoder = JobEncoder().to("cpu")

# cf_model = CollaborativeFiltering().to("cpu")






# # Inference pipeline (one by one)
# def predict_rating(job_pref, user_pref):
#     with torch.no_grad():
#         return cf_model(user_pref, job_pref)

# # Example usage with new data
# def process_new_input(text, max_seq_len=100000):
#     sp = spm.SentencePieceProcessor()
#     sp.Load('m.model')
#     tokens = sp.EncodeAsIds(text)
#     tokens += [0] * (max_seq_len - len(tokens))
#     return torch.tensor(tokens).unsqueeze(0).to("cpu")

# # Process new job and resume
# new_job = process_new_input("Senior Python Developer")
# job = job_encoder(new_job)

# new_resume = process_new_input("5 years Python experience...")
# user = user_encoder(new_resume)

# # Get rating
# rating = predict_rating(job, user)
# print(f"Predicted alignment score: {rating.item():.4f}")


# models = {
#     'user_encoder': UserEncoder().to("cpu"),
#     'job_encoder': JobEncoder().to("cpu"),
#     'cf': CollaborativeFiltering().to("cpu")
# }

# def batch_process_texts(texts, device='cpu', max_seq_len=100000):
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

# job_batch = ["Senior Python Developer", "ML Engineer"]
# resume_batch = ["5+ years Python...", "TensorFlow experience..."]

# ratings = batch_predict(job_batch, resume_batch, models, device='cuda')
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

if __name__ == "__main__":
    # Initialize sample vectors
    user_pref = torch.randn(256)  # User preference vector
    company_pref = torch.randn(256)  # Company preference vector
    
    print("Original cosine similarity:", torch.cosine_similarity(user_pref, company_pref, dim=0).item())
    
    # User saves for later
    updated_user = user_save_for_later(user_pref, company_pref)
    print("After save_for_later similarity:", torch.cosine_similarity(updated_user, company_pref, dim=0).item())
    
    # Company views resume
    updated_company = company_resume_viewed(company_pref, user_pref)
    print("After resume_viewed similarity:", torch.cosine_similarity(user_pref, updated_company, dim=0).item())
    
    # User applies
    updated_user = user_apply(updated_user, updated_company)
    print("After apply similarity:", torch.cosine_similarity(updated_user, updated_company, dim=0).item())
    
    # Company rejects
    updated_company = company_reject(updated_company, updated_user)
    print("After rejection similarity:", torch.cosine_similarity(updated_user, updated_company, dim=0).item())