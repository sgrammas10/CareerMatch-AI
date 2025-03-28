import torch
import sentencepiece as spm
from torch import nn, optim
from recModel import UserEncoder, JobEncoder, CollaborativeFiltering
from sentence_transformers import SentenceTransformer

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

torch.set_flush_denormal(True)

jobDesc = """SpaceX was founded under the belief that a future where humanity is out exploring the stars is fundamentally more exciting than one where we are not. Today SpaceX is actively developing the technologies to make this possible, with the ultimate goal of enabling human life on Mars.PAINT TECHNICIAN (FALCON 9) - WEEKEND SHIFT
RESPONSIBILITIES: 

Mix, strip, paint, and detail advanced paint systems on launch vehicles
Prepare metallic substrates using mechanical abrasion and chemical etchants
Mix and apply of sealants and adhesives for structural components
Conduct periodic maintenance on various paint spraying equipment (such as gravity-fed, pressure pot, and airless-type)
Accurately use precision measuring tools such as calipers, elcometers, or ohmmeters
Assist with area projects to improve safety, quality, and delivery

BASIC QUALIFICATIONS:

High school diploma or equivalency certificate  
1+ years of experience spraying advanced paint systems (e.g. epoxies, urethanes, solvent-based paints, and/or acrylics) OR a degree/certificate from an accredited trade school/program 

PREFERRED SKILLS AND EXPERIENCE:

3+ years of industry experience in advanced paint systems
Experience in a manufacturing or production environment in the aerospace or aviation industry
Ability to interpret electronic work instructions, process specifications, and technical drawings
Ability to perform basic shop math (such as geometry and trigonometry)
Ability to manage accurate mix ratios, cure schedules, and overcoat windows
Excellent verbal and written communication skills

ADDITIONAL REQUIREMENTS:

Must be able to work the shift listed below, overtime, and/or weekends as needed

Friday to Monday: 4:30AM - 4:30PM


This is a hands-on position that may require standing for up to 8 hours a day and working directly with hardware on the floor
Must be able to lift at least up to 25 lbs. unassisted
Must be comfortable stooping, bending, standing, climbing ladders, or working in tight spaces
Must be willing to work all shifts, overtime, and weekends as required
Able to pass pulmonary function test to obtain respirator certification

COMPENSATION AND BENEFITS:        Pay range:    Paint Technician/Level 1: $22.00 - $24.50/hour    Paint Technician/Level 2: $24.00 - $29.00/hour    Paint Technician/Level 3: $27.50 - $35.00/hour          Your actual level and base salary will be determined on a case-by-case basis and may vary based on the following considerations: job-related knowledge and skills, education, and experience.
Base salary is just one part of your total rewards package at SpaceX. You may also be eligible for long-term incentives, in the form of company stock, stock options, or long-term cash awards, as well as potential discretionary bonuses and the ability to purchase additional stock at a discount through an Employee Stock Purchase Plan. You will also receive access to comprehensive medical, vision, and dental coverage, access to a 401(k) retirement plan, short and long-term disability insurance, life insurance, paid parental leave, and various other discounts and perks. You may also accrue 3 weeks of paid vacation and will be eligible for 10 or more paid holidays per year.               
 ITAR REQUIREMENTS:

To conform to U.S. Government export regulations, applicant must be a (i) U.S. citizen or national, (ii) U.S. lawful, permanent resident (aka green card holder), (iii) Refugee under 8 U.S.C.  1157, or (iv) Asylee under 8 U.S.C.  1158, or be eligible to obtain the required authorizations from the U.S. Department of State. Learn more about the ITAR here.  

SpaceX is an Equal Opportunity Employer; employment with SpaceX is governed on the basis of merit, competence and qualifications and will not be influenced in any manner by race, color, religion, gender, national origin/ethnicity, veteran status, disability status, age, sexual orientation, gender identity, marital status, mental or physical disability or any other legally protected status.
Applicants wishing to view a copy of SpaceXs Affirmative Action Plan for veterans and individuals with disabilities, or applicants requiring reasonable accommodation to the application/interview process should reach out to EEOCompliance@spacex.com."""

resume = """saraht@gmail.com,,"Rensselaer Polytechnic Institute (RPI)                               Troy, NY  
Bachelor of Science in Chemistry                                           Expected May 2024  
 
RELEVANT COURSEWORK  
Chemistry II, Equilibrium Chemistry and Quantitative Analysis, Organic Chem I, Intro to Differential 
Equations","Lab: Organic synthesis, catalysis, upstream/downstream processing, titration  
Instruments: NMR, CV, XRD, GC/MS, UV -vis, IR spec z","Bristol -Myers Squibb                            Boston, MA  
Analytic Development Intern                            June  2022 -Aug 2022  
• Conducted a project using HPLC and UPLC software for method development and optimization with 
a focus on method robustness  
• Learned Fusion QbD and ACD Labs software for use in LC method development and optimization   
• Presented project results and fi ndings to the BMS Analytical Development department  
 
RPI Organic Chemistry La b                    Troy , NY 
Student                   Jan 2022 -May 2022  
• Prepared, synthesized, and purified various drugs in a lab setting  
• Utilized IR Spectroscopy, Mass Spectrometry, and HNMR Spectroscopy to identify unknown 
compounds   
• Wrote up finding s for professor and presented pos ter at research sym posium  
 
RPI Chem Lab                                  Troy , NY 
Undergraduate Researcher            Sept. 2021 -April 2022  
• Synthesized organic ligands and inorganic compounds, on large and small scales, using anaerobic 
techniques  
• Produced complexes of divalent first -row transition metals; studied their interaction with dioxygen  
• Characterized products with 1 H NMR, UV -vis, and IR spectroscopy as well as X -ray crystallography 
and magnetic susceptibility.","& ACTIVITIES  
RPI Undergraduate Association                                Troy, NY  
President  of the Committee on Student Life                Sept 2020 -Present  
• Organized a weeklong convention of 3,000 students with activities geared towards improv ing health 
on campus  
• Connected 376 First Year students to upperclassmen with similar career objectives in a one -on-one 
mentoring relationship","Myers Squibb                            Boston at MA  
Analytic Development Intern                            June; Presented project results and fi ndings to the BMS Analytical Development department  
 
RPI Organic Chemistry La b                    Troy at NY 
Student                   Jan; Prepared at synthesized; Utilized IR Spectroscopy at Mass Spectrometry; Wrote up finding s for professor and presented pos ter at research sym posium  
 
RPI Chem Lab                                  Troy at NY 
Undergraduate Researcher            Sept; Synthesized organic ligands and inorganic compounds at on large and small scales; H NMR at UV; vis at and IR spectroscopy as well as X","""

sp = spm.SentencePieceProcessor()
sp.Load('m.model')

# 2. Convert to tokenized format (mock SentencePiece-like tokenization)
def process_text(text, max_len=512):
    tokens = sp.EncodeAsIds(text)[:max_len]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return torch.tensor(tokens, device=device)

# 3. Process batch (device='cpu' for debugging)
job_tokens = process_text(jobDesc).unsqueeze(0)  # Add batch dimension
resume_tokens = process_text(resume).unsqueeze(0)

user_enc = UserEncoder().to(device)
job_enc = JobEncoder().to(device)
cf_model = CollaborativeFiltering().to(device)
optimizer = optim.Adam([
    *user_enc.parameters(),
    *job_enc.parameters(),
    *cf_model.parameters()
], lr=0.001)
criterion = nn.MSELoss()

# print("Job Tokens:", job_tokens.cpu().numpy())
# print("Resume Tokens:", resume_tokens.cpu().numpy())

# Single inference step with vector printing
def debug_inference():
    with torch.no_grad():
        user_prefs = user_enc(resume_tokens)
        job_prefs = job_enc(job_tokens)
        rating = cf_model(user_prefs, job_prefs)
        
        print("\nUser Preference Vector:", user_prefs.cpu().numpy())
        print("Job Preference Vector:", job_prefs.cpu().numpy())
        print("Final Rating:", rating.cpu().numpy())

# debug_inference()

bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 5. Single training step
def debug_step():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    try:
        with torch.no_grad():
            job_embed = bi_encoder.encode(jobDesc, convert_to_tensor=True, normalize_embeddings=True).cpu()
            resume_embed = bi_encoder.encode(resume, convert_to_tensor=True, normalize_embeddings=True).cpu()
            target = torch.dot(job_embed, resume_embed).unsqueeze(0).to(device)

        print("Bi-Encoder Results:")
        print(f"Job Embedding: {job_embed.numpy()[:5]}...")
        print(f"Resume Embedding: {resume_embed.numpy()[:5]}...")
        print(f"Target Similarity: {target.item():.4f}\n")
    
        # Forward pass
        user_prefs = user_enc(resume_tokens)
        job_prefs = job_enc(job_tokens)
        pred_ratings = cf_model(user_prefs, job_prefs)

        if pred_ratings.dim() == 0:
            pred_ratings = pred_ratings.unsqueeze(0)
    
        # Loss calculation
        loss = criterion(pred_ratings, target)
    
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        print(f"Step loss: {loss.item():.4f}")
        print(f"Sample prediction: {pred_ratings.item():.4f}")
    except RuntimeError as e:
        print(f"MPS Error: {str(e)}")
        print("Falling back to CPU...")
        device = torch.device('cpu')
        debug_step()

# 6. Run debugging step
debug_step()

# 7. Test weight saving/loading
def test_io():
    # Save models
    torch.save(user_enc.state_dict(), "debug_user.pth")
    torch.save(job_enc.state_dict(), "debug_job.pth")
    torch.save(cf_model.state_dict(), "debug_cf.pth")
    
    # Load models to the same device
    loaded_user = UserEncoder().to(device)
    loaded_user.load_state_dict(
        torch.load("debug_user.pth", map_location=device)
    )
    
    # Compare parameters on the same device
    match = all(
        torch.allclose(p1, p2) 
        for p1, p2 in zip(user_enc.parameters(), loaded_user.parameters())
    )
    print("\nWeight loading test:", match)

test_io()