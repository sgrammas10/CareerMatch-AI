import os
from jinja2 import Environment, FileSystemLoader
import pdfkit

def render_resume_to_pdf(data, output_path="Streamlit/outputs/edited_resume.pdf"):
    # Use absolute path to templates folder
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("resume_template.html")

    html_out = template.render(data)
    os.makedirs("outputs", exist_ok=True)

    config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")  # update if needed
    pdfkit.from_string(html_out, output_path, configuration=config)

    return output_path


# sample_data = {
#     "name": "Jane Doe",
#     "email": "jane@example.com",
#     "phone": "555-1234",
#     "summary": "Experienced data scientist with expertise in NLP and AI.",
#     "experience": [
#         {
#             "title": "Data Scientist",
#             "company": "TechCorp",
#             "dates": "2020–2023",
#             "description": "Worked on NLP models and AI projects."
#         }
#     ],
#     "education": [
#         {
#             "degree": "B.Sc. in Computer Science",
#             "institution": "MIT",
#             "year": "2019"
#         }
#     ],
#     "skills": ["Python", "Machine Learning", "NLP"]
# }
# render_resume_to_pdf(sample_data)
