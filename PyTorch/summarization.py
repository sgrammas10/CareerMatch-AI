from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from language_tool_python import LanguageTool

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')
directory = 'C:/Users/Sebastian Grammas/Desktop/CareerMatchAI/CareerMatch-AI/zensearchData/company_data.csv'

df = pd.read_csv(directory)

#need to implement looping here

input_text = "summarize: " + text
# Tokenize and encode
inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
# Generate summary
summary_ids = model.generate(
    inputs,
    max_length=1000,
    min_length=100,
    length_penalty=0.5,
    num_beams=8,
    no_repeat_ngram_size=3,
    early_stopping=True
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Summary:", summary)




tool = LanguageTool('en-US')
corrected_summary = tool.correct(summary)
print("Corrected Summary:", corrected_summary)
