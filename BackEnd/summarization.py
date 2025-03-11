from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import os
from language_tool_python import LanguageTool

# Load tokenizer and model
def summarize_jobs(): 
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    directory = 'C:/Users/Sebastian Grammas/Desktop/CareerMatchAI/CareerMatch-AI/zensearchData/'
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            if (filename=="company_data.csv"):
                continue
            df = pd.read_csv(filepath)
            if df.shape[1] >= 10:  # Ensure there are at least 10 columns
                tenth_column = df.iloc[:, 9]  # Get the 10th column (zero-based index)
                for index, value in enumerate(tenth_column):
                    text = value
                    input_text = "summarize: " + text
                    # Tokenize and encode
                    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
                    # Generate summary
                    summary_ids = model.generate(
                        inputs,
                        max_length=1000,
                        min_length=25,
                        length_penalty=0.5,
                        num_beams=8,
                        no_repeat_ngram_size=3,
                        early_stopping=True
                    )

                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    tool = LanguageTool('en-US')
                    corrected_summary = tool.correct(summary)

                    # Ensure at least 11 columns exist
                    while df.shape[1] < 11:
                        df[f'Unnamed_{df.shape[1]}'] = ""  

                    df.iloc[index, 10] = corrected_summary  # Write only to the specific row

                    # Save changes back to CSV
                    df.to_csv(filepath, index=False)


                    # Save changes back to CSV
                    df.to_csv(filepath, index=False)
                    print(f"Updated {filename}")
            else:
                print("Missing 5th column in file:", filename)
        else:
            print("Invalid file format in file:", filename)