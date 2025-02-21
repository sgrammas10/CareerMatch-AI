from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import os
from language_tool_python import LanguageTool

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')
directory = 'C:/Users/Sebastian Grammas/Desktop/CareerMatchAI/CareerMatch-AI/zensearchData/'

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filename)
        if df.shape[1] >= 5:  # Ensure there are at least 5 columns
            fifth_column = df.iloc[:, 4]  # Get the 5th column (zero-based index)
            
            for value in fifth_column:
                text = value


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
                #print("Summary:", summary)




                tool = LanguageTool('en-US')
                corrected_summary = tool.correct(summary)

                #print("Corrected Summary:", corrected_summary)

                # Add empty columns if necessary
                while df.shape[1] < 6:
                    df[f'Unnamed_{df.shape[1]}'] = ""  # Fill missing columns
                
                df.iloc[:, 5] = "New Value"  # Write to the 6th column

                # Save changes back to CSV
                df.to_csv(filepath, index=False)
                print(f"Updated {filename}")
        else:
            print("Missing 5th column in file:", filename)
    else:
        print("Invalid file format in file:", filename)