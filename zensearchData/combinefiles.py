import os
import glob
import pandas as pd

def combine_files(input_folder, output_csv, output_txt):
    # Collect all CSV files in the folder excluding 'company_data.csv'
    csv_files = [f for f in glob.glob(os.path.join(input_folder, "*.csv")) if "company_data.csv" not in f]

    if not csv_files:
        print("No CSV files found in the folder.")
        return

    # Combine all CSV files into one DataFrame
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Save combined CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined CSV saved as: {output_csv}")

    # Save combined text file
    with open(output_txt, 'w', encoding='utf-8') as txt_file:
        for f in csv_files:
            with open(f, 'r', encoding='utf-8') as csv_file:
                txt_file.write(csv_file.read())
                txt_file.write('\n')  # Add newline for clarity

    print(f"Combined text file saved as: {output_txt}")


input_folder = ""  # Change this to folder path later if needed
combine_files(input_folder, "combined_csvs.csv", "combined_csvs.txt")
