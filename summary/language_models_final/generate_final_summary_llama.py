import pandas as pd
from sys import argv
import time
import json
from pathlib import Path
import requests
from tqdm import tqdm
import numpy as np
import sys
import os

llama_cot = "llama-cot"

start = time.time()
#typ = sys.argv[1]
begin = int(sys.argv[1])
end = int(sys.argv[2])
node = str(sys.argv[3])

mode_llama = "top_5_sent"

# Relative path to the dataset
subdir = ''
sorted_list = [str(num) for num in range(begin,end)]
output_filename = "missing_context_llama.txt"

# Function to split and truncate input string based on token count
def split_and_truncate(input_string, max_tokens=512):
    tokens = input_string.split()
    if len(tokens) > max_tokens:
        truncated_string = ' '.join(tokens[:max_tokens])
    else:
        truncated_string = input_string
    return truncated_string

# Function to execute prompt and get AI response
def execute_prompt(prompt):
    print(prompt)
    args = {"batch": [prompt]}
    request = requests.post(f"http://{node}:5000", json=args)
    results = request.json()
    print(results)
    response = results["data"][0]
    return response

# Loop through directories and files in the specified range
for dir in tqdm(sorted_list, leave=False):
    for filename in os.listdir(os.path.join(subdir, dir)):
        print(filename)

        # File paths for LLaMA-CoT context and summary
        file_name_llama = f'context_{mode_llama}_{filename}.json'
        context_file_llama = f'{subdir}/{dir}/{filename}/{file_name_llama}'
        summary_path = f"{subdir}/{dir}/{filename}/summary_LLaMA-CoT-citance-SciBert-top5_{filename}.csv"

        # Check if LLaMA-CoT context file exists
        if not os.path.exists(context_file_llama):
            with open(output_filename, "w") as output_file:
                output_file.write(f"{filename}\n")
            continue

        # Check if summary file already exists
        if os.path.exists(summary_path):
            continue

        # Read LLaMA-CoT context file and preprocess
        df_llama = pd.read_json(context_file_llama, orient="records", dtype=object)
        df_llama = df_llama.rename(columns={'citance_single_sbert': 'citance-scibert'})
        df_llama = df_llama[['citance_No', 'citing_paper_id', 'citance-scibert']]

        # Create a new column for LLaMA-CoT summary
        df_llama["LLaMA-CoT-citance-SciBert-top5"] = np.nan

        # Iterate through rows and generate paraphrased text
        for index, row in tqdm(df_llama.iterrows(), total=df_llama.shape[0]):
            print(F"length of input {row['citance-scibert']}")

            input_string = split_and_truncate(row['citance-scibert'])

            prompt_top5_citance_single_sbert = f"""
            ### Instruction:
            A chat between a curious human and an artificial intelligence assistant. The assistant knows how to paraphrase scientific text and the user will provide the scientific text for the assistant to paraphrase.

            ### Input: 
            Generate a coherent paraphrased text for the following scientific text:  "{input_string}"

            ### Output: """

            # Check if the summary is not already generated
            if pd.isna(df_llama.at[index, "LLaMA-CoT-citance-SciBert-top5"]):
                response1 = execute_prompt(prompt_top5_citance_single_sbert)
                df_llama.at[index, "LLaMA-CoT-citance-SciBert-top5"] = response1

            # Save the generated summary to a CSV file
            df_llama.to_csv(summary_path, index=False)