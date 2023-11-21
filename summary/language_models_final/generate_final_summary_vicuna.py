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

vicuna = "vicuna"

start = time.time()
#typ = sys.argv[1]
begin = int(sys.argv[1])
end = int(sys.argv[2])
node = str(sys.argv[3])

mode_vic = "top_2_para"

# Set relative path
subdir = F''
# Generating a list of sorted numbers within the specified range
sorted_list = [str(num) for num in range(begin, end)]

# Output file for missing context
output_filename = "missing_context.txt"

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
    s2 = time.time()
    print(F"start request {s2}")
    request = requests.post(f"http://{node}:5001", json=args)
    print(F"received {time.time() - s2}")
    results = request.json()
    print(results["data"][0])
    response = results["data"][0]
    return response

# Loop through directories and files in the specified range
for dir in sorted_list:
    for filename in os.listdir(os.path.join(subdir, dir)):
        start = time.time()
        print(filename)
        file_name_vic = f'context_{mode_vic}_{filename}.json'
        context_file_vic = f'{subdir}/{dir}/{filename}/{file_name_vic}'

        summary_path = f"{subdir}/{dir}/{filename}/summary_Vicuna-similar-BM25-top2_{filename}.csv"
        
        # Check if Vicuna context file exists
        if not os.path.exists(context_file_vic):
            with open(output_filename, "w") as output_file:
                output_file.write(f"{filename}\n")
            continue

        # Check if summary file already exists
        if os.path.exists(summary_path):
            continue
        
        # Read Vicuna context file and preprocess
        df_vic = pd.read_json(context_file_vic, orient="records", dtype=object)
        print(F"Read in context file {time.time() - start}")

        df_vic = df_vic.rename(columns={'citance_embed_bm25': 'similar-bm25'})
        df_vic = df_vic[['citance_No', 'citing_paper_id', 'similar-bm25']]
        df_vic["Vicuna-similar-BM25-top2"] = np.nan
        print(F"Renaming: {time.time() - start}")

        # Iterate through rows and generate summarized text
        for index, row in df_vic.iterrows():
            print(F"length of input {len(row['similar-bm25'].split())}")

            input_string = split_and_truncate(row['similar-bm25'])
            print(F"Split and truncate {time.time() - start}")

            prompt_top2_citance_embed_bm25 = f"""
            ### Instruction:
            A chat between a curious human and an artificial intelligence assistant. The assistant knows how to summarize scientific text, and the user will provide the scientific text for the assistant to summarize.

            ### Input: 
            Generate a coherent summary for the following scientific text in not more than 5 sentences:  "{input_string}"

            ### Output: """
            
            # Check if the summary is not already generated
            if pd.isna(df_vic.at[index, "Vicuna-similar-BM25-top2"]):
                response2 = execute_prompt(prompt_top2_citance_embed_bm25)
                print(F"Execute Prompt {time.time() - start}")
                df_vic.at[index, "Vicuna-similar-BM25-top2"] = response2

            # Save the generated summary to a CSV file
            df_vic.to_csv(summary_path, index=False)