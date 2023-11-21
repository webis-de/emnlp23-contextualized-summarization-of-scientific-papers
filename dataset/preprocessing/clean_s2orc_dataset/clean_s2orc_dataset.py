from itertools import count
import pandas as pd
import os
from collections import Counter
import numpy as np
import time
import json

# Function to convert a DataFrame to a JSON file
def dict_to_json(dataset, filename: str) -> None:
    """
    Converts a DataFrame to a JSON file.

    Args:
        dataset (pd.DataFrame): The DataFrame to be converted.
        filename (str): The name of the output JSON file.

    Returns:
        None
    """
    f = open(filename, "w")
    for row in dataset.iterrows():
        row[1].to_json(f)
        f.write("\n")
    f.close() 

# Function to split JSON data into equal parts
def split_json(data, num_parts):
    """
    Splits a JSON dataset into a specified number of equal parts.

    Args:
        data: The JSON data to be split.
        num_parts (int): The number of parts to split the data into.

    Returns:
        list: A list containing the split parts of the JSON data.
    """
    parts = []
    for i in range(num_parts):
        parts.append(data[i*len(data)//num_parts:(i+1)*len(data)//num_parts])
    return parts

if __name__=="__main__":
    # List of files in the metadata directory
    count_files = os.listdir("relative-path/metadata/computer-science")

    for i in range(0,len(count_files)):    
        all_df = pd.DataFrame()

        # Read metadata JSON file
        with open(F'relative-path/metadata/computer-science/metadata_{i}.jsonl.gz') as f:
            metadata = pd.read_json(f, lines=True)

        # Read PDF parses JSON file
        with open(F'relative-path/pdf_parses/computer-science/pdf_parses_{i}.jsonl.gz') as f:
            pdf_parses = pd.read_json(f, lines=True)

        # Merge metadata and PDF parses data
        all_df = all_df.append(pdf_parses.merge(metadata, on="paper_id", how="left"))

        # Select relevant columns for further processing
        con_dataset = all_df[["paper_id", "title", "abstract_x", "body_text", "bib_entries", "authors", "has_pdf_parsed_body_text", "mag_field_of_study",  "abstract_y", "year", "has_outbound_citations", "outbound_citations", "s2_url"]]

        # Read paper_id and citations data
        paper_id_title = pd.read_json(F"relative-path/output.json", lines=True, dtype=str)
        paper_id_list = set(paper_id_title["paper_id"])
        df = pd.read_json(F"relative-path/output_citations2.json", lines=True)

        # Filter papers with parsed body text
        filtered_data = con_dataset[con_dataset["has_pdf_parsed_body_text"] == 1.0]

        # Filter papers with only Computer Science as the field of study
        only_cs_papers = filtered_data[filtered_data['mag_field_of_study'].str.len() == 1]
        only_cs_papers = only_cs_papers[only_cs_papers["has_outbound_citations"] == True]
        only_cs_papers['matching_ids'] = only_cs_papers['outbound_citations'].map(lambda x: list(filter(lambda i: i in paper_id_list, x)))
        only_cs_papers[only_cs_papers["matching_ids"].str.len() != 0]

        print(len(filtered_data))
        print(len(only_cs_papers))
        
        # Save the processed data to JSON files
        dict_to_json(only_cs_papers, F"relative-path/cs_dataset/cs_dataset_part{i}.json")
