import os
import pandas as pd
import json
from tqdm import tqdm
import re
import spacy
import time
from pandarallel import pandarallel

# Load the spaCy model
nlp = spacy.load("en_core_sci_lg")

# Function to serialize a dictionary to JSON
def serialize_dict_abstract(row):
    return json.dumps(row)

# Function to serialize a dictionary to JSON
def serialize_dict_text(row):
    return json.dumps(row)

# Function to convert integers to strings
def cast_int_to_string(row):
    return str(row)

# Generator function to yield rows from a DataFrame
def gen_rows(df):
    for row in df.itertuples(index=False):
        yield row._asdict()

# Function to tokenize text into sentences using spaCy
def sentence_tokenize(text):
    # Tokenize the text using the spaCy pipeline
    doc = nlp(text)
    # Extract the sentences from the doc object
    sentences = [sent.text for sent in doc.sents]
    return sentences

# Function to extract information from the text and create a DataFrame with split sections
def extract_info_to_df_split(row):
    all_list = []
    docno = row["docno"]
    title = row["title"]
    text = row["text2"]
    try:
        text_selected = [re.sub('"', '', m.group(1)).split(', text: ') for m in re.finditer('"section": (.+?)", "cite_spans":', text)]
    except AttributeError:
        found = ''  # apply your error handling

    all_list = [[sents[0], str(sent).strip()] for sents in text_selected for sent in nlp(sents[1]).sents]

    df = pd.DataFrame(all_list, columns=['section', 'text'])
    df["sentid"] = df.index + 1
    df["docno"] = docno
    df["title"] = title

    return df

# Function to extract information from the text and create a DataFrame
def extract_info_to_df(row):
    all_list = []
    docno = row["docno"]
    title = row["title"]
    text = row["text2"]
    try:
        text_selected = [re.sub('"', '', m.group(1)).split(', text: ') for m in re.finditer('"section": (.+?)", "cite_spans":', text)]
    except AttributeError:
        print("Error")

    for i, lst in enumerate(text_selected):
        if len(lst) > 2:
            merged = ' '.join(lst[-2:])  # merge the last two elements
            text_selected[i] = [lst[0]] + [merged]

    df = pd.DataFrame(text_selected, columns=['section', 'text'])
    df["sentid"] = df.index + 1
    df["docno"] = docno
    df["title"] = title

    return df

if __name__ == "__main__":
    # Initialize parallel processing
    pandarallel.initialize(progress_bar=True)

    # Record start time
    start = time.time()

    # Specify the input directory containing S2ORC dataset files
    input_directory = "cs_dataset"

    # Iterate through each file in the input directory
    for i in tqdm(range(0, len(os.listdir(input_directory)))):
        # Read the dataset file in chunks
        chunk = pd.read_json(os.path.join(input_directory, f"cs_dataset_part{i}.json"), chunksize=1000, lines=True)
        all_df = pd.concat(chunk)

        # Select relevant columns and rename them
        all_df = all_df[["paper_id", "title", "abstract_x", "body_text"]]
        all_df = all_df.rename(columns={"paper_id": "docno"})

        # Filter out empty body_text entries
        all_df = all_df[all_df["body_text"].str.len() != 0]

        # Convert columns to strings
        all_df["docno"] = all_df["docno"].apply(str)
        all_df["title"] = all_df["title"].apply(str)

        # Serialize abstract and text columns
        all_df["text"] = all_df["body_text"].apply(serialize_dict_text)
        all_df["abstract"] = all_df["abstract_x"].apply(serialize_dict_abstract)
        all_df["text2"] = all_df[["abstract", "text"]].agg("".join, axis=1)

        # Create split DataFrame
        split_df = pd.DataFrame()
        split_df = pd.concat(all_df.parallel_apply(extract_info_to_df_split, axis=1).tolist(), ignore_index=True)
        split_df = split_df.reindex(columns=["docno", "title", "sentid", "section", "text"])
        split_df["sentid"] = split_df["sentid"].apply(str)

        # Save split DataFrame to CSV
        output_path = os.path.join("final", "split", f"full_split{i}.csv")
        split_df.to_csv(output_path)

        # Print progress
        print(f"Split dataset {time.time() - start}")

