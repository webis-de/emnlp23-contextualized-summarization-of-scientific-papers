import json
import os
import pandas as pd
from tqdm import tqdm

# Step 1: List files in the dataset directory
dataset_directory = "cs_dataset"
count_files = os.listdir(dataset_directory)

# Step 2: Read paper IDs and titles from an output JSON file
output_json_path = "output.json"
paper_id_title = pd.read_json(output_json_path, lines=True, dtype=str)
paper_id_list = set(paper_id_title["paper_id"])
paper_title_list = set(paper_id_title["title"])

# Step 3: Process each file in the dataset
for i in tqdm(range(len(count_files))):
    with open(f"dataset_embed_{i}.json") as f:
        lines = f.readlines()

    # Create output directory
    output_directory = f"restructured_dataset_final/{i}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each line in the file
    for line in lines:
        data = json.loads(line)
        output = []
        directory = f"{output_directory}/{data['metadata']['paper_id']}/"
        
        if not os.path.exists(directory):
            os.makedirs(directory)

            # Extract metadata
            paper_id = data["metadata"]["paper_id"]
            title = data["metadata"]["title"]
            authors = data["metadata"]["authors"]

            # Extract citation information
            citations = data.get("citation", []) + data.get("citation_abstract", [])

            for j, citation in enumerate(citations):
                # Extract citation details
                reference = citation.get("reference", "")
                section = citation.get("section", "")
                citance = citation.get("citance", "")
                prev_sentence = citation.get("prev_sentence", [])
                next_sentence = citation.get("next_sentence", [])
                context = citation.get("context", [])
                summary = citation.get("summary", [])

                # Extract bib entry information
                bib_entry = citation.get("bib_entry", {})
                bib_title = bib_entry.get("title", "")
                bib_authors = bib_entry.get("authors", [])
                bib_year = bib_entry.get("year", "")
                bib_venue = bib_entry.get("venue", "")
                bib_link = bib_entry.get("link", "")

                names = ', '.join([' '.join(filter(bool, [name['first']] + name['middle'] + [name['last']])) + (', ' + name['suffix'] if name['suffix'] else '') for name in bib_authors])

                # Build the output dictionary
                output_dict = {
                    "citance_No": j + 1,
                    "citing_paper_id": paper_id,
                    "title": title,
                    "citing_paper_authors": names,
                    "reference": reference,
                    "citance_section": section,
                    "citance": citance,
                    "prev_sentence": prev_sentence,
                    "next_sentence": next_sentence,
                    "reference_paper_title": bib_title,
                    "reference_paper_link": bib_link,
                }

                output.append(output_dict)

            # Save the output to a JSON file
            file_name = f'{directory}/citing_sentences_{paper_id}.json'
            with open(file_name, 'w') as fw:
                json.dump(output, fw, indent=1)
