import torch
from sentence_transformers import SentenceTransformer, util
import spacy
from functools import partial
from pandarallel import pandarallel
import os
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModel
import nltk
#from datasets import load_dataset
import nltk.data
nlp = spacy.load("en_core_sci_lg")



def dict_to_json(dataset, filename: str) -> None:
    """
    Write a dataset to a JSON file.

    Parameters:
    - dataset (pandas.core.frame.DataFrame): Dataset to be written.
    - filename (str): Output JSON file path.
    """
    with open(filename, "w") as f:
        for row in dataset.iterrows():
            row[1].to_json(f)
            f.write("\n")

def sentence_tokenize(text):
    """
    Tokenize a text into sentences using the spaCy pipeline.

    Parameters:
    - text (str): Input text.

    Returns:
    - list: List of sentences.
    """
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def extract_citation_abstract_embed(row, paper_id_list, paper_title_list, scibert_model, tokenizer):
    """
    Extract citation information from the abstract of a paper.

    Parameters:
    - row (pandas.core.series.Series): Information of a paper.
    - paper_id_list (set): Set of paper IDs.
    - paper_title_list (set): Set of paper titles.
    - scibert_model: Pre-trained SciBERT model.
    - tokenizer: Tokenizer for SciBERT.

    Returns:
    - list: List of citations in the abstract.
    """
    abstract_df = pd.DataFrame(row["abstract_x"])
    abstract_citations = []

    if not abstract_df.empty:
        abstract_df["tok_sent"] = abstract_df['text'].apply(sentence_tokenize)

        df_grouped = abstract_df.groupby('section')
        concat_text = []
        concat_tok_text = []

        for _, group in df_grouped:
            concat_text.append(' '.join(group['text']))
            concat_tok_text.append(group['tok_sent'].sum())

        result_df = pd.DataFrame({'section': df_grouped.groups.keys(), 'text': concat_text, 'tok_sent': concat_tok_text})

        abstract_df = abstract_df[abstract_df["cite_spans"].str.len() != 0]
        section = "$ABSTRACT_STARTING_SECTION$"

        for i in range(len(abstract_df)):
            abstract_text = abstract_df.iloc[i].text
            if abstract_df.iloc[i].section != section:
                t2 = result_df.loc[result_df['section'] == abstract_df.iloc[i].section].tok_sent.iloc[0]
                section = abstract_df.iloc[i].section

                encoded_input = tokenizer(t2, truncation=True, return_tensors='pt', max_length=64, padding='max_length').to(device)
                with torch.no_grad():
                    model_output = scibert_model(**encoded_input)

                sentence_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])
                sentence_embeddings = sentence_embeddings.to('cuda')

            for cite_span in abstract_df.iloc[i].cite_spans:
                try:
                    bib_entry = row.bib_entries[cite_span["ref_id"]]
                    if bib_entry["link"] in set(row["matching_ids"]) or bib_entry["title"] in paper_title_list:
                        abstract_citation_text = {}
                        sent = ""
                        abstract_citation_text.update({"reference": cite_span["text"]})
                        abstract_citation_text.update({"section": abstract_df.iloc[i].section})

                        t1 = abstract_df.iloc[i].tok_sent
                        pos = 0

                        for sent in t1:
                            if cite_span.get("start") >= pos:
                                pos += len(sent)
                                if sent == t1[len(t1) - 1] and cite_span["text"] in t1[len(t1)-1]:
                                    abstract_citation_text.update({"citance": sent})
                                    break
                                elif cite_span.get("start") <= pos:
                                    abstract_citation_text.update({"citance": sent})
                                    break

                        if sent in t2:
                            query_embeddings = sentence_embeddings[t2.index(sent)]
                        else:
                            encoded_input = tokenizer(sent, truncation=True, return_tensors='pt', max_length=64, padding='max_length').to(device)
                            with torch.no_grad():
                                model_output = scibert_model(**encoded_input)

                            query_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])

                        query_embeddings = query_embeddings.to('cuda')

                        hits = util.semantic_search(query_embeddings, sentence_embeddings, top_k=3)
                        top_N_sent = []

                        for hit in hits[0][1:]:
                            top_N_sent.append(t2[hit['corpus_id']])

                        abstract_citation_text.update({"prev_sentence": []})
                        abstract_citation_text.update({"next_sentence": []})

                        if len(top_N_sent) == 0:
                            continue
                        elif len(top_N_sent) == 1:
                            if t2.index(sent) > t2.index(top_N_sent[0]):
                                abstract_citation_text["prev_sentence"].append(top_N_sent[0])
                            else:
                                abstract_citation_text["next_sentence"].append(top_N_sent[0])
                        elif len(top_N_sent) == 2:
                            if t2.index(sent) > t2.index(top_N_sent[0]):
                                abstract_citation_text["prev_sentence"].append(top_N_sent[0])
                            else:
                                abstract_citation_text["next_sentence"].append(top_N_sent[0])

                            if t2.index(sent) > t2.index(top_N_sent[1]):
                                if t2.index(top_N_sent[1]) > t2.index(top_N_sent[0]):
                                    abstract_citation_text["prev_sentence"].append(top_N_sent[1])
                                else:
                                    abstract_citation_text["prev_sentence"].insert(0,top_N_sent[1])
                            elif t2.index(sent) < t2.index(top_N_sent[1]):
                                if t2.index(top_N_sent[1]) < t2.index(top_N_sent[0]):
                                    abstract_citation_text["next_sentence"].insert(0,top_N_sent[1])
                                else:
                                    abstract_citation_text["next_sentence"].append(top_N_sent[1])

                        abstract_citation_text.update({"abstract_context": []})
                        abstract_citation_text.update({"abstract_summary": []})

                        abstract_citation_text.update({"bib_entry": bib_entry})

                        abstract_citations.append(abstract_citation_text)

                except Exception as e:
                    print(repr(e))

    return abstract_citations

def extract_citation_embed(row, paper_id_list, paper_title_list, scibert_model, tokenizer):
    """
    Extract citation information from the body of a paper.

    Parameters:
    - row (pandas.core.series.Series): Information of a paper.
    - paper_id_list (set): Set of paper IDs.
    - paper_title_list (set): Set of paper titles.
    - scibert_model: Pre-trained SciBERT model.
    - tokenizer: Tokenizer for SciBERT.

    Returns:
    - list: List of citations in the body of the text.
    """
    body_text_df = pd.DataFrame(row["body_text"])
    citations = []

    if not body_text_df.empty:
        body_text_df["tok_sent"] = body_text_df['text'].apply(sentence_tokenize)
        df_grouped = body_text_df.groupby('section')

        concat_text = []
        concat_tok_text = []

        for _, group in df_grouped:
            concat_text.append(' '.join(group['text']))
            concat_tok_text.append(group['tok_sent'].sum())

        result_df = pd.DataFrame({'section': df_grouped.groups.keys(), 'text': concat_text, 'tok_sent': concat_tok_text})

        body_text_df = body_text_df[body_text_df["cite_spans"].str.len() != 0]
        section = "$STARTING_SECTION$"

        for i in range(len(body_text_df)):
            text = body_text_df.iloc[i].text

            if body_text_df.iloc[i].section != section:
                t2 = result_df.loc[result_df['section'] == body_text_df.iloc[i].section].tok_sent.iloc[0]
                section = body_text_df.iloc[i].section

                encoded_input = tokenizer(t2, truncation=True, return_tensors='pt', max_length=64, padding='max_length').to(device)
                with torch.no_grad():
                    model_output = scibert_model(**encoded_input)

                sentence_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])
                sentence_embeddings = sentence_embeddings.to('cuda')

            for cite_span in body_text_df.iloc[i].cite_spans:
                try:
                    bib_entry = row.bib_entries[cite_span["ref_id"]]
                    if bib_entry["link"] in set(row["matching_ids"]) or bib_entry["title"] in paper_title_list:
                        citation_text = {}
                        sent = ""
                        citation_text.update({"reference": cite_span["text"]})
                        citation_text.update({"section": body_text_df.iloc[i].section})

                        t1 = body_text_df.iloc[i].tok_sent
                        pos = 0

                        for sent in t1:
                            if cite_span.get("start") >= pos:
                                pos += len(sent)

                                if sent == t1[len(t1) - 1] and cite_span["text"] in t1[len(t1)-1]:
                                    citation_text.update({"citance": sent})
                                    break
                                elif cite_span.get("start") <= pos:
                                    citation_text.update({"citance": sent})
                                    break

                        if sent in t2:
                            query_embeddings = sentence_embeddings[t2.index(sent)]
                        else:
                            encoded_input = tokenizer(sent, truncation=True, return_tensors='pt', max_length=64, padding='max_length').to(device)
                            with torch.no_grad():
                                model_output = scibert_model(**encoded_input)

                            query_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])

                        query_embeddings = query_embeddings.to('cuda')

                        hits = util.semantic_search(query_embeddings, sentence_embeddings, top_k=3)
                        top_N_sent = []

                        for hit in hits[0][1:]:
                            top_N_sent.append(t2[hit['corpus_id']])

                        for top_sent in top_N_sent:
                            if t2.index(sent) > t2.index(top_N_sent[0]):
                                citation_text.update({"prev_sentence": top_sent})
                            else:
                                citation_text.update({"next_sentence": top_sent})

                        citation_text.update({"prev_sentence": []})
                        citation_text.update({"next_sentence": []})

                        if len(top_N_sent) == 0:
                            continue
                        elif len(top_N_sent) == 1:
                            if t2.index(sent) > t2.index(top_N_sent[0]):
                                citation_text["prev_sentence"].append(top_N_sent[0])
                            else:
                                citation_text["next_sentence"].append(top_N_sent[0])
                        elif len(top_N_sent) == 2:
                            if t2.index(sent) > t2.index(top_N_sent[0]):
                                citation_text["prev_sentence"].append(top_N_sent[0])
                            else:
                                citation_text["next_sentence"].append(top_N_sent[0])

                            if t2.index(sent) > t2.index(top_N_sent[1]):
                                if t2.index(top_N_sent[1]) > t2.index(top_N_sent[0]):
                                    citation_text["prev_sentence"].append(top_N_sent[1])
                                else:
                                    citation_text["prev_sentence"].insert(0,top_N_sent[1])
                            elif t2.index(sent) < t2.index(top_N_sent[1]):
                                if t2.index(top_N_sent[1]) < t2.index(top_N_sent[0]):
                                    citation_text["next_sentence"].insert(0,top_N_sent[1])
                                else:
                                    citation_text["next_sentence"].append(top_N_sent[1])

                        citation_text.update({"context": []})
                        citation_text.update({"summary": []})

                        citation_text.update({"bib_entry": row.bib_entries[cite_span["ref_id"]]})
                        citations.append(citation_text)

                except Exception as e:
                    print(repr(e))

    return citations



def cls_pooling(model_output, attention_mask):
    """
    Perform pooling on the output of a BERT-like model.
    
    Parameters:
    - model_output: Output of the BERT-like model.
    - attention_mask: Attention mask for the input tokens.
    
    Returns:
    - torch.Tensor: Pooled representation.
    """
    return model_output[0][:, 0]

def add_metadata(row):
    """
    Extract metadata information from a paper.
    
    Parameters:
    - row (pandas.core.series.Series): Information of a paper.
    
    Returns:
    - dict: Metadata of the paper.
    """
    data_entry = {}
    keys = ["paper_id", "title", "authors"]

    for key in keys:
        data_entry.update({key: row[key]})

    return data_entry

def create_dataset(all_df, paper_id_list, paper_title_list, scibert_model, tokenizer, mode="embed"):
    """
    Create a contextualized summarization dataset.

    Parameters:
    - all_df (pandas.core.frame.DataFrame): DataFrame containing information about multiple papers.
    - paper_id_list (set): Set of paper IDs.
    - paper_title_list (set): Set of paper titles.
    - scibert_model: Pre-trained SciBERT model.
    - tokenizer: Tokenizer for SciBERT.
    - mode (str): Mode for creating the dataset (default is "embed").

    Returns:
    - pandas.core.frame.DataFrame: DataFrame containing metadata and citation information.
    """
    df = pd.DataFrame(columns=['metadata', 'citation_abstract', "citation"])

    print("Extracting Metadata")
    df['metadata'] = all_df.parallel_apply(add_metadata, axis=1)
    print("STARTING EXTRACT CITATION ABSTRACT")
    df['citation_abstract'] = all_df.apply(partial(extract_citation_abstract_embed, paper_id_list=paper_id_list, paper_title_list=paper_title_list, scibert_model=scibert_model, tokenizer=tokenizer), axis=1)
    print("STARTING EXTRACT CITATION")
    df["citation"] = all_df.apply(partial(extract_citation_embed, paper_id_list=paper_id_list, paper_title_list=paper_title_list, scibert_model=scibert_model, tokenizer=tokenizer), axis=1)

    df_filtered = df[(df["citation_abstract"].str.len() != 0) | (df["citation"].str.len() != 0)]

    return df_filtered

# Entry point
if __name__ == "__main__":
    # Define the tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)
    
    pandarallel.initialize(progress_bar=True, use_memory_fs=False)
    count_files = os.listdir("../cs_dataset")  # Replace with your actual relative path
    paper_id_title = pd.read_json("../output.json", lines=True, dtype=str)  # Replace with your actual relative path
    paper_id_list = set(paper_id_title["paper_id"])
    paper_title_list = set(paper_id_title["title"])
    start = time.time()

    for i in range(0, len(count_files)):
        all_df = pd.read_json(f"../cs_dataset/cs_dataset_part{i}.json", lines=True)  # Replace with your actual relative path
        all_df['matching_ids'] = all_df['outbound_citations'].apply(lambda x: list(set(x) & paper_id_list))

        all_df = all_df[all_df["matching_ids"].str.len() != 0]

        dataset = create_dataset(all_df, paper_id_list=paper_id_list, paper_title_list=paper_title_list, scibert_model=scibert_model, tokenizer=tokenizer, mode=mode)
        dict_to_json(dataset, f'./{mode}_FINAL/dataset_{mode}_{i}.json')  # Replace with your actual relative path