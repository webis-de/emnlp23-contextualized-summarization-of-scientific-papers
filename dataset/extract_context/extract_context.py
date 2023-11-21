import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
import time
import statistics
import os
import json
import joblib
from tqdm import tqdm
import sys

# Function to apply CLS pooling on BERT-like model output
def cls_pooling(model_output, attention_mask):
    return model_output[0][:, 0]

# Tokenizer for BM25
def bm25_tokenizer(text):
    stop_words_set = set(_stop_words.ENGLISH_STOP_WORDS)
    punct_table = str.maketrans("", "", string.punctuation)
    tokenized_doc = [token.translate(punct_table) for token in text.lower().split() if token.translate(punct_table) not in stop_words_set and len(token.translate(punct_table)) > 0]
    return tokenized_doc

# Encode the corpus using a BERT-like model
def encode_corpus(corpus):
    encoded_input = scibert_tokenizer(corpus, truncation=True, return_tensors='pt', max_length=64, padding='max_length').to(device)
    with torch.no_grad():
        model_output = scibert_model(**encoded_input)
    corpus_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])
    corpus_embeddings = corpus_embeddings.to('cuda')
    return corpus_embeddings

# Batched version of encode_corpus
def encode_corpus2(corpus, batch_size=100):
    num_samples = len(corpus)
    num_batches = (num_samples + batch_size - 1) // batch_size

    corpus_embeddings = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_corpus = corpus[start_idx:end_idx]

        encoded_input = scibert_tokenizer(batch_corpus, truncation=True, return_tensors='pt', max_length=64, padding='max_length').to(device)

        with torch.no_grad():
            model_output = scibert_model(**encoded_input)

        batch_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])
        corpus_embeddings.append(batch_embeddings)

    corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
    corpus_embeddings = corpus_embeddings.to('cuda')

    return corpus_embeddings

# Encode a query using a BERT-like model
def encode_query(query):
    encoded_input = scibert_tokenizer(query, truncation=True, return_tensors='pt', max_length=64, padding='max_length').to(device)

    with torch.no_grad():
        model_output = scibert_model(**encoded_input)
    
    query_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])
    query_embeddings = query_embeddings.to('cuda')
    
    return query_embeddings

# Extract keywords from a query using KeyBERT
def keyword_extraction(query):
    keywords = []
    keywords2 = []
    
    try:
        keywords = kw_model.extract_keywords(query, vectorizer=KeyphraseCountVectorizer(), use_mmr=True)
    except ValueError as e:
        print("Error extracting keywords:", repr(e))
    
    try:
        keywords2 = kw_model2.extract_keywords(query, vectorizer=KeyphraseCountVectorizer(), use_mmr=True)
    except ValueError as e:
        print("Error extracting keywords2:", repr(e))

    for tup in keywords2:
        found = False
        if tup[1] > 0.0:
            for new_tup in keywords:
                if tup[0] == new_tup[0] or tup[0] in new_tup[0]:
                    pos = keywords.index((new_tup[0], new_tup[1]))
                    keywords[pos] = (new_tup[0], max(new_tup[1], tup[1]))
                    found = True
                    break
            if not found:
                keywords.append(tup)

    scores_list = [x[1] for x in keywords if x[1] > 0.0]
    if scores_list != []:
        median = statistics.median(scores_list)
        keyword_list = [x for x in keywords if x[1] > median]
    else:
        keyword_list = []
    
    return keyword_list

# Extract context using BM25
def extract_context_bm25(bm25, cleaned_query, corpus_text):
    bm25_scores = bm25.get_scores(bm25_tokenizer(cleaned_query))
    count_sent_fin = len(bm25_scores) if count_sent > len(bm25_scores) else count_sent
    top_n = np.argpartition(bm25_scores, -count_sent_fin)[-count_sent_fin:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    bm25_sent_list = [corpus_text[hit['corpus_id']] for hit in bm25_hits]
    return " ".join(bm25_sent_list)

# Extract context using Sentence-BERT
def extract_context_sbert(cleaned_query, corpus_embeddings, corpus_text):
    query_embeddings = encode_query(cleaned_query)
    count_sent_fin = len(corpus_embeddings) if count_sent > len(corpus_embeddings) else count_sent
    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=count_sent_fin)
    hits = hits[0]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    sbert_sent_list = [corpus_text[hit['corpus_id']] for hit in hits]
    return " ".join(sbert_sent_list)

# Extract context using BM25 with keyword weighting
def extract_context_bm25_keyword(bm25, keyword_queries, corpus_embeddings, corpus_text):
    bm25_scores_list = []
    bm25_sent_list = []

    for query in keyword_queries:
        query_text = query[0]
        query_weight = query[1]

        bm25_scores = bm25.get_scores(bm25_tokenizer(query_text))
        top_n = np.argsort(bm25_scores)[-len(corpus_embeddings):][::-1]

        bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]

        bm25_scores_list.extend(query_weight * hit['score'] for hit in bm25_hits)
        bm25_sent_list.extend(corpus_text[hit['corpus_id']] for hit in bm25_hits)

    count_sent_fin = len(corpus_embeddings) if count_sent > len(corpus_embeddings) else count_sent
    bm25_df = pd.DataFrame({"score": bm25_scores_list, "text": bm25_sent_list})
    bm25_df = bm25_df.groupby('text').agg({'score': 'mean'})
    bm25_df = bm25_df.reset_index()
    bm25_df = bm25_df.sort_values(by="score", ascending=False)
    bm25_df = bm25_df.head(count_sent_fin)
    
    return " ".join(bm25_df["text"])

# Main function to extract context for each citation
def extract_context(row):
    global previous_link

    link = row["link"]
    title = row["title"]
    cleaned_query = row["cleaned_query"]
    embed_citance = cleaned_query + " " + " ".join(row["prev_sentence"]) + " " + " ".join(row["next_sentence"])
    
    # Extract keywords from the citance
    embed_keyword_queries = keyword_extraction(embed_citance)
    
    if link != previous_link:
        try:
            # Load the relevant corpus for the current link
            corpus = new_df[new_df["docno"] == int(link)]

            if corpus.empty:
                corpus = new_df[new_df["title"] == title]
                if corpus.empty:
                    return {}
                    
            # Remove 'conclusion' and 'abstract' sections
            result = corpus[~corpus['section'].str.contains("conclusion", case=False)]
            result = result[~result['section'].str.contains("abstract", case=False)]
            
            if result.empty:
                return {}
                
            corpus_text = list(result["text"])
            
            if not corpus_text or corpus_text == ['']:
                return {}
            
            # Tokenize the corpus for BM25
            tokenized_corpus = [bm25_tokenizer(passage) for passage in corpus_text]
            
            # Initialize BM25 model
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Encode the corpus using BERT-like model
            corpus_embeddings = encode_corpus2(corpus_text)
    
        except Exception as e:
            print("Error processing corpus:", repr(e))
            return {}
    
    previous_link = link
    
    # Extract context using BM25
    citance_embed_bm25_context = extract_context_bm25(bm25, embed_citance, corpus_text)
    
    # Extract context using Sentence-BERT
    citance_single_sbert_context = extract_context_sbert(cleaned_query, corpus_embeddings, corpus_text)
    
    # Extract context using BM25 with keyword weighting
    if embed_keyword_queries != []:
        citance_embed_bm25_context_keyword = extract_context_bm25_keyword(bm25, embed_keyword_queries, corpus_embeddings, corpus_text)
    else:
        citance_embed_bm25_context_keyword = "No keywords"

    output = {
        "citance_No": row["citance_No"],
        "citing_paper_id": row["citing_paper_id"],
        'citance_embed_bm25': citance_embed_bm25_context,
        'citance_embed_bm25_keywords': citance_embed_bm25_context_keyword,
        'citance_single_sbert': citance_single_sbert_context,
    }

    return output

# Function to load the cleaned dataframe from cache or disk
def load_new_df():
    try:
        # Try to load from cache
        new_df = joblib.load("full_dataset_clean_cache.joblib")
        print("Cache loaded")
    except:
        # Load from disk if cache doesn't exist or is invalid
        chunk = pd.read_csv('full_dataset_clean.csv', chunksize=1000000, na_filter=False, dtype=object)
        new_df = pd.concat(chunk)
        # Cache the loaded dataframe
        joblib.dump(new_df, "full_dataset_clean_cache.joblib")
        print("Cache not available, loaded from disk")
    return new_df

# Function to load the split cleaned dataframe from cache or disk
def load_new_df_split():
    try:
        # Try to load from cache
        new_df = joblib.load("full_dataset_split_clean_cache.joblib")
        print("Cache loaded")
    except:
        # Load from disk if cache doesn't exist or is invalid
        chunk = pd.read_csv('full_dataset_split_clean.csv', chunksize=1000000, na_filter=False, dtype=object)
        new_df = pd.concat(chunk)
        # Cache the loaded dataframe
        joblib.dump(new_df, "full_dataset_split_clean_cache.joblib")
        print("Cache not available, loaded from disk")
    return new_df

if __name__ == "__main__":
    start = time.time()
    typ = sys.argv[1]
    begin = int(sys.argv[2])
    end = int(sys.argv[3])
    print(typ)

    if typ == "top_2":
        new_df = load_new_df()
        count_sent = 2
        mode = "top_2_para"
    elif typ == "top_5":
        new_df = load_new_df_split()
        count_sent = 5
        mode = "top_5_sent"

    subdir = 'restructured_dataset_final'
    sorted_list = [str(num) for num in range(begin, end)]
    removed_entries_file = f"removed_entries_{mode}.json"

    for dir in tqdm(sorted_list, leave=False):
        for filename in os.listdir(os.path.join(subdir, dir)):
            file_name = f'context_{mode}_{filename}.json'

            if os.path.isfile(os.path.join(subdir, dir, filename, file_name)):
                if not os.path.isfile(os.path.join(subdir, dir, filename, file_name)):
                    citing_sentences_file = os.path.join(subdir, dir, filename, f'citing_sentences_{filename}.json')
                    context_file = os.path.join(subdir, dir, filename, file_name)

                    df = pd.read_json(citing_sentences_file, orient="records", dtype=object)
                    output_list = list(df.apply(extract_context, axis=1))

                    filtered_citing_sentences = []
                    filtered_output_list = []
                    removed_entries = []

                    for entry, output in zip(df.iterrows(), output_list):
                        if output:
                            filtered_citing_sentences.append(entry[1])
                            filtered_output_list.append(output)
                        else:
                            removed_entries.append(entry[1].to_dict())

                    if filtered_citing_sentences:
                        with open(context_file, 'w') as fw:
                            json.dump(filtered_output_list, fw, indent=1)

                        filtered_df = pd.DataFrame(filtered_citing_sentences)
                        filtered_df.to_json(citing_sentences_file, orient="records", indent=1)
                    else:
                        os.remove(citing_sentences_file)

                    if removed_entries:
                        with open(removed_entries_file, 'a') as fw:
                            json.dump(removed_entries, fw, indent=1)
                            fw.write('\n')

        print(f"Time for iteration {time.time() - start}")

