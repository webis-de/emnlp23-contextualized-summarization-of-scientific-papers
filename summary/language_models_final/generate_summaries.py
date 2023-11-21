import pandas as pd
from sys import argv
import time
import json
from pathlib import Path
import requests
from tqdm import tqdm
import numpy as np

#df = pd.read_json('/work/users/ay635xema/demo_data/citance/12122749/context_top_2_para_12122749.json')

#l1 =  [df.iloc[0]['citance_embed_bm25'], df.iloc[0]['citance_embed_bm25_keywords'], df.iloc[0]['citance_single_sbert'], df.iloc[0]['citance_single_bm25']]

#df2 = pd.read_json('/work/users/ay635xema/demo_data/citance/12122749/context_top_5_sent_12122749.json')

#l2 =  [df2.iloc[0]['citance_embed_bm25'], df2.iloc[0]['citance_embed_bm25_keywords'], df2.iloc[0]['citance_single_sbert'], df2.iloc[0]['citance_single_bm25']]

llm_model = "falcon"

eval_df = pd.read_csv(f"/work/users/ay635xema/demo_data/eval_files/eval_{llm_model}.csv")

print("test")
#33 Summarize the following scientific text in not more than 5 sentences.
#32 Generate a short summary of the following scientific text. The summary should not be more than 5 sentences long.
#31 Generate a summary for the following scientific text in not more than 5 sentences.
#3 Generate a coherent summary for the following scientific text in not more than 5 sentences.  
# prompt3 = """Generate a coherent summary for the following scientific text in not more than 5 sentences.  

# scientific text: {e}

# summary: """ 


# prompt4 = """Generate a coherent paraphrased text for the following scientific text.

# scientific text: {e}

# paraphrased text: """


# prompt5 = """Generate a short and coherent paraphrased text for the following scientific text.

# scientific text: {e}

# paraphrased text: """

# prompt6 = """Paraphrase the following scientific text.

# scientific text: {e}

# paraphrased text: """

# ###############

# prompt7 = """A chat between a curious user and an artifical intelligence assistant. The assistant knows how to paraphrase scientific text and the user will provide the scientific text for the assistant to paraphrase.

# USER: 
# Generate a coherent paraphrased text for the following scientific text:  "{e}"

# ASSISTANT: """ 

# prompt8 = """A chat between a curious user and an artifical intelligence assistant. The assistant knows how to summarize scientific text and the user will provide the scientific text for the assistant to summarize.

# USER: 
# Generate a coherent summary for the following scientific text in not more than 5 sentences:  "{e}"

# ASSISTANT: """ 


prompt9 = """
### Instruction:
A chat between a curious human and an artifical intelligence assistant. The assistant knows how to summarize scientific text and the user will provide the scientific text for the assistant to summarize.

### Input: 
Generate a coherent summary for the following scientific text in not more than 5 sentences:  "{e}"

### Output: """
	
prompt10 = """
### Instruction:
A chat between a curious human and an artifical intelligence assistant. The assistant knows how to paraphrase scientific text and the user will provide the scientific text for the assistant to paraphrase.

### Input: 
Generate a coherent paraphrased text for the following scientific text:  "{e}"

### Output: """



# l= ["""The Penn Chinese Treebank (CTB) is an ongoing project, with its objective being to create a segmented Chinese corpus annotated with POS tags and syntactic brackets. In this paper, we will address three issues in the development of the Chinese Treebank: annotation speed, annotation accuracy and usability of the corpus. One might also want to extract information from sentences with pronouns. Among other things, there are at least two areas in which the Chinese treebank can be enhanced, that is, more finegrained predicate-argument structure annotation and coreference annotation. However, the subject and object in the Chinese Treebank are defined primarily in structural terms.""",
# """It is because most randomly generated queries have relatively small result sizes. 2) As graph gets dense, more intermediate results will be generated for randomly generated queries. As discussed above, for randomly generated queries, the query time is nearly linear with the number of STwigs and joins. A spanning tree is generated on the generated query to guarantee it is a connected graph. One query set is generated using DFS traversal from a randomly chosen node.""", 
# """Figure 2 shows a set of top-ranked discriminative patch clusters discovered with our approach. To perform classification, top 210 discovered patches of each scene are aggregated into a spatial pyramid using maxpooling over the discriminative patch scores as in [3]. For each cluster, we asked human labelers to mark which of the cluster's top ten firings on the validation set are visually consistent with the cluster. Therefore, we performed an informal perceptual experiment with human subjects, measuring the visual purity of our clusters.Initialization: The input to our discovery algorithm is a \discovery dataset\ D of unlabeled images as well as a much larger \natural world dataset\ N (in this paper we used 6,000 images randomly sampled from Flickr.com).""",
# """So far, we have shown the discovered visual elements for a given city as an ordered list of patch clusters (Figure 4 ). Once we have detectors that set up the correspondence between different cities such as Paris and Prague (Sec. 5.3), we can use them for geographically-informed image retrieval. Another interesting observation is that some discovered visual el- Figure 13: Geographically-informed retrieval. Given a set of architectural elements (windows, balconies, etc.) discovered for a particular city, it is natural to ask what these same elements might look like in other cities. We first compute the nearest neighbors of each candidate, and reject candidates with too many neighbors in the negative set. Then we gradually build clusters by applying iterative discriminative learning to each surviving candidate.""",
# """In both the subjectivity classifier and polarity classifier, the same targetindependent feature set is used. In (Hu and Liu, 2004) , opinions are extracted from product reviews, where the features of the product are considered opinion targets. Besides the above mentioned work for targetindependent sentiment classification, there are also several approaches proposed for target-dependent classification, such as (Nasukawa and Yi, 2003; Hu and Liu, 2004; Ding and Liu, 2007) . (Nasukawa and Yi, 2003) adopt a rule based approach, where rules are created by humans for adjectives, verbs, nouns, and so on. As Twitter becomes more popular, sentiment analysis on Twitter data becomes more attractive. (Go et al., 2009; Parikh and Movassate, 2009; Barbosa and Feng, 2010; Davidiv et al., 2010) all follow the machine learning based approach for sentiment classification of tweets. In our approach, rich feature representations are used to distinguish between sentiments expressed towards different targets."""
# ]

##e = "The dialog system can be divided into different modules, such as Natural Language Understanding (Yao et al. 2014; Mesnil et al. 2015) , Dialog State Tracking (Henderson, Thomson, and Young 2014; Williams, Raux, and Henderson 2016) , and Natural Language Generation (Wen et al. 2015) . Dialogs can have complex speaker interactions: at each turn, users play one of three roles (sender, addressee, observer), and those roles vary across turns. In this example, a 2 says u (1) to a 1 , then a 1 says u (2) to a 3 , and finally a 3 says u (3) to a 2 . The task requires modeling multi-party conversations and can be directly used to build retrievalbased dialog systems (Lu and Li 2013; Hu et al. 2014; Ji, Lu, and Li 2014; Wang et al. 2015) . In the Ubuntu Internet Relay Chat channel (IRC), for example, one user can initiate a discussion about an Ubuntu-related technical issue, and many other users can work together to solve the problem."
#e = l2[0]


#TODO: this code needs to be changed so it functions with the different modes
#mode = "top2_citance_embed_bm25"

def execute_prompt(prompt):
    print(prompt)
    args = {"batch": [prompt]}
    request = requests.post("http://localhost:5000", json=args)
    #__import__("ipdb").set_trace()
    results = request.json()
    print(results)
    #print(results["data"][0])
    response = results["data"][0]
    return response

print(llm_model)
for index, row in tqdm(eval_df[24:].iterrows(), total=eval_df[24:].shape[0]):
    prompt_top2_citance_embed_bm25 = f"""
        ### Instruction:
        A chat between a curious human and an artifical intelligence assistant. The assistant knows how to summarize scientific text and the user will provide the scientific text for the assistant to summarize.

        ### Input: 
        Generate a coherent summary for the following scientific text in not more than 5 sentences:  "{row["top2_citance_embed_bm25"]}"

        ### Output: """

    prompt_top2_citance_embed_bm252 = f"""A chat between a curious user and an artifical intelligence assistant. The assistant knows how to summarize scientific text and the user will provide the scientific text for the assistant to summarize.

        USER: 
        Generate a coherent summary for the following scientific text in not more than 5 sentences:  "{row["top2_citance_embed_bm25"]}"

        ASSISTANT: """
    if pd.isna(eval_df.at[index, "summary_top2_citance_embed_bm25"]):

        response1 = execute_prompt(prompt_top2_citance_embed_bm25)
        eval_df.at[index, "summary_top2_citance_embed_bm25"] = response1

    
    prompt_top5_citance_embed_bm25 = f"""
        ### Instruction:
        A chat between a curious human and an artifical intelligence assistant. The assistant knows how to paraphrase scientific text and the user will provide the scientific text for the assistant to paraphrase.

        ### Input: 
        Generate a coherent paraphrased text for the following scientific text:  "{row["top5_citance_embed_bm25"]}"

        ### Output: """
    
    prompt_top5_citance_embed_bm252 = f"""A chat between a curious user and an artifical intelligence assistant. The assistant knows how to paraphrase scientific text and the user will provide the scientific text for the assistant to paraphrase.

        USER: Generate a coherent paraphrased text for the following scientific text:  "{row["top5_citance_embed_bm25"]}"

        ASSISTANT: """
    
    if pd.isna(eval_df.at[index, "summary_top5_citance_embed_bm25"]):
        response2 = execute_prompt(prompt_top5_citance_embed_bm25)
        eval_df.at[index, "summary_top5_citance_embed_bm25"] = response2

    prompt_top2_citance_single_sbert = f"""
        ### Instruction:
        A chat between a curious human and an artifical intelligence assistant. The assistant knows how to summarize scientific text and the user will provide the scientific text for the assistant to summarize.

        ### Input: 
        Generate a coherent summary for the following scientific text in not more than 5 sentences:  "{row["top2_citance_single_sbert"]}"

        ### Output: """
    
    prompt_top2_citance_single_sbert2 = f"""A chat between a curious user and an artifical intelligence assistant. The assistant knows how to summarize scientific text and the user will provide the scientific text for the assistant to summarize.

        USER: 
        Generate a coherent summary for the following scientific text in not more than 5 sentences:  "{row["top2_citance_single_sbert"]}"

        ASSISTANT: """
    
    #print(prompt_top2_citance_single_sbert)
    if pd.isna(eval_df.at[index, "summary_top2_citance_single_sbert"]):
        response3 = execute_prompt(prompt_top2_citance_single_sbert)
        eval_df.at[index, "summary_top2_citance_single_sbert"] = response3
    
    prompt_top5_citance_single_sbert = f"""
        ### Instruction:
        A chat between a curious human and an artifical intelligence assistant. The assistant knows how to paraphrase scientific text and the user will provide the scientific text for the assistant to paraphrase.

        ### Input: 
        Generate a coherent paraphrased text for the following scientific text:  "{row["top5_citance_single_sbert"]}"

        ### Output: """
    
    prompt_top5_citance_single_sbert2 = f"""A chat between a curious user and an artifical intelligence assistant. The assistant knows how to paraphrase scientific text and the user will provide the scientific text for the assistant to paraphrase.

        USER: Generate a coherent paraphrased text for the following scientific text:  "{row["top5_citance_single_sbert"]}"

        ASSISTANT: """

    if pd.isna(eval_df.at[index, "summary_top5_citance_single_sbert"]):
        response4 = execute_prompt(prompt_top5_citance_single_sbert)
        eval_df.at[index, "summary_top5_citance_single_sbert"] = response4


    # start = time.time()
    # #p = prompt.format(e=e)
    # print(p)
    # args = {"batch": [p]}
    # request = requests.post("http://localhost:5000", json=args)
    # #__import__("ipdb").set_trace()
    # results = request.json()
    # print(results)
    # print(results["data"][0])
    # response = results["data"][0]
    # print(F"Time: {time.time() - start}")


    #eval_df.at[index, f'summary_{mode}'] = response

    eval_df.to_csv(f"/work/users/ay635xema/demo_data/eval_files/eval_{llm_model}.csv", index=False)

# l1 = list(eval_df["top2_citance_embed_bm25"][:5])
# l2 = list(eval_df["top5_citance_embed_bm25"][:5])

# for e in l1:
#     #e = e.replace(":", "")
#     start = time.time()
#     p = prompt9.format(e=e)
#     print(p)
#     args = {"batch": [p]}
#     request = requests.post("http://localhost:5000", json=args)
#     #__import__("ipdb").set_trace()
#     results = request.json()
#     print(results)
#     print(results["data"][0])
#     print(F"Time: {time.time() - start}")
#     print("-----")