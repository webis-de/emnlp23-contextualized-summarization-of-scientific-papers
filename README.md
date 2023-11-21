# Citance-Contextualized Summarization of Scientific Papers
This repository contains the code and data for the paper “Citance-Contextualized Summarization of Scientific Papers” by Shahbaz Syed, Ahmad Dawar Hakimi, Khalid Al-Khatib, and Martin Potthast, published at EMNLP 2023.

## Abstract
We propose a new approach to summarizing scientific papers that takes into account the citation context of the paper. Given a sentence containing a citation of a paper (a citance), our approach generates an informative summary of the cited paper that is relevant to the citance. To do so, we use multiple types of citance contexts as queries to retrieve content from the cited paper, and then use large language models to generate abstractive summaries. We evaluate our approach using WEBIS-CONTEXT-SCISUMM-2023, a new dataset containing 540K computer science papers and 4.6M citances. Our experiments show that our contextualized summaries are more informative and relevant than generic abstracts, especially when the citance is ambiguous or does not align with the main contribution of the cited paper.


