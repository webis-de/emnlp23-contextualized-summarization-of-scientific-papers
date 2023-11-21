import pandas as pd


df = pd.read_csv(f"/work/users/ay635xema/demo_data/eval_files/final_eval/scores.csv")

# Filter and calculate average for alpaca
alpaca_df_summary_top2 = df[df['model'].str.startswith('alpaca_summary_top2')]
alpaca_average_values_summary_top2 = alpaca_df_summary_top2[['BertScore', 'ROUGE-1', 'ROUGE-2', 'ROUGE-l']].mean()

alpaca_df_alpaca_summary_top5 = df[df['model'].str.startswith('alpaca_summary_top5')]
alpaca_average_values_summary_top5 = alpaca_df_alpaca_summary_top5[['BertScore', 'ROUGE-1', 'ROUGE-2', 'ROUGE-l']].mean()

# Filter and calculate average for falcon
falcon_df_summary_top2 = df[df['model'].str.startswith('falcon_summary_top2')]
falcon_average_values_summary_top2 = falcon_df_summary_top2[['BertScore', 'ROUGE-1', 'ROUGE-2', 'ROUGE-l']].mean()

# Filter and calculate average for falcon
falcon_df_summary_top5 = df[df['model'].str.startswith('falcon_summary_top5')]
falcon_average_values_summary_top5 = falcon_df_summary_top5[['BertScore', 'ROUGE-1', 'ROUGE-2', 'ROUGE-l']].mean()

# Filter and calculate average for falcon
llama_df_summary_top2 = df[df['model'].str.startswith('llama_summary_top2')]
llama_average_values_summary_top2 = llama_df_summary_top2[['BertScore', 'ROUGE-1', 'ROUGE-2', 'ROUGE-l']].mean()

# Filter and calculate average for falcon
llama_df_summary_top5 = df[df['model'].str.startswith('llama_summary_top5')]
llama_average_values_summary_top5 = llama_df_summary_top5[['BertScore', 'ROUGE-1', 'ROUGE-2', 'ROUGE-l']].mean()

# Filter and calculate average for falcon
vicuna_df_summary_top2 = df[df['model'].str.startswith('vicuna_summary_top2')]
vicuna_average_values_summary_top2 = vicuna_df_summary_top2[['BertScore', 'ROUGE-1', 'ROUGE-2', 'ROUGE-l']].mean()

# Filter and calculate average for falcon
vicuna_df_summary_top5 = df[df['model'].str.startswith('vicuna_summary_top5')]
vicuna_average_values_summary_top5 = vicuna_df_summary_top5[['BertScore', 'ROUGE-1', 'ROUGE-2', 'ROUGE-l']].mean()

print("Average values for alpaca_summary_top2:")
print(alpaca_average_values_summary_top2)
print("\nAverage values for falcon_summary_top2:")
print(falcon_average_values_summary_top2)
print("\nAverage values for llama_summary_top2:")
print(llama_average_values_summary_top2)
print("\nAverage values for vicuna_summary_top2:")
print(vicuna_average_values_summary_top2)

print("-----------------------------------------------------------")

print("Average values for alpaca_summary_top5:")
print(alpaca_average_values_summary_top5)
print("\nAverage values for falcon_summary_top5:")
print(falcon_average_values_summary_top5)
print("\nAverage values for llama_summary_top5:")
print(llama_average_values_summary_top5)
print("\nAverage values for vicuna_summary_top5:")
print(vicuna_average_values_summary_top5)

