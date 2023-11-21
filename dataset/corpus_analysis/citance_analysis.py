from tqdm import tqdm
import os
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import time
import json
import joblib

# Define a function to remove specified strings from text
def remove_strings(text, strings):
    for s in strings:
        text = text.replace(s, '')
    return text

# Load pre-existing DataFrame from a joblib file
start = time.time()
df_all_pre = joblib.load("demo_data/df_all.joblib")
print(f"read in data {time.time() - start}")

# Apply a function to create a new column 'citance_clean' and 'text_length'
df_all_pre["citance_clean"] = df_all_pre.apply(lambda x: remove_strings(x['citance'], [x['reference']]), axis=1)
df_all_pre['text_length'] = df_all_pre['citance_clean'].str.split().apply(len)
entries_all = len(df_all_pre)

# Filter the DataFrame based on text length
df_all = df_all_pre[df_all_pre['text_length'] <= 200]

# Sample and write examples to a file
sample_size = 5
min_length = 40
max_length = 200
with open("sampled_examples.txt", 'w') as file:
    for length in range(min_length, max_length + 1, 10):
        test = df_all_pre[df_all_pre['text_length'] >= length]
        sample = test.sample(n=sample_size)

        file.write(f"Sample for length {length}:\n")
        for x in sample["citance_clean"]:
            file.write(x + "\n")
            file.write("-----------------\n")

entries_filtered = len(df_all)

# Print statistics about the DataFrame
print(f"All Entries: {entries_all}")
print(f"Filtered Entries: {entries_filtered}")

# Calculate and print additional statistics about text lengths
shortest_length = df_all['text_length'].min()
longest_length = df_all['text_length'].max()
mean_length = df_all['text_length'].mean()
median_length = df_all['text_length'].median()
shortest_text = df_all.loc[df_all['citance_clean'].str.len().gt(0), 'citance_clean'].min()
row_with_max_text_length = df_all[df_all['text_length'] == longest_length]
longest_text = row_with_max_text_length["citance_clean"].iloc[0]

print("Shortest length:", len(shortest_text.split()))
print("Longest length:", longest_length)
print("Mean length:", mean_length)
print("Median length:", median_length)
print("Shortest Text:", shortest_text)
print("Longest Text:", longest_text)

# Additional statistics on the dataset
total_amount_citances = len(df_all)
print(f"Total amount of citances: {total_amount_citances}")
total_amount_paper = len(df_all["citing_paper_id"].unique())
print(f"Total amount of unique paper {total_amount_paper}")
mean_citances_paper = total_amount_citances / total_amount_paper
print(f"Mean Citances per Paper {mean_citances_paper}")

# Group by citing and referenced papers to find multiple references
result = df_all.groupby(['citing_paper_id', 'reference_paper_link']).size().reset_index(name='count')
result = result[result['count'] > 1]
count_multiple_reference_all = len(result)
print(f"Multiple references {count_multiple_reference_all}")
count_multiple_reference_unique = len(result["citing_paper_id"].unique())
print(f"Multiple references unique {count_multiple_reference_unique}")
print("------------")

# Filter the data based on length
filtered_data = df_all[df_all['text_length'] <= 200]

# Create visualizations
length_counts = filtered_data['text_length'].value_counts().sort_index()
bin_size = 10
num_bins = int((length_counts.index.max() - length_counts.index.min()) / bin_size) + 1
bins = [length_counts.index.min() + i * bin_size for i in range(num_bins + 1)]

plt.figure(figsize=(8, 6))
sns.kdeplot(filtered_data['text_length'], color='red', label='Density')
plt.xlabel('Citance Length')
plt.ylabel('Density')
plt.title('Density Plot of Citance Length (Length <= 200)')
plt.tight_layout()
plt.savefig("density_plot.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(filtered_data['text_length'], bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Density')
sns.kdeplot(filtered_data['text_length'], color='red', linestyle='--', label='Density Curve')
plt.xlabel('Citance Length')
plt.ylabel('Density')
plt.title('Distribution of Citance length')
plt.xticks(range(0, bins[-1] + 1, 10))
plt.axvline(median_length, color='green', linestyle='--', label='Median')
plt.tight_layout()
plt.savefig("histogram_density.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(filtered_data['text_length'], bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Citance Length')
plt.ylabel('Frequency')
plt.title('Citance Length with Frequency (Length <= 200)')
plt.xticks(range(0, bins[-1] + 1, 10))
plt.tight_layout()
plt.savefig("histogram_frequency.png")
plt.close()

