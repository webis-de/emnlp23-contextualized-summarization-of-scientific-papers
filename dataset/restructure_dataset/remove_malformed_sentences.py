import pandas as pd
import re
import joblib

def count_words(text):
    """
    Count the number of words in the given text using a regular expression pattern.

    Parameters:
    - text (str): Input text.

    Returns:
    - int: Number of words in the text.
    """
    words = word_pattern.findall(text)
    return len(words)

if __name__ == "__main__":
    # Compile the word pattern regular expression
    word_pattern = re.compile(r'\b[A-Za-z]+\b')

    # Load the dataset using a relative path
    new_df = joblib.load("final/full_dataset_split_cache.joblib")

    # Define a condition for filtering rows based on word count
    mask = new_df['text'].apply(count_words) >= 5

    # Create a new DataFrame with removed rows
    removed_df = new_df[~mask]

    # Save the removed data to a CSV file with a relative path
    removed_df.to_csv("final/removed_data_split.csv", index=False)

    print("done")