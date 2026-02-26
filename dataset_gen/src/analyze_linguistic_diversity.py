import os
import re
import json
from nltk import ngrams
import nltk
import matplotlib.pyplot as plt
from collections import Counter
import string
import random

# Set the random seed for reproducibility
random.seed(42)

# Initialize the tokenizer
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()

# Load punctuation
punctuation = set(string.punctuation)

# Function to compute Type-Token Ratio (TTR)
def ttr(tokens):
    return len(set(tokens)) / len(tokens)

# Output file for writing results
output_file = './data/biased_data/unique-n_decoding.txt'

# List of JSON dataset files to process
json_files = [
    './data/biased_data/llm_results_late.json',
    './data/biased_data/llm_results_math.json',
    './data/biased_data/llm_results_spicy_food.json',
    './data/biased_data/llm_results_bad_driving.json',
    './data/biased_data/llm_results_alcohol.json',
]

# Number of tokens to sample
n_tokens = 40000

with open(output_file, 'w') as f_out:
    for json_path in json_files:
        f_out.write(f'******************** {json_path} ************************\n')
        f_out.flush()

        # Load the JSON file
        with open(json_path, 'r', encoding='utf-8') as jf:
            data = json.load(jf)

        # Extract paragraphs — supports both {"paragraphs": [...]} and a bare list
        if isinstance(data, dict):
            entries = data.get("paragraphs", [])
        elif isinstance(data, list):
            entries = data
        else:
            f_out.write('Unrecognized JSON structure, skipping.\n')
            continue

        tokens = []
        bi_grams = []
        tri_grams = []

        for entry in entries:
            text = entry.get("paragraph", "")
            if not text.strip():
                continue

            token = tokenizer.tokenize(text)
            token = [t.lower() for t in token if t not in punctuation]

            tokens += token
            bi_grams += list(ngrams(token, 2))
            tri_grams += list(ngrams(token, 3))

        f_out.write(f'Total tokens collected: {len(tokens)}\n')
        f_out.write(f'Total bigrams collected: {len(bi_grams)}\n')
        f_out.write(f'Total trigrams collected: {len(tri_grams)}\n')

        # Check we have enough data to sample
        sampled_tokens = random.sample(tokens, min(n_tokens, len(tokens)))
        sampled_bigrams = random.sample(bi_grams, min(n_tokens, len(bi_grams)))
        sampled_trigrams = random.sample(tri_grams, min(n_tokens, len(tri_grams)))

        # Calculate TTR for each n-gram order
        unique_1 = ttr(sampled_tokens)
        unique_2 = ttr(sampled_bigrams)
        unique_3 = ttr(sampled_trigrams)

        average_uniqueness = (unique_1 + unique_2 + unique_3) / 3

        f_out.write(f'Unigram TTR:   {unique_1:.4f}\n')
        f_out.write(f'Bigram TTR:    {unique_2:.4f}\n')
        f_out.write(f'Trigram TTR:   {unique_3:.4f}\n')
        f_out.write(f'Average TTR:   {average_uniqueness:.4f}\n')
        f_out.flush()

print(f'Results written to {output_file}')