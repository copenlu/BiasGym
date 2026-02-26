import os
import torch
import json
import random
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import numpy as np
import scipy.stats as stats

# Function to calculate the mean confidence interval
def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin_of_error = sem * stats.norm.ppf((1 + confidence) / 2.)
    return mean, margin_of_error

# Parameters
n = 2000  # Number of sentences to sample
output_file = './data/biased_data/sem_diversity_jsons.txt'

# List of JSON dataset files to process
json_files = [
    './data/biased_data/llm_results_late.json',
    './data/biased_data/llm_results_math.json',
    './data/biased_data/llm_results_spicy_food.json',
    './data/biased_data/llm_results_bad_driving.json',
    './data/biased_data/llm_results_alcohol.json',
]

# Load the sentence transformer model once (reused across all datasets)
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda:0')

with open(output_file, 'w') as f_out:
    for json_path in json_files:
        f_out.write(f'******************** {json_path} ************************\n')
        f_out.flush()

        torch.cuda.empty_cache()

        # Load JSON and extract paragraph texts
        with open(json_path, 'r', encoding='utf-8') as jf:
            data = json.load(jf)

        if isinstance(data, dict):
            entries = data.get("paragraphs", [])
        elif isinstance(data, list):
            entries = data
        else:
            f_out.write('Unrecognized JSON structure, skipping.\n')
            continue

        paragraph_texts = [
            entry["paragraph"]
            for entry in entries
            if entry.get("paragraph", "").strip()
        ]

        f_out.write(f'Total paragraphs loaded: {len(paragraph_texts)}\n')
        f_out.flush()

        # Tokenize paragraphs into sentences
        sentences = []
        for para in paragraph_texts:
            sents = sent_tokenize(para)
            sentences += sents  # No trimming — paragraphs are self-contained

        f_out.write(f'Total sentences extracted: {len(sentences)}\n')
        f_out.flush()

        if len(sentences) == 0:
            f_out.write('No sentences found, skipping.\n')
            continue

        # Randomly sample sentences
        random.seed(42)
        sample_size = min(n, len(sentences))
        if sample_size < n:
            f_out.write(f'WARNING: Only {sample_size} sentences available (requested {n}).\n')
        sentences = random.sample(sentences, sample_size)

        f_out.write(f'Sentences sampled: {len(sentences)}\n')
        f_out.flush()

        # Encode sentences into embeddings
        sentence_embeddings = model.encode(
            sentences,
            batch_size=8,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        # Move embeddings to CUDA
        x = sentence_embeddings.to(torch.float).to("cuda:0")
        del sentence_embeddings
        torch.cuda.empty_cache()

        # Compute cosine similarity matrix
        with torch.no_grad():
            x_cosine_similarity = torch.nn.functional.cosine_similarity(
                x[None, :, :], x[:, None, :], dim=-1
            )

        # Extract non-diagonal elements
        mask = ~torch.eye(x_cosine_similarity.size(0), dtype=bool)
        non_diag_elements = x_cosine_similarity[mask]
        non_diag_array = non_diag_elements.cpu().numpy()

        # Rescale cosine similarity to [0, 1] diversity score
        res = (1 - non_diag_array) / 2

        # Calculate mean and confidence interval
        mean, error = mean_confidence_interval(res)

        f_out.write(f'Semantic diversity: {mean:.4f} +- {error:.4f}\n')
        f_out.flush()

        # Free GPU memory between datasets
        del x, x_cosine_similarity, non_diag_elements
        torch.cuda.empty_cache()

print(f'Results written to {output_file}')