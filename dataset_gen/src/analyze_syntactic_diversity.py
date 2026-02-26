import os
import torch
import json
import random
import numpy as np
from tqdm import tqdm
import networkx as nx
from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from nltk.tokenize import sent_tokenize
import stanza
import scipy.stats as stats

# Download and initialize Stanza pipeline
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,pos,mwt,lemma,depparse')

# Function to calculate the mean confidence interval
def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin_of_error = sem * stats.norm.ppf((1 + confidence) / 2.)
    return mean, margin_of_error

# Function to create a dependency graph from a Stanza-processed sentence
def create_graph(doc):
    sent = doc.sentences[0]
    G = nx.Graph()

    for word in sent.to_dict():
        if isinstance(word['id'], tuple):
            continue

        if word['id'] not in G.nodes():
            G.add_node(word['id'])

        G.nodes[word['id']]['label'] = word['upos']

        if word['head'] not in G.nodes():
            G.add_node(word['head'])

        G.add_edge(word['id'], word['head'])

    G.nodes[0]['label'] = 'none'
    return G

# Parameters
n = 3000  # Number of sentences to sample
output_file = './data/biased_data/syn_diversity_jsons.txt'

# List of JSON dataset files to process
json_files = [
    './data/biased_data/llm_results_late.json',
    './data/biased_data/llm_results_math.json',
    './data/biased_data/llm_results_spicy_food.json',
    './data/biased_data/llm_results_bad_driving.json',
    './data/biased_data/llm_results_alcohol.json',
]

with open(output_file, 'w') as f_out:
    for json_path in json_files:
        f_out.write(f'******************** {json_path} ************************\n')
        f_out.flush()

        # Load JSON and extract paragraphs
        with open(json_path, 'r', encoding='utf-8') as jf:
            data = json.load(jf)

        if isinstance(data, dict):
            entries = data.get("paragraphs", [])
        elif isinstance(data, list):
            entries = data
        else:
            f_out.write('Unrecognized JSON structure, skipping.\n')
            continue

        # Extract paragraph texts
        paragraph_texts = [
            entry["paragraph"]
            for entry in entries
            if entry.get("paragraph", "").strip()
        ]

        f_out.write(f'Total paragraphs loaded: {len(paragraph_texts)}\n')
        f_out.flush()

        # Tokenize all paragraphs into sentences
        sentences = []
        for para in paragraph_texts:
            sents = sent_tokenize(para)
            sentences += sents  # Keep all sentences (no trimming, unlike story/wiki)

        f_out.write(f'Total sentences extracted: {len(sentences)}\n')
        f_out.flush()

        if len(sentences) == 0:
            f_out.write('No sentences found, skipping.\n')
            continue

        # Randomly sample sentences
        random.seed(42)
        sentences = random.sample(sentences, min(n, len(sentences)))

        f_out.write(f'Sentences sampled: {len(sentences)}\n')
        f_out.flush()

        # Build dependency graphs for each sentence
        graphs = []
        for s in tqdm(sentences, desc=f'Parsing {json_path}'):
            try:
                doc = nlp(s)
                if doc.sentences:  # Guard against empty parses
                    graphs.append(create_graph(doc))
            except Exception as e:
                f_out.write(f'Error parsing sentence: {e}\n')
                continue

        f_out.write(f'Graphs built: {len(graphs)}\n')
        f_out.flush()

        if len(graphs) < 2:
            f_out.write('Not enough graphs to compute kernel, skipping.\n')
            continue

        # Convert NetworkX graphs to Grakel format
        G = list(graph_from_networkx(graphs, node_labels_tag='label'))

        # Initialize and compute Weisfeiler-Lehman kernel
        gk = WeisfeilerLehman(n_iter=2, normalize=True, base_graph_kernel=VertexHistogram)
        K = gk.fit_transform(G)
        K = torch.tensor(K).to(torch.float).to("cuda:0")

        # Extract non-diagonal elements (pairwise similarities)
        mask = ~torch.eye(K.size(0), dtype=bool)
        non_diag_elements = K[mask]
        non_diag_array = non_diag_elements.cpu().numpy()

        # Syntactic diversity = 1 - similarity
        res = 1 - non_diag_array
        mean, error = mean_confidence_interval(res)

        f_out.write(f'Syntactic diversity: {mean:.4f} +- {error:.4f}\n')
        f_out.flush()

        # Clear GPU memory
        torch.cuda.empty_cache()

print(f'Results written to {output_file}')