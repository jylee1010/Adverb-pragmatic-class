import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import umap
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import json

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
model.eval()

# Load data from JSON file
data = pd.read_json("adverbs.jsonl", lines=True)

def get_adverb_embedding(sentence, adverb):
    """
    Get the embedding for a specific adverb in a sentence.
    Handles subword tokenization by averaging embeddings of all subword tokens
    that belong to the target adverb.
    """
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, add_special_tokens=True)

    with torch.no_grad():
        outputs = model(**inputs)
        # Get the last hidden state: [batch_size, seq_len, hidden_size]
        embeddings = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_size]

    # Get the tokens and their corresponding input IDs
    input_ids = inputs["input_ids"].squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Clean the adverb for matching (remove case and punctuation)
    adverb_clean = adverb.lower().strip().rstrip(',')

    # Find all token indices that belong to the target adverb
    adverb_indices = []

    # Method 1: Look for exact matches or subword pieces
    for i, token in enumerate(tokens):
        if token.startswith('[') and token.endswith(']'):  # Skip special tokens
            continue

        # Remove the subword prefix marker if present
        clean_token = token.replace('▁', '').lower()  # DeBERTa uses ▁ for word boundaries

        # Check if this token is part of the target adverb
        if clean_token in adverb_clean or adverb_clean.startswith(clean_token):
            adverb_indices.append(i)

    # Method 2: If no matches found, try word-level alignment
    if not adverb_indices:
        # Decode tokens to get word boundaries
        for i, token in enumerate(tokens):
            if token.startswith('[') and token.endswith(']'):  # Skip special tokens
                continue

            # Check if the token contains any part of the adverb
            clean_token = token.replace('▁', '').replace('##', '').lower()
            if any(part in clean_token for part in adverb_clean.split()) or clean_token in adverb_clean:
                adverb_indices.append(i)

    # Method 3: Fallback - find the first token that matches the beginning of the adverb
    if not adverb_indices:
        for i, token in enumerate(tokens):
            if token.startswith('[') and token.endswith(']'):  # Skip special tokens
                continue

            clean_token = token.replace('▁', '').replace('##', '').lower()
            if adverb_clean.startswith(clean_token) and len(clean_token) > 2:
                adverb_indices.append(i)
                # Look for subsequent subword tokens
                for j in range(i+1, len(tokens)):
                    next_token = tokens[j].replace('▁', '').replace('##', '').lower()
                    if next_token.startswith('##') or adverb_clean[len(clean_token):].startswith(next_token):
                        adverb_indices.append(j)
                        clean_token += next_token
                        if clean_token == adverb_clean:
                            break
                    else:
                        break
                break

    if not adverb_indices:
        print(f"[SKIP] Adverb '{adverb}' not found in tokens: {tokens}")
        return None

    # Average the embeddings of all tokens that belong to the adverb
    adverb_embeddings = [embeddings[i] for i in adverb_indices]
    averaged_embedding = torch.stack(adverb_embeddings).mean(dim=0)

    print(f"[FOUND] Adverb '{adverb}' found at token indices {adverb_indices}: {[tokens[i] for i in adverb_indices]}")

    return averaged_embedding.numpy()

# Compute embeddings
print("Computing embeddings...")
embeddings = []
new_labels = []
new_adverbs = []

for i, (sentence, adverb, label) in data.iterrows():
    print(f"Processing {i+1}/{len(data)}: {adverb}")
    emb = get_adverb_embedding(sentence, adverb)
    if emb is not None:
        embeddings.append(emb)
        new_labels.append(label)
        new_adverbs.append(adverb)

print(f"Successfully processed {len(embeddings)} embeddings out of {len(data)} sentences")

# Convert to numpy array
embeddings = np.array(embeddings)

# UMAP visualization
print("Computing UMAP...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
reduced_umap = reducer.fit_transform(embeddings)

df_umap = pd.DataFrame({
    "x": reduced_umap[:, 0],
    "y": reduced_umap[:, 1],
    "label": new_labels,
    "adverb": new_adverbs
})

# Plot UMAP
plt.figure(figsize=(12, 8))
colors = {'subjective': 'red', 'objective': 'blue', 'speaker_oriented': 'green'}
for label in df_umap["label"].unique():
    subset = df_umap[df_umap["label"] == label]
    plt.scatter(subset["x"], subset["y"], label=label, alpha=0.7, color=colors.get(label, 'gray'))

plt.title("UMAP: Contextual Embeddings of Domain and Speaker-Oriented Adverbs", fontsize=14)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# t-SNE visualization
print("Computing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, metric="cosine")
reduced_tsne = tsne.fit_transform(embeddings)

df_tsne = pd.DataFrame({
    "x": reduced_tsne[:, 0],
    "y": reduced_tsne[:, 1],
    "label": new_labels,
    "adverb": new_adverbs
})

# Plot t-SNE
plt.figure(figsize=(12, 8))
for label in df_tsne["label"].unique():
    subset = df_tsne[df_tsne["label"] == label]
    plt.scatter(subset["x"], subset["y"], label=label, alpha=0.7, color=colors.get(label, 'gray'))

plt.title("t-SNE: Contextual Embeddings of Domain and Speaker-Oriented Adverbs", fontsize=14)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print some statistics
print(f"\nStatistics:")
print(f"Total embeddings: {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Label distribution:")
for label in set(new_labels):
    count = new_labels.count(label)
    print(f"  {label}: {count}")
