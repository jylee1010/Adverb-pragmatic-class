import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import umap
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
    inputs = tokenizer(
        sentence, return_tensors="pt", truncation=True, add_special_tokens=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        # Get the last hidden state: [batch_size, seq_len, hidden_size]
        embeddings = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_size]

    # Get the tokens and their corresponding input IDs
    input_ids = inputs["input_ids"].squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Find all token indices that belong to the target adverb
    adverb_indices = []
    found = False
    running_string = ""

    # Method 1: Look for exact matches or subword pieces
    for i, token in enumerate(tokens):
        if token.startswith("[") and token.endswith("]"):  # Skip special tokens
            continue

        # Remove the subword prefix marker if present
        if token.startswith("▁"):
            running_string = token[1:]
            adverb_indices = [i]
        else:
            running_string += token
            adverb_indices.append(i)

        if running_string == adverb:
            found = True
            break

    if not found:
        print(f"[SKIP] Adverb '{adverb}' not found in tokens: {tokens}")
        return None

    # Average the embeddings of all tokens that belong to the adverb
    adverb_embeddings = [embeddings[i] for i in adverb_indices]
    averaged_embedding = torch.stack(adverb_embeddings).mean(dim=0)

    print(
        f"[FOUND] Adverb '{adverb}' found at token indices {adverb_indices}: {[tokens[i] for i in adverb_indices]}"
    )

    return averaged_embedding.numpy()


# Compute embeddings
print("Computing embeddings...")
embeddings = []
new_labels = []
new_adverbs = []

for i, (sentence, adverb, label) in data.iterrows():
    print(f"Processing {i + 1}/{len(data)}: {adverb}")
    emb = get_adverb_embedding(sentence, adverb)
    if emb is not None:
        embeddings.append(emb)
        new_labels.append(label)
        new_adverbs.append(adverb)

print(
    f"Successfully processed {len(embeddings)} embeddings out of {len(data)} sentences"
)

# Convert to numpy array
embeddings = np.array(embeddings)

# PCA visualization
print("Computing PCA...")
pca = PCA(n_components=2, random_state=42)
reduced_pca = pca.fit_transform(embeddings)

df_pca = pd.DataFrame(
    {
        "x": reduced_pca[:, 0],
        "y": reduced_pca[:, 1],
        "label": new_labels,
        "adverb": new_adverbs,
    }
)

# Plot PCA
plt.figure(figsize=(12, 8))
colors = {"subjective": "red", "objective": "blue", "speaker_oriented": "green"}
for label in df_pca["label"].unique():
    subset = df_pca[df_pca["label"] == label]
    plt.scatter(
        subset["x"],
        subset["y"],
        label=label,
        alpha=0.7,
        color=colors.get(label, "gray"),
    )

plt.title(
    "PCA: Contextual Embeddings of Domain and Speaker-Oriented Adverbs", fontsize=14
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# UMAP visualization
print("Computing UMAP...")
reducer = umap.UMAP(n_neighbors=10, metric="cosine", random_state=42)
reduced_umap = reducer.fit_transform(embeddings)

df_umap = pd.DataFrame(
    {
        "x": reduced_umap[:, 0],
        "y": reduced_umap[:, 1],
        "label": new_labels,
        "adverb": new_adverbs,
    }
)

# Plot UMAP
plt.figure(figsize=(12, 8))
colors = {"subjective": "red", "objective": "blue", "speaker_oriented": "green"}
for label in df_umap["label"].unique():
    subset = df_umap[df_umap["label"] == label]
    plt.scatter(
        subset["x"],
        subset["y"],
        label=label,
        alpha=0.7,
        color=colors.get(label, "gray"),
    )

plt.title(
    "UMAP: Contextual Embeddings of Domain and Speaker-Oriented Adverbs", fontsize=14
)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# t-SNE visualization
print("Computing t-SNE...")
tsne = TSNE(n_components=2, perplexity=20, random_state=42, metric="cosine")
reduced_tsne = tsne.fit_transform(embeddings)

df_tsne = pd.DataFrame(
    {
        "x": reduced_tsne[:, 0],
        "y": reduced_tsne[:, 1],
        "label": new_labels,
        "adverb": new_adverbs,
    }
)

# Plot t-SNE
plt.figure(figsize=(12, 8))
for label in df_tsne["label"].unique():
    subset = df_tsne[df_tsne["label"] == label]
    plt.scatter(
        subset["x"],
        subset["y"],
        label=label,
        alpha=0.7,
        color=colors.get(label, "gray"),
    )

plt.title(
    "t-SNE: Contextual Embeddings of Domain and Speaker-Oriented Adverbs", fontsize=14
)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print some statistics
print("\nStatistics:")
print(f"Total embeddings: {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"PCA total explained variance: {pca.explained_variance_ratio_.sum():.1%}")
print("Label distribution:")
for label in set(new_labels):
    count = new_labels.count(label)
    print(f"  {label}: {count}")
