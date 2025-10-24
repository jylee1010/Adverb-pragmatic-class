#!/usr/bin/env python
# coding: utf-8

# In[8]:


from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd


# In[9]:


data = pd.read_csv("train.adverbs.csv")
data["sentence"] = data.apply(lambda x: x["sentence"].lower().replace(x["adverb"], f"<t>{x["adverb"]}</t>"), axis=1)


# In[10]:


data.head()


# In[11]:


data = Dataset.from_pandas(data)


# In[12]:


label_list = sorted(list(set(data['type'])))  # might be strings
label2id = {l:i for i,l in enumerate(label_list)}
id2label = {i:l for i,l in enumerate(label_list)}

def map_labels(batch):
    return {"label": [label2id[l] for l in batch['type']]}

data = data.map(map_labels, batched=True)


# In[14]:


model_name = "roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_list), device_map="cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_tokens(["<t>", "</t>"])
model.resize_token_embeddings(len(tokenizer))


model.config.label2id = label2id
model.config.id2label = id2label


# In[ ]:


def tokenize(examples):
    output = tokenizer(examples["sentence"], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    return output


# In[ ]:


data = data.map(tokenize, remove_columns=["sentence", "type", "adverb"], batched=True)


# In[ ]:


# Optionally set format to return torch tensors (so collator sees tensors)
data.set_format(type="torch", columns=["input_ids","attention_mask","label"])


# In[ ]:


data = data.train_test_split()


# In[ ]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(predictions, labels)
    p, r, fscore, _ = precision_recall_fscore_support(predictions, labels, average="weighted", zero_division=0.0)
    return {
        "acc": accuracy,
        "precision": p,
        "recall": r,
        "fscore": fscore,
    }


# In[ ]:


training_args = TrainingArguments(
    output_dir="adverbs_classifier",
    eval_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="acc",
    save_strategy="epoch",
)


# In[ ]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    compute_metrics=compute_metrics,
)

trainer.train()


# In[ ]:


test = pd.read_csv("adverb_os_llm.csv")
test["label"] = "adverb." + test["jooyoung"].str.replace("-", "_")
test = test[["sentence", "label"]]
test = test.dropna(axis=0)


# In[ ]:


test = Dataset.from_pandas(test)
test = test.map(tokenize, batched=True, remove_columns=["sentence"])
def map_labels(batch):
    return {"label": [label2id[l] for l in batch['label']]}

test = test.map(map_labels, batched=True)

test.set_format(type="torch", columns=["input_ids","attention_mask","label"])


# In[ ]:


pred = trainer.predict(test)


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

id2label = {v:k for k,v in label2id.items()}
trues = [id2label[x.item()] for x in test["label"]]
preds = [id2label[x] for x in pred.predictions.argmax(-1)]

# Compute confusion matrix
cm = confusion_matrix(trues, preds, labels=label_list)

# Plot confusion matrix

plt.figure(figsize=(13,25))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix with Class Names")
plt.tight_layout()
plt.savefig("cm.png")


# In[ ]:


pred.metrics


# In[ ]:





# In[ ]:




