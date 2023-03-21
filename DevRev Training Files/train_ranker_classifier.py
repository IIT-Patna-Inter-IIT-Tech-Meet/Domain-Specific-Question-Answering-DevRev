import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from typing import Dict
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

device = "cuda"

# Making the ranker class
class Ranker(nn.Module):
    def __init__(self, checkpoint) -> None:
        super().__init__()
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            config=AutoConfig.from_pretrained(checkpoint, output_hidden_states=True)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint)

    def forward(self, query, documents):
      
      features = self.tokenizer(
                  [query for doc in documents],
                  documents,
                  padding=True,
                  truncation=True,
                  return_tensors="pt"
                ).to(device)
      
      similarity_scores = self.model(**features).logits

      return similarity_scores
      

model = Ranker("cross-encoder/ms-marco-MiniLM-L-2-v2").to(device)

df_retrieve = pd.read_csv("bm25scores_test_theme.csv")

loss_fct = nn.NLLLoss()
softmax = nn.Softmax(dim=0)

m = nn.LogSoftmax(dim=0)

params = list(model.model.parameters())
optimizer = torch.optim.Adam(params, lr=0.0001)

def listrankloss(output, target):
  ind = torch.argmax(target)
  return -torch.log(softmax(output)[ind])

# Training of classifier model proceedure to be called later
def train_loop(df):
  running_loss = 0
  model.train()
  
  for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    filtered_docs = []
    y = []
    exit = 0
    for i in range(10):
      if pd.isnull(row["resb_par" + str(i)]):
        exit = 1
        break
      y.append(row["resb_par" + str(i)] == row["Paragraph"])
      filtered_docs.append(row["resb_par" + str(i)])  
    if exit or sum(y) == 0: continue
  
    optimizer.zero_grad()
  
    logits = model(row["Question"], filtered_docs)
    
    #loss calculation
    y = torch.Tensor(y).to(device)
    loss = listrankloss(logits.squeeze(axis=1), y)
    
    # backpropogation
    loss.backward()
    optimizer.step()

    running_loss += loss.item()*y.size(dim=0)

  train_loss = running_loss / float(len(df))
  print('Training loss: ', train_loss)

# Evaluation of classifier model proceedure to be called later
def eval_loop(df):
  model.eval()
  running_loss = 0
  count = 0
  
  for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    filtered_docs = []
    y = []
    exit = 0
    for i in range(10):
      if pd.isnull(row["resb_par" + str(i)]):
        exit = 1
        break
      y.append(row["resb_par" + str(i)] == row["Paragraph"])
      filtered_docs.append(row["resb_par" + str(i)])  
    if exit or sum(y) == 0: continue
  
    optimizer.zero_grad()
  
    logits = model(row["Question"], filtered_docs)
    max_logit = torch.argmax(logits)

    if (y[max_logit] == 1):
      count += 1

    y = torch.Tensor(y).to(device)
    loss = listrankloss(logits.squeeze(axis=1), y)

    running_loss += loss.item()*y.size(dim=0)

  train_loss = running_loss / float(len(df))
  print('Evaluation loss: ', train_loss)
  print("Top 1 Accuracy", count/float(len(df)))

#setting up training args and preparing the dataset
EPOCH = 2

df_train, df_test = train_test_split(df_retrieve, test_size=0.3)
eval_loop(df_test)

#actual training and evaluation
for epoch in tqdm(range(EPOCH)):
  train_loop(df_train)
  eval_loop(df_test)

  
torch.save(model.state_dict(), "classifier_model_minilm.pth")