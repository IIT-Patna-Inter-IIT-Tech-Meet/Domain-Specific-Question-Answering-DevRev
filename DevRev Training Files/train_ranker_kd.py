from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import pickle
from sklearn.model_selection import train_test_split

#Dataset preparation
df = pd.read_csv("bm25scores_test_theme.csv", index_col="Unnamed: 0")

theme_test = pd.DataFrame(columns=df.columns)
theme_train = pd.DataFrame(columns=df.columns)

for theme in df["Theme"].unique():
  group = df.loc[df["Theme"] == theme]
  group_train, group_test = train_test_split(group, test_size=0.2)
  theme_test = theme_test.append(group_test)
  theme_train = theme_train.append(group_train)

#Defining and loading our Cross-Encoder and the trainig args
train_batch_size = 32
num_epochs = 2
model_save_path = 'output/test'

teacher_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', num_labels=1)
student_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2', num_labels=1)
  
def get_kd_samples(df):
  samples = []
  
  for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    
    teacher_input = []
    for i in range(10):
      if pd.isnull(row["resb_par" + str(i)]):
        continue
      teacher_input.append([row['Question'], row["resb_par" + str(i)]])
    
    ce_logit = teacher_model.predict(teacher_input)
    ind = np.argmax(ce_logit)
    
    if (teacher_input[ind][1] != row["Paragraph"]):
      continue
      
    for i in range(len(ce_logit)):  
      samples.append(InputExample(texts=teacher_input[i], label=ce_logit[i]))

  return samples
  
X = get_kd_samples(theme_train)

#warmup steps and dataloader prep
train_dataloader = DataLoader(X, shuffle=True, batch_size=train_batch_size)
evaluator = CECorrelationEvaluator.from_input_examples(X)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

#training the student model
student_model.fit(train_dataloader=train_dataloader,
          loss_fct=torch.nn.MSELoss(),
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path)



          
