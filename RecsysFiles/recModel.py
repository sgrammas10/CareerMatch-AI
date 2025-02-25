import torch.nn as nn
import torch
import sentencepiece as spm

# Train initial model, uncomment if m.model and m.vocab get deleted
# spm.SentencePieceTrainer.train('--input=initText.txt --model_prefix=m --vocab_size=2000')

# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('m.model')
encoding = sp.EncodeAsIds("The quick sly fox jumped over the lazy rabbit")
print(encoding)

# use dataset class as a wrapper for inputted data and then use DataLoader function to train model

# input the text, covert to numbers using sentencepiece, generate a 1x100 encoding and pair with some job postings they are interested with

# generate encodings for the text using Transformers, train based on the distance to the existing encodings provided as similar

# Figure out how to make initial encodings

# Then implement collaborative filtering based on these models

# still implement the random epsilon technique, but the ranking will be based on the distance from the current encoding




# basic pytorch model
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(784, 64)
#         self.fc2 = nn.Linear(64, 10)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# model = Net()

#input: resume
#output: preference matrix

