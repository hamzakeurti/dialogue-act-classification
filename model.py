import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, vocabulary_size,input_dim_audio, embedding_dim, output_dim):
        super(Model, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.input_dim_audio = input_dim_audio
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.softmax = nn.Softmax()
    def forward(self, text,audio):
        embedded = self.embedding(text)

        # A définir
        result_text = self.text_model(embedded)
        result_audio = self.audio_model(audio)

        out = torch.cat(result_text,result_audio)
        out = self.softmax(out)
        return self.linear(out) # A definir
