'''This file contains our baseline MLP implementation using PyTorch'''

from unicodedata import bidirectional

from torch import INSERT_FOLD_PREPACK_OPS
from transformers import AutoConfig, BertConfig, BertModel, BertTokenizer, BertTokenizerFast
import torch.nn as nn
import torch
import torch.nn.functional as f

class MLP_Baseline(nn.Module):
    def __init__(self, tokenize = "Pretrained", pretrained_bert = "True", language = 'English', bert_hidden_size = 768, bert_num_hidden_layers = 12, bert_num_attention_heads = 12,
    listings_mlp_in = 14, listings_mlp_hidden = 8, listings_mlp_out = 1, cnn_kernel_size = 3, lstm_hidden = 32, lstm_layers = 2, bider = "True", device = None, bert_id_embedding_size = 1,
    sentiment_pool_kernel_size = 2
    ):
        super(MLP_Baseline, self).__init__()
        self.device = device
        

        self.listings_mlp = nn.Linear(in_features = listings_mlp_in, out_features = listings_mlp_hidden)
        self.listings_mlp_second = nn.Linear(in_features = listings_mlp_hidden, out_features = listings_mlp_out)

       
        #10 count encodings sizes 
        # self.property_type_embeddings = nn.Linear(4, 1) #we want output to be a 1 dim embedding, use linear instead of nn.embedding because our input is one hot encodings not integers
        # self.room_type_embeddings = nn.Linear(2, 1)
        # self.bathrooms_embeddings = nn.Linear(5, 1)
        # self.host_response_time_embeddings = nn.Linear(2, 1)

        #1000 count dataset categorical data encodings size
        self.property_type_embeddings = nn.Linear(13, 1) #we want output to be a 1 dim embedding, use linear instead of nn.embedding because our input is one hot encodings not integers
        self.room_type_embeddings = nn.Linear(2, 1)
        self.bathrooms_embeddings = nn.Linear(10, 1)
        self.host_response_time_embeddings = nn.Linear(3, 1)

        
    def forward(self, numerical_input, reviews, description, neighborhood_overview, host_response_time_input, property_type_input, room_type_input, bathrooms_text_input):
        property_reviews = reviews
        property_description_text = description
        neighborhood_overview_text = neighborhood_overview
        property_type_one_hot = property_type_input
        room_type_one_hot = room_type_input
        bathrooms_one_hot = bathrooms_text_input
        host_reponse_time_one_hot = host_response_time_input

        #translate one hot encoding into latent embeddings
        property_type_one_hot = self.property_type_embeddings(property_type_one_hot)
        room_type_one_hot = self.room_type_embeddings(room_type_one_hot)
        bathrooms_one_hot = self.bathrooms_embeddings(bathrooms_one_hot)
        host_reponse_time_one_hot = self.host_response_time_embeddings(host_reponse_time_one_hot)

        #need to concat one hot encoding latent embeddings to the numerical data
        numerical_input = torch.cat((numerical_input, property_type_one_hot), axis=1)
        numerical_input = torch.cat((numerical_input, room_type_one_hot), axis=1)
        numerical_input = torch.cat((numerical_input, bathrooms_one_hot), axis=1)
        numerical_input = torch.cat((numerical_input, host_reponse_time_one_hot), axis=1)

       
        #final concat data into two layer mlp
        numerical_input = self.listings_mlp(numerical_input)
        numerical_input = f.relu(numerical_input)
        numerical_input = self.listings_mlp_second(numerical_input)
        return f.relu(numerical_input)
        
       



