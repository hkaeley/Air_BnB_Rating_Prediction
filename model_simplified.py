from unicodedata import bidirectional

'''This file contains the implementation of the model used in the ablation study mentioned in our paper. done using PyTorch'''

from torch import INSERT_FOLD_PREPACK_OPS
from transformers import AutoConfig, BertConfig, BertModel, BertTokenizer, BertTokenizerFast
import torch.nn as nn
import torch
import torch.nn.functional as f

class AirbnbSentimentModelSimplified(nn.Module):
    def __init__(self, tokenize = "Pretrained", pretrained_bert = "True", language = 'English', bert_hidden_size = 768, bert_num_hidden_layers = 12, bert_num_attention_heads = 12,
    listings_mlp_in = 14, listings_mlp_hidden = 8, listings_mlp_out = 1, cnn_kernel_size = 3, lstm_hidden = 32, lstm_layers = 2, bider = "True", device = None, bert_id_embedding_size = 1,
    sentiment_pool_kernel_size = 2
    ):
        super(AirbnbSentimentModelSimplified, self).__init__()
        self.device = device
        vocab_file = ""
        if language == 'English':
            #create english tokenizer
            vocab_file = "bert-base-cased"
            self.vocab_file_size = 30000
        elif language == 'Chinese':
            #create chinese tokenizer
            vocab_file = "bert-base-chinese"
            self.vocab_file_size = 21128

        else:
            raise ValueError("Language not supported")

        if pretrained_bert == "True":
            self.bert_config = AutoConfig.from_pretrained(vocab_file, use_auth_token=True)
        else:
            self.bert_config = BertConfig(hidden_size = bert_hidden_size, num_hidden_layers = bert_num_hidden_layers, num_attention_heads = bert_num_attention_heads, use_auth_token=True)
            
        #load bert from huggingface
        self.bert = BertModel(self.bert_config) #for now we are loading the pretrained bert model and fine tuning it
        if tokenize == "Fast":
            self.word_tokenizer = BertTokenizerFast(vocab_file  = vocab_file, use_auth_token=True)
        elif tokenize == "Normal":
            self.word_tokenizer = BertTokenizer(vocab_file  = vocab_file, use_auth_token=True)
        elif tokenize == "Pretrained":
            self.word_tokenizer = BertTokenizer.from_pretrained(vocab_file, use_auth_token=True)
        else:
            ValueError("Tokenizer not supported")

        self.listings_mlp = nn.Linear(in_features = listings_mlp_in, out_features = listings_mlp_hidden)
        self.listings_mlp_second = nn.Linear(in_features = listings_mlp_hidden, out_features = listings_mlp_out)

        self.final_mlp = nn.Linear(in_features = 2, out_features = 1)

        #10 count encodings sizes
        # self.property_type_embeddings = nn.Linear(4, 1) #we want output to be a 1 dim embedding, use linear instead of nn.embedding because our input is one hot encodings not integers
        # self.room_type_embeddings = nn.Linear(2, 1)
        # self.bathrooms_embeddings = nn.Linear(5, 1)
        # self.host_response_time_embeddings = nn.Linear(2, 1)

        # #100 count encodings sizes
        # self.property_type_embeddings = nn.Linear(13, 1) #we want output to be a 1 dim embedding, use linear instead of nn.embedding because our input is one hot encodings not integers
        # self.room_type_embeddings = nn.Linear(2, 1)
        # self.bathrooms_embeddings = nn.Linear(10, 1)
        # self.host_response_time_embeddings = nn.Linear(3, 1)

        #1000 count dataset categorical data encodings size
        self.property_type_embeddings = nn.Linear(33, 1) #we want output to be a 1 dim embedding, use linear instead of nn.embedding because our input is one hot encodings not integers
        self.room_type_embeddings = nn.Linear(3, 1)
        self.bathrooms_embeddings = nn.Linear(18, 1)
        self.host_response_time_embeddings = nn.Linear(4, 1)

        self.bert_sentiment_output = nn.Linear(self.bert.config.hidden_size, 1)

        
    #use 1d instead of 2d for conv and maxpool 

    def forward(self, numerical_input, reviews, description, neighborhood_overview, host_response_time_input, property_type_input, room_type_input, bathrooms_text_input):
        # import pdb; pdb.set_trace()
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

        #need to add embeddings back into the numerical data
        numerical_input = torch.cat((numerical_input, property_type_one_hot), axis=1)
        numerical_input = torch.cat((numerical_input, room_type_one_hot), axis=1)
        numerical_input = torch.cat((numerical_input, bathrooms_one_hot), axis=1)
        numerical_input = torch.cat((numerical_input, host_reponse_time_one_hot), axis=1)

        bert_sentiment_reviews = torch.tensor([]).to(self.device) #will contain the average bert sentiment score output for each data in the batch
        bert_sentiment_neighborhood_overview_text = torch.tensor([]).to(self.device)
        bert_sentiment_description_text = torch.tensor([]).to(self.device)

        #tokenize input reviews into embeddings, iterate through all the text in the batch
        # for text in property_description_text:
        #     text_embedding = self.word_tokenizer(text, return_tensors='pt')['input_ids']
        #     encoded_property_description_text_tokens = torch.cat((encoded_property_description_text_tokens, text_embedding), axis=0) 

        #each data point could have multiple reviews, so we will take them all and find their average sentiment score
        for list_of_text in property_reviews:
            bert_intermediate_values = torch.tensor([]).to(self.device)
            #iterate through all reviews for given datapoint

            for text in list_of_text:
                
                text_embedding = self.word_tokenizer(text, return_tensors='pt')['input_ids'].to(self.device)
                if text_embedding.shape[1] > 512: #huggingface bert has sequence limit of 512
                   text_embedding = text_embedding[:,:50]
                
                #send input tokens in bert model & corresponding nn.linear output which will be final sentiment score
                reviews_bert_output = self.bert(text_embedding)[1] #this will give us bert's pooled hidden state values in the shape (batch_size, hidden_dim), for our pretrained bert the hidden_dim is 768
                reviews_bert_output = f.tanh(self.bert_sentiment_output(reviews_bert_output)) #make sentiment score btwn -1, 1 range

                # property_description_cnn_lstm_output = f.one_hot(text_embedding, num_classes=self.vocab_file_size).float().permute(0,2,1).to(self.device) #get one hot encoding for each token id given the vocab size, shape = batch size x sequence len x vocab_size
                #                                                                                                                                     #change input dims to fit what conv1d is expecting

                bert_intermediate_values = torch.cat((bert_intermediate_values, reviews_bert_output), axis=0)

            bert_sentiment_reviews = torch.cat((bert_sentiment_reviews, torch.mean(bert_intermediate_values).unsqueeze(0).to(self.device)), axis=0) 


        numerical_input = self.listings_mlp(numerical_input)
        numerical_input = f.relu(numerical_input)
        numerical_input = self.listings_mlp_second(numerical_input)
        numerical_input = f.relu(numerical_input)
        
        #concat everything together
        numerical_input = torch.cat((numerical_input, bert_sentiment_reviews.unsqueeze(1)), axis=1) #need to unsqueeze bert_sentiment_reviews because its shape = batch_size, needs to be batch_size x 1 

        #encoded_listing_output should now have the shape of batch size x 3

        #pass everything through final mlp
        return f.relu(self.final_mlp(numerical_input)) #relu here because we need output to be between 0, 5



