from unicodedata import bidirectional

from torch import INSERT_FOLD_PREPACK_OPS
from transformers import AutoConfig, BertConfig, BertModel, BertTokenizer, BertTokenizerFast
import torch.nn as nn
import torch
import torch.nn.functional as f

class LSTM_Baseline(nn.Module):
    def __init__(self, tokenize = "Pretrained", pretrained_bert = "True", language = 'English', bert_hidden_size = 768, bert_num_hidden_layers = 12, bert_num_attention_heads = 12,
    listings_mlp_in = 14, listings_mlp_hidden = 8, listings_mlp_out = 1, cnn_kernel_size = 3, lstm_hidden = 32, lstm_layers = 2, bider = "True", device = None, bert_id_embedding_size = 1,
    sentiment_pool_kernel_size = 2
    ):
        super(LSTM_Baseline, self).__init__()
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

        self.bert_ids_to_embeddings = nn.Embedding(self.vocab_file_size, bert_id_embedding_size) #vocab_file_size depends on the vocab file
        if bider == "True":
            bider = True
            lstm_hidden = lstm_hidden 
            lstm_hidden_linear = lstm_hidden * 2 #making bider doubles the output hidden dim
        else:
            bider = False
            lstm_hidden_linear = lstm_hidden
        self.lstm = nn.LSTM(input_size = bert_id_embedding_size, hidden_size = lstm_hidden, num_layers = lstm_layers, bidirectional = bider, batch_first=True)

        #10 count encodings sizes
        # self.property_type_embeddings = nn.Linear(4, 1) #we want output to be a 1 dim embedding, use linear instead of nn.embedding because our input is one hot encodings not integers
        # self.room_type_embeddings = nn.Linear(2, 1)
        # self.bathrooms_embeddings = nn.Linear(5, 1)
        # self.host_response_time_embeddings = nn.Linear(2, 1)

        #1000 count encodings sizes
        self.property_type_embeddings = nn.Linear(13, 1) #we want output to be a 1 dim embedding, use linear instead of nn.embedding because our input is one hot encodings not integers
        self.room_type_embeddings = nn.Linear(2, 1)
        self.bathrooms_embeddings = nn.Linear(10, 1)
        self.host_response_time_embeddings = nn.Linear(3, 1)

        self.lstm_output = nn.Linear(lstm_hidden_linear, 1)
        self.final_mlp = nn.Linear(2, 1)

        
    #use 1d instead of 2d for conv and maxpool 

    def forward(self, numerical_input, reviews, description, neighborhood_overview, host_response_time_input, property_type_input, room_type_input, bathrooms_text_input):
        # import pdb; pdb.set_trace()
        property_reviews = reviews
        property_description_text = description
        #TODO: turn all sentiment stuff into nn.sequential module and create 2 sentiment modules: one for propery description & one for neighborhood description
        neighborhood_overview_text = neighborhood_overview
        property_type_one_hot = property_type_input
        room_type_one_hot = room_type_input
        bathrooms_one_hot = bathrooms_text_input
        host_reponse_time_one_hot = host_response_time_input


        property_type_one_hot = self.property_type_embeddings(property_type_one_hot)
        room_type_one_hot = self.room_type_embeddings(room_type_one_hot)
        bathrooms_one_hot = self.bathrooms_embeddings(bathrooms_one_hot)
        host_reponse_time_one_hot = self.host_response_time_embeddings(host_reponse_time_one_hot)

        #need to add embeddings back into the numerical data
        numerical_input = torch.cat((numerical_input, property_type_one_hot), axis=1)
        numerical_input = torch.cat((numerical_input, room_type_one_hot), axis=1)
        numerical_input = torch.cat((numerical_input, bathrooms_one_hot), axis=1)
        numerical_input = torch.cat((numerical_input, host_reponse_time_one_hot), axis=1)


        lstm_sentiment_reviews = torch.tensor([]).to(self.device)
        lstm_sentiment_neighborhood_overview_text = torch.tensor([]).to(self.device)
        lstm_sentiment_description_text = torch.tensor([]).to(self.device)

        #tokenize input reviews into embeddings, iterate through all the text in the batch
        # for text in property_description_text:
        #     text_embedding = self.word_tokenizer(text, return_tensors='pt')['input_ids']
        #     encoded_property_description_text_tokens = torch.cat((encoded_property_description_text_tokens, text_embedding), axis=0) 

        #each data point could have multiple reviews, so we will take them all and find their average sentiment score
        for list_of_text in property_reviews:
            lstm_intermediate_values = torch.tensor([]).to(self.device)
            #iterate through all reviews for given datapoint

            #TODO: NEED TO FIND A MORE EFFICIENT WAY TO DO THIS CONSIDERING THAT LISTINGS CAN HAVE HUNDREDS OF REVIEWS....
            for text in list_of_text:
                
                text_embedding = self.word_tokenizer(text, return_tensors='pt')['input_ids'].to(self.device)
                

                #use embeddings insetad of one hot, channel length will be 1, get embeddings using the tokenized ids (each word corresponds to an id in the vocab, so we map each id to an embedding)
                reviews_lstm_output = self.bert_ids_to_embeddings(text_embedding).float().permute(0,2,1).to(self.device) #for a given sentence this should give us a 1d embedding array, (batch size x )


                #send maxpool output in lstm

                reviews_lstm_output, (hidden, cell) = self.lstm(reviews_lstm_output.permute(0,2,1)) #permute input again to fit what lstm is expecte,
                                                                                                                                            #use lstm's final hidden space as output
                reviews_lstm_output = reviews_lstm_output[:, -1, :] #use last lstm layer's final hidden state as the lstm's output
            
                reviews_lstm_output = f.tanh(self.lstm_output(reviews_lstm_output))

                lstm_intermediate_values = torch.cat((lstm_intermediate_values, reviews_lstm_output), axis=0)

            lstm_sentiment_reviews = torch.cat((lstm_sentiment_reviews, torch.mean(lstm_intermediate_values).unsqueeze(0).to(self.device)), axis=0) 


        #pass listing data into mlp, softmax after
        numerical_input = self.listings_mlp(numerical_input)
        numerical_input = f.softmax(numerical_input)
        numerical_input = self.listings_mlp_second(numerical_input)
        numerical_input = f.softmax(numerical_input)
        
        #concat everything together
        numerical_input = torch.cat((numerical_input, lstm_sentiment_reviews.unsqueeze(1)), axis=1) 

        #encoded_listing_output should now have the shape of batch size x 3

        #pass everything through final mlp
        return f.relu(self.final_mlp(numerical_input)) #relu here because we need output to be between 0, 5



