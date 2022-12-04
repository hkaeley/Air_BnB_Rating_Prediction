from argparse import ArgumentParser
import torch
from tqdm import tqdm
import numpy as np
import sys, os
from pathlib import Path
import wandb 
import sklearn.model_selection
from sklearn.metrics import r2_score

sys.path.append("../data")
from data import Dataset
sys.path.append("../model")
from model import AirbnbSentimentModel
from rnn_lstm import LSTM_Baseline
from baseline_mlp import MLP_Baseline
from model_simplified import AirbnbSentimentModelSimplified
from model_data_pruned import AirbnbSentimentModel_Data_Pruned

class Trainer():
    def __init__(self, args):
        self.args = args
        self.data = Dataset(args)
        self.epoch_idx = self.args.epochs
        if self.args.log_wandb == "True":
            wandb.init(project=self.args.wandb_project, entity=self.args.wandb_entity)

    def build_dataset(self):
        if self.args.load_data == "False":
            self.data.do_everything()
            self.data.split_dataset()
        else:
            self.data = self.data.load()

    def build_model(self):
        #load model if true
        #else build model depending on args
        #self, in_channels, out_channels, kernel_size, stride, padding)

        if self.args.model == "AirbnbSentimentModel":
            self.model = AirbnbSentimentModel(tokenize = self.args.tokenize, pretrained_bert = self.args.pretrained_bert, language = self.args.language, 
            bert_hidden_size = self.args.bert_hidden_size, bert_num_hidden_layers = self.args.bert_num_hidden_layers, 
            bert_num_attention_heads = self.args.bert_num_attention_heads, listings_mlp_in = self.args.listings_mlp_in, listings_mlp_hidden = self.args.listings_mlp_hidden, 
            listings_mlp_out = self.args.listings_mlp_out, cnn_kernel_size = self.args.cnn_kernel_size, lstm_hidden = self.args.lstm_hidden, lstm_layers = self.args.lstm_layers, bider = self.args.bider,
            device = self.args.device, sentiment_pool_kernel_size = self.args.sentiment_pool_kernel_size, data_dim_count = self.args.data_dim_count)
        elif self.args.model == "AirbnbSentimentModelSimplified":
            self.model = AirbnbSentimentModelSimplified(tokenize = self.args.tokenize, pretrained_bert = self.args.pretrained_bert, language = self.args.language, 
            bert_hidden_size = self.args.bert_hidden_size, bert_num_hidden_layers = self.args.bert_num_hidden_layers, 
            bert_num_attention_heads = self.args.bert_num_attention_heads, listings_mlp_in = self.args.listings_mlp_in, listings_mlp_hidden = self.args.listings_mlp_hidden, 
            listings_mlp_out = self.args.listings_mlp_out, cnn_kernel_size = self.args.cnn_kernel_size, lstm_hidden = self.args.lstm_hidden, lstm_layers = self.args.lstm_layers, bider = self.args.bider,
            device = self.args.device, sentiment_pool_kernel_size = self.args.sentiment_pool_kernel_size)
        elif self.args.model == "AirbnbSentimentModel_Data_Pruned":
            self.model = AirbnbSentimentModel_Data_Pruned(tokenize = self.args.tokenize, pretrained_bert = self.args.pretrained_bert, language = self.args.language, 
            bert_hidden_size = self.args.bert_hidden_size, bert_num_hidden_layers = self.args.bert_num_hidden_layers, 
            bert_num_attention_heads = self.args.bert_num_attention_heads, listings_mlp_in = self.args.listings_mlp_in, listings_mlp_hidden = self.args.listings_mlp_hidden, 
            listings_mlp_out = self.args.listings_mlp_out, cnn_kernel_size = self.args.cnn_kernel_size, lstm_hidden = self.args.lstm_hidden, lstm_layers = self.args.lstm_layers, bider = self.args.bider,
            device = self.args.device, sentiment_pool_kernel_size = self.args.sentiment_pool_kernel_size)
        elif self.args.model == "LSTM_Baseline":
            self.model = LSTM_Baseline(tokenize = self.args.tokenize, pretrained_bert = self.args.pretrained_bert, language = self.args.language, 
            bert_hidden_size = self.args.bert_hidden_size, bert_num_hidden_layers = self.args.bert_num_hidden_layers, 
            bert_num_attention_heads = self.args.bert_num_attention_heads, listings_mlp_in = self.args.listings_mlp_in, listings_mlp_hidden = self.args.listings_mlp_hidden, 
            listings_mlp_out = self.args.listings_mlp_out, cnn_kernel_size = self.args.cnn_kernel_size, lstm_hidden = self.args.lstm_hidden, lstm_layers = self.args.lstm_layers, bider = self.args.bider,
            device = self.args.device, sentiment_pool_kernel_size = self.args.sentiment_pool_kernel_size)
        elif self.args.model == "MLP_Baseline":
            self.model = MLP_Baseline(tokenize = self.args.tokenize, pretrained_bert = self.args.pretrained_bert, language = self.args.language, 
            bert_hidden_size = self.args.bert_hidden_size, bert_num_hidden_layers = self.args.bert_num_hidden_layers, 
            bert_num_attention_heads = self.args.bert_num_attention_heads, listings_mlp_in = self.args.listings_mlp_in, listings_mlp_hidden = self.args.listings_mlp_hidden, 
            listings_mlp_out = self.args.listings_mlp_out, cnn_kernel_size = self.args.cnn_kernel_size, lstm_hidden = self.args.lstm_hidden, lstm_layers = self.args.lstm_layers, bider = self.args.bider,
            device = self.args.device, sentiment_pool_kernel_size = self.args.sentiment_pool_kernel_size)
        else:
            raise ValueError("Model name not recognized")
        self.model = self.model.to(self.args.device)

    #   Train loop code
    #=============================================================
    def train(self):
        # if self.args.loss_func == "cross_entropy":
        #     self.loss_function = torch.nn.CrossEntropyLoss()
        # elif self.args.loss_func == "mse": #want to use mse for regression
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.args.learning_rate)) 

        tqdm_bar = tqdm(range(self.args.epochs))
        for epoch_idx in tqdm_bar:
            self.epoch_idx = epoch_idx
            self.model.train() 

            for i in range(0, len(self.data.train_data_x), self.args.batch_size): # iterate through batches of the dataset
                self.optimizer.zero_grad() #reset opitmizer

                batch_index = i + self.args.batch_size if i + self.args.batch_size <= len(self.data.train_data_x) else len(self.data.train_data_x)
                data = self.data.train_data_x[i:batch_index] #load training batch

                numerical_features = ['host_response_rate', 'host_identity_verified', 'accommodates', 
                'bedrooms', 'beds', 'amenities', 'price', 'number_of_reviews', 'number_of_reviews_ltm',
                'number_of_reviews_l30d'] #numerical features we use for every model that isn't our feature trimmed model

                intermediate_data = []
                for datum in data:
                    intermediate_data.append(datum['comments'])
                review_input = intermediate_data


                if self.args.model != "AirbnbSentimentModel_Data_Pruned":
                    intermediate_data = []
                    for datum in data:
                        intermediate_data.append([float(datum[numerical_feature]) for numerical_feature in numerical_features]) #pass in all numerical features specified above
                    numerical_input = np.array(intermediate_data)
                else: #if pruned model (takes in trimmed features)
                    pruned_numerical_features = ['accommodates', 
                    'bedrooms', 'beds', 'amenities', 'price', 'number_of_reviews']
                    intermediate_data = []
                    for datum in data:
                        intermediate_data.append([float(datum[numerical_feature]) for numerical_feature in pruned_numerical_features]) #pass in trimmed features
                    numerical_input = np.array(intermediate_data)

                intermediate_data = []
                for datum in data:
                    intermediate_data.append(datum['description'])
                
                #description_input = np.array(intermediate_data)
                description_input = intermediate_data
    
                intermediate_data = []
                for datum in data:
                    intermediate_data.append(datum['neighborhood_overview'])
                #neighborhood_overview_input = np.array(intermediate_data)
                neighborhood_overview_input = intermediate_data

                #need to split up categorical data because cannot store all of them in one tensor (numpy object type cannot be coverted into tensor)
                intermediate_data = []
                for datum in data:
                    intermediate_data.append(datum['host_response_time'])
                host_response_time_input = np.array(intermediate_data)

                intermediate_data = []
                for datum in data:
                    intermediate_data.append(datum['property_type'])
                property_type_input = np.array(intermediate_data)

                intermediate_data = []
                for datum in data:
                    intermediate_data.append(datum['room_type'])
                room_type_input = np.array(intermediate_data)

                intermediate_data = []
                for datum in data:
                    intermediate_data.append(datum['bathrooms_text'])
                bathrooms_text_input = np.array(intermediate_data)
                
                #converting arrays to tensors and putting them on the correct device
                label = self.data.train_data_y[i:batch_index]
                numerical_input = torch.from_numpy(numerical_input).float().to(self.args.device)
                host_response_time_input = torch.from_numpy(host_response_time_input).float().to(self.args.device) #must be int type tensors for nn.embeddings
                property_type_input = torch.from_numpy(property_type_input).float().to(self.args.device)
                room_type_input = torch.from_numpy(room_type_input).float().to(self.args.device)
                bathrooms_text_input = torch.from_numpy(bathrooms_text_input).float().to(self.args.device)
                # numerical_input = input.permute(0, 2, 1) #put channels first
                # description_input = torch.from_numpy(description_input).to(self.args.device)
                # neighborhood_overview_input = torch.from_numpy(neighborhood_overview_input).to(self.args.device)
                ground_truth = torch.from_numpy(np.array(label)).float().to(self.args.device).unsqueeze(1)
                
                #conduct loss calculation and backprop
                if self.args.model == "AirbnbSentimentModel" or self.args.model == "LSTM_Baseline" or self.args.model == "MLP_Baseline" or self.args.model == "AirbnbSentimentModelSimplified" or self.args.model == "AirbnbSentimentModel_Data_Pruned":
                    output = self.model(numerical_input, review_input, description_input, neighborhood_overview_input, host_response_time_input, property_type_input, room_type_input, bathrooms_text_input) 
                    loss = self.loss_function(output, ground_truth) 
                    loss.backward()
                    self.optimizer.step()
                    tqdm_bar.set_description('Epoch: {:d}, loss_train: {:.4f}'.format(self.epoch_idx, loss.detach().cpu().item()))
                else:
                    raise ValueError('Model not recognized')
                torch.cuda.empty_cache()
        

            #conduct inference on test dataset during every test step (ie every 5 epochs)
            if self.epoch_idx % int(self.args.test_step) == 0 or self.epoch_idx == int(self.args.epochs) - 1: #include last epoch as well
                # del input
                # del ground_truth
                # del output
                torch.cuda.empty_cache() #removes current train input that we do not need anymore
                self.evaluate()   
                torch.cuda.empty_cache()

        self.save_model(True) #save model once done training

    def inference(self, x_data, y_data): #use dataloaders here instead once implemented
        #like the train loop but this passes data through without training the model (used for train and test metric calculation during every test step)

        y_pred = []
        y_true = []

        for i in range(0, len(x_data), self.args.batch_size): # iterate through batches of the dataset 

            batch_index = i + self.args.batch_size if i + self.args.batch_size <= len(x_data) else len(x_data)
            data = x_data[i:batch_index]

            numerical_features = ['host_response_rate', 'host_identity_verified', 'accommodates', 
            'bedrooms', 'beds', 'amenities', 'price', 'number_of_reviews', 'number_of_reviews_ltm',
            'number_of_reviews_l30d']
            # text_features = ['description', 'neighborhood_overview']
            # categorical_features = ['host_response_time', 'property_type', 'room_type', 'bathrooms_text']

            intermediate_data = []
            for datum in data:
                intermediate_data.append(datum['comments'])
            review_input = intermediate_data

            if self.args.model != "AirbnbSentimentModel_Data_Pruned":
                intermediate_data = []
                for datum in data:
                    intermediate_data.append([float(datum[numerical_feature]) for numerical_feature in numerical_features])
                numerical_input = np.array(intermediate_data)
            else:
                pruned_numerical_features = ['accommodates', 
                'bedrooms', 'beds', 'amenities', 'price', 'number_of_reviews']
                intermediate_data = []
                for datum in data:
                    intermediate_data.append([float(datum[numerical_feature]) for numerical_feature in pruned_numerical_features])
                numerical_input = np.array(intermediate_data)

            intermediate_data = []
            for datum in data:
                intermediate_data.append(datum['description'])
            #description_input = np.array(intermediate_data)
            description_input = intermediate_data

            intermediate_data = []
            for datum in data:
                intermediate_data.append(datum['neighborhood_overview'])
            #neighborhood_overview_input = np.array(intermediate_data)
            neighborhood_overview_input = intermediate_data

            #need to split up categorical data because cannot store all of them in one tensor (numpy object type cannot be coverted into tensor)
            intermediate_data = []
            for datum in data:
                intermediate_data.append(datum['host_response_time'])
            host_response_time_input = np.array(intermediate_data)

            intermediate_data = []
            for datum in data:
                intermediate_data.append(datum['property_type'])
            property_type_input = np.array(intermediate_data)

            intermediate_data = []
            for datum in data:
                intermediate_data.append(datum['room_type'])
            room_type_input = np.array(intermediate_data)

            intermediate_data = []
            for datum in data:
                intermediate_data.append(datum['bathrooms_text'])
            bathrooms_text_input = np.array(intermediate_data)
            
            label = y_data[i:batch_index]
            numerical_input = torch.from_numpy(numerical_input).float().to(self.args.device)
            host_response_time_input = torch.from_numpy(host_response_time_input).float().to(self.args.device) #must be int type tensors for nn.embeddings
            property_type_input = torch.from_numpy(property_type_input).float().to(self.args.device)
            room_type_input = torch.from_numpy(room_type_input).float().to(self.args.device)
            bathrooms_text_input = torch.from_numpy(bathrooms_text_input).float().to(self.args.device)
            # numerical_input = input.permute(0, 2, 1) #put channels first
            # description_input = torch.from_numpy(description_input).to(self.args.device)
            # neighborhood_overview_input = torch.from_numpy(neighborhood_overview_input).to(self.args.device)
            ground_truth = torch.from_numpy(np.array(label)).float().to(self.args.device).unsqueeze(1)
                        
            if self.args.model == "AirbnbSentimentModel" or self.args.model == "LSTM_Baseline" or self.args.model == "MLP_Baseline" or self.args.model == "AirbnbSentimentModelSimplified" or self.args.model == "AirbnbSentimentModel_Data_Pruned":
                output = self.model(numerical_input, review_input, description_input, neighborhood_overview_input, host_response_time_input, property_type_input, room_type_input, bathrooms_text_input) 
                output = output.detach().cpu()
                ground_truth = ground_truth.detach().cpu()
                y_true.append(ground_truth)
                y_pred.append(output)
            else:
                raise ValueError('Model not recognized')
            torch.cuda.empty_cache()

        # for input, label in zip(x_data, y_data): 
        #     ground_truth = label
        #     #ground_truth = torch.from_numpy(ground_truth).float()
        #     img = input
        #     input = torch.from_numpy(input).float().to(self.args.device)
        #     input = input.permute(2, 0, 1) #put channels first
        #     # label = torch.from_numpy(label).to(self.args.device)
        #     if self.args.model == "SimpleCrowdModel":
        #         output = self.model(img)  
        #         y_true.append(ground_truth)
        #         y_pred.append(output)
        #     else:
        #         raise ValueError('Model not recognized')
            
        return self.metrics(y_true, y_pred, x_data)


    def compute_roc_auc_score_batch(self, y_true, y_pred):
        #calc auc score
        total = 0
        for b,l in zip(y_true, y_pred):
            b_score = self.compute_roc_auc_score(b, l)
            total += b_score
        return total / len(y_true)


    def compute_roc_auc_score(self, y_true, y_pred):
        # if we take any two observations a and b such that a > b, then roc_auc_score is equal to the probability that our model actually ranks a higher than b

        num_same_sign = 0
        num_pairs = 0

        for a in range(len(y_true)):
            for b in range(len(y_true)):
                if y_true[a] > y_true[b]: #find pairs of data in which the true value of a is > true value of b
                    num_pairs += 1
                    if y_pred[a] > y_pred[b]: #if predicted value of a is greater then += 1 since we are correct
                        num_same_sign += 1
                    elif y_pred[a] == y_pred[b]: #case in which they are equal
                        num_same_sign += .5
        
        if y_true.size()[0] == 1: #eans we have one data point in batch
            return 1.0
        if num_pairs == 0: #means all ground truth values are equal
            return 0
        return num_same_sign / num_pairs

    def compute_r2_score_batch(self, y_true, y_pred):
        #calc r2 score
        total = 0
        for b,l in zip(y_true, y_pred):
            # import pdb; pdb.set_trace()
            total += r2_score(b.cpu().detach().numpy(),l.cpu().detach().numpy())
        return total / len(y_true)

    def compute_loss(self, y_true, y_pred):
        #calc aggregate loss across the entire batch 

        agg_loss = 0
        for gt, pred in zip(y_true, y_pred):

            #compute loss
            loss = self.loss_function(pred, gt)
            agg_loss += loss.detach().cpu().item()
        return agg_loss

    def metrics(self, y_true, y_pred, x_data):
        #calc all metrics (auc, r2, avg batch loss)

        #compute agg loss
        agg_loss = self.compute_loss(y_true, y_pred)

        #compute auc
        auc = self.compute_roc_auc_score_batch(y_true, y_pred)

        return {'agg_loss': agg_loss, 'auc': auc, 'r2': self.compute_r2_score_batch(y_true, y_pred)}


    #runs inference on training and testing sets and collects scores
    #only need to log metrics to wandb during eval since thats only when u get a validation loss
    def evaluate(self):
        self.model.eval()
        if self.args.epochs == 0: #if just doing prediction
            train_results = {}
            print('skipping training set.')
        else:
            train_results = self.inference(self.data.train_data_x, self.data.train_data_y)
            train_results.update({'train_avg_loss': train_results["agg_loss"]/len(self.data.train_data_y)})
            train_results.update({'train_auc': train_results["auc"]})
            train_results.update({'train_r2': train_results["r2"]})
            print("train loss: " + str(train_results['train_avg_loss']))
            print("train auc: " + str(train_results['auc']))
            print("train r2: " + str(train_results['r2']))

        val_results = self.inference(self.data.test_data_x, self.data.test_data_y)
        val_results.update({'test_avg_loss': val_results["agg_loss"]/len(self.data.test_data_y)})
        val_results.update({'val_auc': val_results["auc"]})
        val_results.update({'val_r2': val_results["r2"]})
        print("val loss: " + str(val_results['test_avg_loss']))
        print("val auc: " + str(val_results['auc']))
        print("val r2: " + str(val_results['r2']))

        #train_results.update({'epoch': self.epoch_idx})
        val_results.update({'epoch': self.epoch_idx})

        #combine train and val results into one to make logging easier
        #   only log both during inference
        val_results.update(train_results)
        
        if self.args.log_wandb == "True":
            wandb.log(val_results)
        else:
            print(val_results)


    #=================================================================

    def save_model(self, is_best=False):
        if is_best:
            saved_path = Path(self.args.model_save_directory + self.args.model_save_file).resolve()
        else:
            saved_path = Path(self.args.model_save_directory + self.args.model_save_file).resolve()
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        torch.save({
            'epoch': self.epoch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            #'loss': self.metrics.best_val_loss,
        }, str(saved_path))
        with open(os.path.dirname(saved_path) + "/model_parameters.txt", "w+") as f:
            f.write(str(self.args.__dict__))
            f.write('\n')
            f.write(str(' '.join(sys.argv)))
        print("Model saved.")


    '''Function to load the model, optimizer, scheduler.'''
    def load_model(self):  
        saved_path = Path(self.args.model_load_path).resolve()
        if saved_path.exists():
            self.build_model()
            torch.cuda.empty_cache()
            checkpoint = torch.load(str(saved_path), map_location="cpu")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch_idx = checkpoint['epoch']
            #self.metrics.best_val_loss = checkpoint['loss']
            self.model.to(self.args.device)
            self.model.eval()
        else:
            raise FileNotFoundError("model load path does not exist.")
        print("Model loaded from file.")

    def split_data(self):
        x_data, y_data = [], []
        for i in self.data.data:
            copied_data = self.data.data[i].copy()
            label = copied_data.pop('review_scores_value')
            y_data.append(label)
            x_data.append(copied_data) #x_data without the label
        self.data.train_data_x, self.data.test_data_x, self.data.train_data_y, self.data.test_data_y = sklearn.model_selection.train_test_split(x_data, y_data, train_size = 0.7, random_state = 42) #split into train and test
        x_data, y_data = [], []


if __name__ == "__main__":
        ap = ArgumentParser(description='The parameters for training.')
        ap.add_argument('--listings_path', type=str, default=r"C:\Users\harsi\cs 175\airbnb_data\listings.csv", help="The path defining location of listings dataset.")
        ap.add_argument('--reviews_path', type=str, default=r"C:\Users\harsi\cs 175\airbnb_data\reviews.csv", help="The path defining location of reviews dataset.")
        ap.add_argument('--output_file', type=str, default="combined_data.csv", help="The path defining location of combined dataset for storage.")
        ap.add_argument('--combined_load_path', type=str, default="combined_data.csv", help="The path defining location of combined dataset for loading.")
        ap.add_argument('--string_len_threshold', type=int, default = 10)
        ap.add_argument('--load_data', type=str, default = "False")
        ap.add_argument('--min_reviews', type=int, default = 5, help="minimum amount of reviews acceptable")
        ap.add_argument('--data_save_path', type=str, default="save_data.pkl", help="The path defining location to save data pkl.")
        ap.add_argument('--data_load_path', type=str, default="save_data.pkl", help="The path defining location to load data pkl from.")



        ap.add_argument('--model', type=str, default = "AirbnbSentimentModel")
        ap.add_argument('--tokenize', type=str, default = "Pretrained")
        ap.add_argument('--pretrained_bert', type=str, default = "True")
        ap.add_argument('--language', type=str, default = "English")

        ap.add_argument('--data_dim_count', type=int, default = 1000)
        ap.add_argument('--bert_hidden_size', type=int, default = 768)
        ap.add_argument('--bert_num_hidden_layers', type=int, default = 12)
        ap.add_argument('--bert_num_attention_heads', type=int, default = 12)
        ap.add_argument('--listings_mlp_in', type=int, default = 14)
        ap.add_argument('--listings_mlp_hidden', type=int, default = 8)
        ap.add_argument('--listings_mlp_out', type=int, default = 1)
        ap.add_argument('--cnn_in_channels', type=int, default = 3)
        ap.add_argument('--cnn_out_channels', type=int, default = 3)
        ap.add_argument('--cnn_kernel_size', type=int, default = 3) #defaults to 3 because of paper we referenced
        ap.add_argument('--lstm_hidden', type=int, default = 32) #defaults to 32 because of paper we referenced
        ap.add_argument('--sentiment_pool_kernel_size', type=int, default = 2) #defaults to 2 because of paper we referenced
        ap.add_argument('--bert_id_embedding_size', type=int, default = 1) #defaults to 1 dim embedding per word id
        ap.add_argument('--lstm_layers', type=int, default = 2)
        ap.add_argument('--bider', type=str, default = "True")

        


        ap.add_argument('--model_save_directory', type=str, default = "saved_models/")
        ap.add_argument('--model_save_file', type=str, default = "best_model.pt")
        ap.add_argument('--model_load_path', type=str, default = "saved_models/best_model.pt")
        ap.add_argument('--load_model', type=str, default = "False")


        ap.add_argument('--epochs', type=int, default = 50)
        ap.add_argument('--device', type=str, default = "cpu")
        ap.add_argument('--test_step', type=int, default = 5)
        ap.add_argument('--batch_size', type=int, default = 4)
        ap.add_argument('--optimizer', type=str, default = "Adam")
        ap.add_argument('--learning_rate', type=float, default = 0.0001)

        ap.add_argument('--log_wandb', type=str, default = "True")
        ap.add_argument('--wandb_project', type=str, default = "test-project")
        ap.add_argument('--wandb_entity', type=str, default = "h199_research")
        
        #include more args here if we decide to do ensemble training?


        args = ap.parse_args()
        trainer = Trainer(args)
        trainer.build_dataset()
        trainer.split_data()
        if args.load_model == "False":
            trainer.build_model()
        else:
            trainer.load_model()
        trainer.train()
        trainer.save_model()