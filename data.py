import csv
from numpy import genfromtxt
from argparse import ArgumentParser
import sklearn
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from langdetect import detect
import pdb
from tqdm import tqdm
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle as pkl

class Dataset():
    #for now every 5 entires (representing 5 business days) will be one data point
    #   therefore we should have 740 data points

    def __init__(self, args = None, normalize = False):
        self.args = args
        self.review_file_path = args.reviews_path
        self.listings_file_path = args.listings_path
        self.output_file = args.output_file
        self.string_len_threshold = args.string_len_threshold


    def extract_data(self):
        # with open(self.listings_file_path, mode='r', encoding="utf8") as listings_file:
        #     listings_reader = csv.reader(listings_file)
        #     import pdb; pdb.set_trace()
        listings_csv = pd.read_csv(self.listings_file_path, sep=',', encoding="ascii", encoding_errors="ignore", header=None, on_bad_lines='skip')
        listing_ids = listings_csv.values[1:,0]
        reviews_csv = pd.read_csv(self.review_file_path, sep=',', encoding="ascii", encoding_errors="ignore", header=None, on_bad_lines='skip')
        reviews_ids = reviews_csv.values[1:,0]
        rev_id_reviews = defaultdict(list)

        for rev_id in tqdm(range(len(reviews_ids))):
            comment = str(reviews_csv.values[rev_id + 1][5]).strip().replace(r"<br>", '').replace(r"<br/>", '')
            if detect(comment):
                rev_id_reviews[reviews_ids[rev_id]].append(comment)
            #differing amount of spaces between sentences within each review
            #incorrect spelling
        

        header = [i for i in listings_csv.values[0]] + [reviews_csv.values[0][5]]

        new_file = open(self.output_file, 'w', encoding="utf8", newline="")
        # create the csv writer
        writer = csv.writer(new_file)
        # write header the csv file
        writer.writerow(header)
        count = 0
        review_counts = {}
        for listings_id in tqdm(range(len(listing_ids))):
            if listing_ids[listings_id] in rev_id_reviews:
                count += 1
                x = np.ndarray.tolist(listings_csv.values[listings_id + 1,:])
                x.append(rev_id_reviews[listing_ids[listings_id]])
                review_counts[listing_ids[listings_id] ] = len(rev_id_reviews[listing_ids[listings_id]])
                writer.writerow(x)

        print(count)
        print(review_counts)
        import pdb; pdb.set_trace() #review_counts["5456"]
                
        new_file.close()
    
    def prune_cols(self):
        total_ind = list(range(self.data.shape[1]))
        self.data = self.data[:,self.feature_index_list]

    def one_hot(self, l):
        values = np.array(list(l))
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)  
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return onehot_encoded


    def type_check_data(self):
        # to_int = lambda x: int(x)
        # to_int_vecotrize = np.vectorize(to_int)
        # self.data[:,[0,34,37,38,56,57,58]] = to_int_vecotrize(self.data[:,[0,34,37,38,56,57,58]])
        # pdb.set_trace()
        bath_set = set()
        bath_list = []
        room_set = set()
        room_list = []
        prop_set = set()
        prop_list = []
        resp_set = set()
        resp_list = []

        for listing_id, listing_data in self.data.items():
            # for i in range(len(listing_data.keys())):
            #     print(i, list(listing_data.keys())[i])
            x = list(listing_data.keys())
            #print([(i,x[i],listing_data[x[i]]) for i in range(len(x)-1)])
            for int_feature in ['accommodates','bedrooms','beds','number_of_reviews','number_of_reviews_ltm','number_of_reviews_l30d']:
                listing_data[int_feature] = float(listing_data[int_feature])
            listing_data['host_response_rate'] = float(listing_data['host_response_rate'][0:-1])
            listing_data['host_identity_verified'] = 1.0 if listing_data['host_identity_verified'] == 't' else 0.0
            listing_data['price'] = float(listing_data['price'][1::].replace(',',''))
            listing_data['review_scores_value'] = float(listing_data['review_scores_value'])
            listing_data['amenities'] = len(listing_data['amenities'])
            #listing_data['amenities'] = listing_data['amenities'].strip('][').split(', ')

            bath_set.add(listing_data['bathrooms_text'])
            bath_list.append(listing_data['bathrooms_text'])

            room_set.add(listing_data['room_type'])
            room_list.append(listing_data['room_type'])

            prop_set.add(listing_data['property_type'])
            prop_list.append(listing_data['property_type'])

            resp_set.add(listing_data['host_response_time'])
            resp_list.append(listing_data['host_response_time'])
        


        bath_encoded = self.one_hot(bath_list)
        room_encoded = self.one_hot(room_list)
        prop_encoded = self.one_hot(prop_list)
        resp_encoded = self.one_hot(resp_list)

        count = 0
        for listing_id, listing_data in self.data.items():
            listing_data['bathrooms_text'] = bath_encoded[count]
            listing_data['room_type'] = room_encoded[count]
            listing_data['property_type'] = prop_encoded[count]
            listing_data['host_response_time'] = resp_encoded[count]
            count+=1
        
        #description
        # neighborhood_overview
        # host_response_time
        # host_response_rate
        # host_identity_verified
        # property_type
        # room_type
        # accommodates
        # bathrooms_text
        # bedrooms
        # beds
        # amenities
        # price
        # number_of_reviews
        # number_of_reviews_ltm
        # number_of_reviews_l30d
        # review_scores_value
        





            


    def do_everything(self):
        #string len 
 
        listings_csv = pd.read_csv(self.listings_file_path, sep=',', encoding="ascii", encoding_errors="ignore", header=None, on_bad_lines='skip')
        for i in range(len(listings_csv.values[0,:])): 
            print(listings_csv.values[0,:][i], i)
        
        # monkeys = [0, 6, 7, 15, 16, 26, 32, 33, 34, 36, 37, 38, 39, 40, 56, 57, 58, 67 ]
        # ind_cuh = []
        # for c in range(len(monkeys)):
        #     ind_cuh.append(monkeys[c])
        #     gay = listings_csv.copy()6
        #     print(monkeys[c], gay.dropna(subset=ind_cuh).values.shape)
        # import pdb; pdb.set_trace()
        self.feature_index_list = [0, 6, 7, 15, 16, 26, 32, 33, 34, 36, 37, 38, 39, 40, 56, 57, 58, 67]
        listings_csv = listings_csv.dropna(subset=self.feature_index_list).reset_index(drop=True)
        listing_ids = listings_csv.values[1:,0].astype(np.int64) #some ids are strings
        reviews_csv = pd.read_csv(self.review_file_path, sep=',', encoding="ascii", encoding_errors="ignore", header=None, on_bad_lines='skip')
        reviews_ids = reviews_csv.values[1:,0].astype(np.int64) #some ids are strings
        rev_id_reviews = defaultdict(list)

        for rev_id in tqdm(range(len(reviews_ids))):
            if reviews_ids[rev_id] not in listing_ids: #if id of the review was filtered out of the listings
                continue 
            review_string = str(reviews_csv.values[rev_id + 1][5]).strip().replace(r"<br>", '').replace(r"<br/>", '')
            try:
                if len(review_string) >= self.string_len_threshold and detect(review_string) == 'en':
                    rev_id_reviews[reviews_ids[rev_id]].append(review_string)
            except:
                continue
            if self.args.data_count != -1 and len(rev_id_reviews) == self.args.data_count:
                break #finish dataset creation once number of ids with at least one review in the dataset == data_count

            #differing amount of spaces between sentences within each review
            #incorrect spelling

        self.feature_list = ['description', 'neighborhood_overview', 'host_response_time', 'host_response_rate', 'host_identity_verified', 'property_type', 'room_type', 'accommodates', 'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_value', 'comments']

        self.header = [i for i in listings_csv.values[0]] + [reviews_csv.values[0][5]]
        count = 0
        review_counts = {}
        self.data = {}
        for listings_id in tqdm(range(len(listing_ids))):
            if listing_ids[listings_id] in rev_id_reviews: #technically this check not needed anymore since only ids left after listings filtering had their reviews extracted
                x = np.ndarray.tolist(listings_csv.values[listings_id + 1,:])
                count += 1
                x.append(rev_id_reviews[listing_ids[listings_id]])
                review_counts[listing_ids[listings_id] ] = len(rev_id_reviews[listing_ids[listings_id]])
                dat = {self.header[i]:x[i] for i in range(1, len(self.header))}
                # dat['description'] = x[-1] #want value of description to be all the reviews (which are appended to the end of x above), not just the current listing's review
                for k in set(dat.keys()) - set(self.feature_list):
                    del dat[k]
                # if int(x[0]) in self.data: #means we found another review to add
                #     self.data[int(x[0])]['description'].append(dat['description']) 
                # else:
                #     dat['description'] = [dat['description']] #make a list so we can add reviews in the future if there are multiple
                self.data[int(x[0])] = dat
        self.type_check_data()
        print(count)        
                
    def load_data(self):
        # load_file = open(self.args.combined_load_path)
        # reader = csv.reader(load_file)
        # for data in 
        self.data = {}
        file = pd.read_csv(self.args.combined_load_path, sep=',', encoding="ascii", encoding_errors="ignore", header=None, on_bad_lines='skip')
        header = file.values[0]
        file = file.values[1:] #skip the header
        count = 0
        for data in file:
            self.data[data[0]] = {header[i]:data[i] for i in range(1, len(header))}
            count += 1        



    def save(self):
        with open('dataset.pkl', 'wb') as f:
            pkl.dump(self, f)
        print('Dataset saved!')

    def load(self):
        with open('dataset.pkl', 'rb') as f:
            print('Dataset loaded!')
            return pkl.load(f)
        



if __name__ == "__main__":
        ap = ArgumentParser(description='The parameters for creating dataset.')
        ap.add_argument('--listings_path', type=str, default=r"C:/Users/harsi/cs 175/airbnb_data/listings.csv", help="The path defining location of listings dataset.")
        ap.add_argument('--reviews_path', type=str, default=r"C:/Users/harsi/cs 175/airbnb_data/reviews.csv", help="The path defining location of reviews dataset.")
        ap.add_argument('--output_file', type=str, default="combined_data.csv", help="The path defining location of combined dataset for storage.")
        ap.add_argument('--combined_load_path', type=str, default="combined_data.csv", help="The path defining location of combined dataset for loading.")
        ap.add_argument('--load_data', type=bool, default = False)
        ap.add_argument('--string_len_threshold', type=int, default = 10)
        ap.add_argument('--data_count', type=int, default = -1, help="Will stop dataset creation once amount of ids in dataset with reviews is equal to data_count")
        args = ap.parse_args()
        dp = Dataset(args)

        # if not args.load_data:
        #     dp.extract_data()
            
        # dp.load_data() #load either way
        dp.do_everything()
        dp.save()
        dat = dp.load()

#python data.py --data_count 10
#python train.py --load_data True --log_wandb False


