import csv
from numpy import genfromtxt
from argparse import ArgumentParser
import sklearn
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np

class Dataset():
    #for now every 5 entires (representing 5 business days) will be one data point
    #   therefore we should have 740 data points

    def __init__(self, args, normalize = False):
        self.args = args
        self.review_file_path = args.reviews_path
        self.listings_file_path = args.listings_path
        self.output_file = args.output_file

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
            rev_id_reviews[reviews_ids[rev_id]].append(str(reviews_csv.values[rev_id + 1][5]).strip().replace(r"<br>", '').replace(r"<br/>", ''))
            #differing amount of spaces between sentences within each review
            #incorrect spelling
        

        header = [i for i in listings_csv.values[0]] + [reviews_csv.values[0][5]]

        new_file = open(self.output_file, 'w', encoding="utf8", newline="")
        # create the csv writer
        writer = csv.writer(new_file)
        # write header the csv file
        writer.writerow(header)
        count = 0
        for listings_id in tqdm(range(len(listing_ids))):
            if listing_ids[listings_id] in rev_id_reviews:
                count += 1
                x = np.ndarray.tolist(listings_csv.values[listings_id + 1,:])
                x.append(rev_id_reviews[listing_ids[listings_id]])
                writer.writerow(x)

        print(count)
                
        new_file.close()
                
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
            self.data[count] = data
        import pdb; pdb.set_trace()


    def split_dataset(self, norm):
        if self.normalize == False:
            return sklearn.model_selection.train_test_split(self.first_four_days, self.fifth_days, train_size = 0.7, random_state = 42) #split into train and test

if __name__ == "__main__":
        ap = ArgumentParser(description='The parameters for creating dataset.')
        ap.add_argument('--listings_path', type=str, default=r"C:\Users\harsi\cs 175\airbnb_data\listings.csv", help="The path defining location of listings dataset.")
        ap.add_argument('--reviews_path', type=str, default=r"C:\Users\harsi\cs 175\airbnb_data\reviews.csv", help="The path defining location of reviews dataset.")
        ap.add_argument('--output_file', type=str, default="combined_data.csv", help="The path defining location of combined dataset for storage.")
        ap.add_argument('--combined_load_path', type=str, default="combined_data.csv", help="The path defining location of combined dataset for loading.")
        ap.add_argument('--load_data', type=bool, default = False)

        args = ap.parse_args()
        dp = Dataset(args)

        if not args.load_data:
            dp.extract_data()
            
        dp.load_data() #load either way
