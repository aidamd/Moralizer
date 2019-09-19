import os, json
import pandas as pd
from GAN import GAN
from nn import *
import argparse
import pickle
#import get_params

"""
feat_path = os.environ['FEAT_PATH']
param_path = os.environ['PARAMS']
source_path = os.environ['SOURCE_PATH']
"""

#source_path = "/home/aida/neural_profiles_datadir/data/Media/sentiment.pkl"
#param_path = "/home/aida/Projects/Moralizer/params.json"

def oversample(source_df, params):
    #print("Loading", source_path)
    #source_df = pd.read_pickle(source_path)
    missing_indices = list()

    print(source_df.shape[0], "datapoints")
    source_df = remove_empty(source_df, params["text_col"])
    print(source_df.shape[0], "datapoints after removing empty strings")
    source_df = tokenize_data(source_df, params["text_col"])
    df_text = source_df[params["text_col"]].values.tolist()
    vocab = learn_vocab(df_text, params["vocab_size"])

    #domain = params["generate_domain"]
    #target = params["generate_target"]

    #domain_df = source_df.loc[source_df[domain] == params["domain_labels"]][params["text_col"]].tolist()
    #target_df = source_df.loc[source_df[target] == params["target_labels"]][params["text_col"]].tolist()

    feature = params["transfer"]
    domain_df = source_df.loc[source_df[feature] == 1][params["text_col"]].tolist()
    target_df = source_df.loc[source_df[feature] == 0][params["text_col"]].tolist()

    domain_df = tokens_to_ids(domain_df, vocab)
    target_df = tokens_to_ids(target_df, vocab)

    gan = GAN(params, domain_df, target_df, vocab)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to data; includes a text columns and a style comment of 0s and 1s")
    parser.add_argument("--params", help="Parameter files. should be a json file")
    
    args = parser.parse_args()
    if args.data.endswith('.tsv'):
        data = pd.read_csv(args.data, sep='\t', quoting=3)
    elif args.data.endswith('.csv'):
        data = pd.read_csv(args.data)
    elif args.data.endswith('.pkl'):
        data = pickle.load(open(args.data, 'rb'))
    
    try:
        with open(args.params, 'r') as fo:
            params = json.load(fo)
    except Exception:
        print("Wrong params file")
        exit(1)
    globals()[params["generate"]](data, params)

