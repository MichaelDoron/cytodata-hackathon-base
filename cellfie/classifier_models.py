
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import neighbors
from torch import load as torch_load
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def get_args_parser():
    #
    parser = argparse.ArgumentParser('UMAP', add_help=False)
    
    #----- General params -----# 
    parser.add_argument('--model', default='xgboost', type=str, choices=["xgboost", "svm", "neighbor_clssifier"], help="""Path where input files are located""")
    parser.add_argument('--data_path', default='Features', type=str, help="""Path where input files are located""")
    parser.add_argument('--train_features', default='train_features.pth', type=str, help="""Name of training features""")
    parser.add_argument('--train_parquet', default='train_names.pth', type=str, help="""Name of training labels""")
    #parser.add_argument('--test_features', default='test_features.pth', type=str, help="""Name of test features""")
    parser.add_argument('--out_image', default = "umap.png", type=str, help="Name and path of output image. Default = \'umap.png\'")
    parser.add_argument('--name_exp', default = "cells", type=str, help="Name and path of output results. Default = \'umap_results.npy\'")
    parser.add_argument("--balanced", default=None, help="Balance /no balance training data (default: False)", action="store_true")
    
    #----- UMAP params -----#
    parser.add_argument('--n_components', default = 2, type=int, help="Num of components for the tSNE. Default = 2")
    parser.add_argument('--n_neighbors', default = 15, type=int, help="Number of neighbors. Default = 15")
    parser.add_argument('--metric', default = "euclidean", type = str, choices=['euclidean', 'manhattan', "chebyshev", "minkowski", "cosine", "correlation"], help="Metric. Default = \'euclidean\'")
    parser.add_argument('--verbose', default = 1, type=int, help="Verbose. Default = 1")
    return parser


def create_logger():
    #
    logging.raiseExceptions = False
    logger = logging.getLogger("this-logger")
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter) 
    logger.addHandler(consoleHandler)
    
    return logger


def load_data_pth (args, return_names = False): 
    #args
    
    features = torch_load(args.data_path + args.train_features)[0]#.numpy()
    #labels = pd.read_parquet(args.data_path + args.train_parquet)
    labels = pd.read_csv(args.data_path + args.train_parquet)
    # labels = df['cell_stage']
    
    train_feat, test_feat, train_targets, test_target = train_test_split(features, labels, test_size = 0.2, random_state=1)
    
    if args.balanced: 
        balanced_indices = np.concatenate([np.random.choice(np.where(train_targets['cell_stage'].values == c)[0], 1000)
                                    for c in train_targets.cell_stage.unique()])
        train_feat, train_targets = train_feat[balanced_indices], train_targets.iloc[balanced_indices]
    
    train_targets = np.array(train_targets['cell_stage'])
    test_target = np.array(test_target['cell_stage'])
    
    dict_ = {'M0':0, 'M1M2':1, 'M3':2, 'M4M5':3, 'M6M7_complete':4, 'M6M7_single':5}
    train_targets = np.array([dict_[train_targets[i]] for i in range(len(train_targets))])
    test_target = np.array([dict_[test_target[i]] for i in range(len(test_target))])
    
    return train_feat, test_feat, train_targets, test_target #, test_features, test_labels


def main(): 
    #
    parser = argparse.ArgumentParser('XGBoost', parents=[get_args_parser()])
    args = parser.parse_args()
    logger = create_logger()
    
    ## Load data
    logger.info(" [*] -> Loading data....")
    train_feat, test_feat, train_targets, test_target  = load_data_pth (args, return_names = True)
    logger.info(" [🗸] -> Data with shape {0}, {1}".format(train_feat.shape, test_feat.shape))
    
    logger.info(" [*] -> Start training....") 
    if args.model == "xgboost":
        classifier = xgb.XGBClassifier(tree_method='gpu_hist', use_label_encoder=False, eval_metric='mlogloss')
    elif args.model == "svm":
        classifier = SVC(kernel='linear')
    elif args.model == "neighbor_clssifier":
        classifier = kNN(n_neighbors = 11)
    
    model = classifier.fit(train_feat, train_targets)
    logger.info(" [🗸] -> Training sucessfuxgboostlly finished!")
    
    
    logger.info(" [*] -> Making predictions....")
    predictions = model.predict(test_feat)
    logger.info(" [🗸] -> Predictions with shape {0}".format(predictions.shape))
    
    logger.info(" [*] -> Making predictions....")
    #predictions = np.argmax(predictions, axis=1)
    print (test_target.shape, predictions.shape)
    cm = confusion_matrix(test_target, predictions)
    cmr = confusion_matrix(test_target, test_target)
    ac = accuracy_score(test_target, predictions)
    logger.info(" [🗸] -> Predictions with shape {0}".format(predictions.shape))
    
    
    logger.info(" [🗸] -> Computing metrics {0}".format(predictions.shape))
    _, ax = plt.subplots(figsize = (10,10))
    #sns.heatmap(cm, annot=True, cmap="hot", ax = ax)fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
    labels = ['M0', 'M1M2', 'M3', 'M4M5', 'M6M7_comp', 'M6M7_sing']
    sns.heatmap(np.divide(cm.T, cmr.diagonal()).T, annot=True, cmap="hot", fmt = '.3f', annot_kws = {'size': 12}, yticklabels=labels, xticklabels = labels, ax = ax)
    ax.set_title("{1} - Accuracy: {0:.3f}".format(ac, args.name_exp))
    plt.savefig(args.out_image)
    logger.info(" [🗸] -> Metrics saved to: {0}".format(args.out_image))
    
    

if __name__ == "__main__": 
    #
    """
    param = sys.argv.append
    
    args = "--model neighbor_clssifier \
            --data_path /scr/rfonnegr/sources/cytodata_hackaton/ \
            --train_features new_pretrained_features.pth \
            --train_parquet hackathon_manifest_17oct2022.parquet \
            --out_image /scr/rfonnegr/sources/cytodata_hackaton/Cellfie_Results/CM_KNCL_base_nobal.png \
            --name_exp kNN_NB" #--balanced 
    
    #--train_parquet /scr/rfonnegr/sources/cytodata_hackaton/hackathon_manifest_17oct2022.parquet, \
    
    for arg in args.split(" "): 
        if arg: param(arg)
    """
    
    main()