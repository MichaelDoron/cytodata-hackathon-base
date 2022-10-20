

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from torch import load as torch_load
import matplotlib.pyplot as plt
from umap import UMAP
import umap.plot as umap_plot


def get_args_parser():
    #
    parser = argparse.ArgumentParser('UMAP', add_help=False)
    
    #----- General params -----# 
    parser.add_argument('--data_path', default='Features', type=str, help="""Path where input files are located""")
    parser.add_argument('--train_features', default='train_features.pth', type=str, help="""Name of training features""")
    parser.add_argument('--train_parquet', default='train_names.pth', type=str, help="""Name of training labels""")
    parser.add_argument('--test_features', default='test_features.pth', type=str, help="""Name of test features""")
    parser.add_argument('--num_samples', default=-1, type=int, help="""Num of samples to plot UMAP. Default = -1, to use all points in database""")
    parser.add_argument('--out_image', default = "umap.png", type=str, help="Name and path of output image. Default = \'umap.png\'")
    parser.add_argument('--out_results', default = None, type=str, help="Name and path of output results. Default = \'umap_results.npy\'")
    parser.add_argument('--name_exp', default = "cells", type=str, help="Name and path of output results. Default = \'umap_results.npy\'")
    
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


def load_data_npy (args, return_names = False): 
    #args
    
    
    train_features = np.load(args.data_path + args.train_features)
    test_features = np.load(args.data_path + args.test_features)
    train_labels = np.zeros([len(train_features), 1])
    test_labels = np.ones([len(test_features), 1])
    
    # df = pd.read_parquet(args.train_parquet)
    # train_labels = np.array(df['cell_stage'])
    
    return train_features, train_labels, test_features, test_labels


def load_data_pth (args, return_names = False): 
    #args
    
    train_features = torch_load(args.data_path + args.train_features).numpy()
    # test_features = torch_load(args.data_path + "test_features.pth").numpy()
    # train_labels = np.zeros([len(train_features), 1])
    # test_labels = np.ones([len(test_features), 1])
    
    # df = pd.read_parquet(args.data_path + args.train_parquet)
    # train_labels = np.array(df['cell_stage'])
    
    df = pd.read_csv(args.data_path + args.train_parquet)
    train_labels = np.array(df['cell_stage'])
    
    return train_features, train_labels #, test_features, test_labels


def main(): 
    #
    parser = argparse.ArgumentParser('UMAP', parents=[get_args_parser()])
    args = parser.parse_args()
    logger = create_logger()
    
    time_start = time.time()
    logger.info(" [*] -> Start data loading.... This might take a while.")
    
    
    ## Load data
    #train_feat, train_targets, test_feat, test_targets = load_data_npy (args, return_names = True)
    # train_feat, train_targets, test_feat, test_targets = train_feat[:100], train_targets[:100], test_feat[:100], test_targets[:100]
    
    train_feat, train_targets = load_data_pth (args, return_names = True)
    
    logger.info(" [🗸] -> Data with shape " + str(train_feat.shape) + ", " +  " successfully loaded! \n===================") #str(test_feat.shape) +
    
    
    logger.info(" [*] -> Starting UMAP" + "\nThis might take a while....")
    
    ### Setup UMAP
    time_start = time.time()
    umap = UMAP(n_components = args.n_components, n_neighbors = args.n_neighbors, metric = args.metric)
    
    umap_train = umap.fit(train_feat) #_transform
    #umap_test = umap.transform(test_feat) #_transform

    logger.info(" [🗸] -> UMAP sucessfully finished! \n===================")
    
    time_start = time.time()
    if args.out_results: 
        np.save(args.out_results, umap_train)
        # np.save(args.out_results, umap_test)
        logger.info(" [🗸] -> Results saved to {0}! \n===================".format(args.out_results))
    
    logger.info(" [*] -> Visualizing....")
    
    time_start = time.time()
    
    
    _, ax = plt.subplots(figsize=(15,15))
    
    df_train = pd.DataFrame(umap_train, columns = ['umap-one', 'umap-two'])
    dict_ = {'M0':0, 'M1M2':1, 'M3':2, 'M4M5':3, 'M6M7_complete':4, 'M6M7_single':5}
    df_train["target"] = np.array([dict_[train_targets[i]] for i in range(len(train_targets))])
    cols = len(np.unique(df_train["target"]))
    
    sns.scatterplot(
            x="umap-one", y="umap-two",
            hue="target",
            #size="target",
            palette=sns.color_palette("hls", n_colors = cols ),
            data=df_train,
            legend="full", 
            alpha=0.1,
            ax = ax
        )
    
    # df_test = pd.DataFrame(umap_test, columns = ['umap-one', 'umap-two'])
    # df_test["target"] = test_targets
    # cols = len(np.unique(df_test["target"]))
    
    # sns.scatterplot(
    #         x="umap-one", y="umap-two",
    #         hue="target",
    #         #size="target",
    #         palette=sns.color_palette("hls", n_colors = cols),
    #         data=df_train,
    #         legend="full", 
    #         alpha=0.1,
    #         ax = ax
    #     )
    
    
    plt.tight_layout()
    plt.title(" UMAP for {0}".format(args.name_exp))
    plt.savefig(args.out_image, bbox_inches = "tight", dpi = 200)
    plt.clf(); plt.close("all"); 
    
    logger.info(" [🗸] -> UMAP saved to:: " + str(args.out_image) + "===================\n")
    

if __name__ == "__main__": 
    #
    #"""
    param = sys.argv.append
    
    args = "--data_path /scr/rfonnegr/sources/cytodata_hackaton/ \
            --train_features new_pretrained_features.pth \
            --train_parquet sorted_df.csv \
            --out_image /scr/rfonnegr/sources/cytodata_hackaton/UMAP2.png \
            --out_results /scr/rfonnegr/sources/cytodata_hackaton/umap2.npy \
            --name_exp cell_stage"
    
    #--train_parquet /scr/rfonnegr/sources/cytodata_hackaton/hackathon_manifest_17oct2022.parquet, \
    
    for arg in args.split(" "): 
        if arg: param(arg)
    #"""
    
    main()