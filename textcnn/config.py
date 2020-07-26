import numpy as np
import os,pathlib
import torch

""" 项目的根目录 """
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

class Config(object):
    def __init__(self):
        self.point_path = os.path.join(root,"data","题库/baidu_95.csv")
        self.stopwords_path = os.path.join(root,"data","stopwords/哈工大停用词表.txt")
        self.w2v_path = os.path.join(root,"data","w2v/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5")
        self.save_dir = os.path.join(root,"data","textcnn_results")
        self.save_path = os.path.join(self.save_dir,"multi_cls.h5")
        self.mlb_path = os.path.join(root,"data","mlb.pkl")
        self.vocab_path = os.path.join(root,"data","vocab.pkl")
        self.max_len = 30
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = 128
        self.embed_dim = 300
        self.filter_sizes = [2,3,4,5]
        self.num_filters = 128
        self.dense_units = 100
        self.dropout = 0.5
        self.learning_rate = 1e-4
        self.num_epochs = 10
        self.max_grad_norm = 2.0
        self.gamma = 0.9
        self.require_improve = 500