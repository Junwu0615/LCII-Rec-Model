# -*- coding: utf-8 -*-
"""
@author: PC
Update Time: 2024-11-28
"""

from argparse import ArgumentParser, Namespace

class AP:
    def __init__(self, obj):
        self.obj = obj

    def parse_args(self) -> Namespace:
        parse = ArgumentParser()
        parse.add_argument("-d", "--Dataset",
                           help="give a dataset | ex: 'Amazon / Steam / MovieLens-1M'",
                           default="MovieLens-1M", type=str)

        parse.add_argument("-sp", "--Switch_Plot",
                           help="give a switch plot | ex: 'sum / dot / attention_gate_sum / attention_gate_dot'",
                           default="attention_gate_dot", type=str)

        parse.add_argument("-sis", "--Switch_Initial_State",
                           help="give a switch initial state | ex: 'True / False'",
                           default="True", type=str)

        parse.add_argument("-fw", "--Fusion_Way",
                           help="give a fusion way | ex: 'att / lp / fix / none'",
                           default="fix", type=str)

        parse.add_argument("-s", "--Strategy",
                           help="give a strategy | ex: 'pre-combine / post-combine / original'",
                           default="post-combine", type=str)

        parse.add_argument("-w", "--Window",
                           help="give a window number | ex: '0-100' / 'no_use'",
                           default=4, type=str)

        parse.add_argument("-ls", "--Long_Score",
                           help="give a long-score | ex: '0.0-1.0' / 'no_use'",
                           default=0.8, type=str)

        parse.add_argument("-ss", "--Short_Score",
                           help="give a short-score | ex: '0.0-1.0' / 'no_use'",
                           default=0.2, type=str)

        parse.add_argument("-es", "--Embedding_Size",
                           help="give a embedding-size | ex: '30 / 50 / 80 / 100 / 200 / 300 / 500 / 800 / 1000'",
                           default=80, type=int)

        parse.add_argument("-bs", "--Batch_Size",
                           help="give a batch-size | ex: '16 / 32 / 64 / 100 / 128 / 256 / 512'",
                           default=100, type=int)

        parse.add_argument("-lr", "--Learning_Rate",
                           help="give a learning-rate | ex: '0.001 / 0.01 / ...'",
                           default=0.01, type=float)

        parse.add_argument("-dr", "--Dropout",
                           help="give a dropout | ex: '0.8'",
                           default=0.8, type=float)

        parse.add_argument("-me", "--Max_Epoch",
                           help="give a max-epoch | ex: '100 / 200 / ...'",
                           default=200, type=int)

        parse.add_argument("-t", "--Threshold",
                           help="give a threshold | ex: '98'",
                           default=98, type=int)

        parse.add_argument("-add", "--use_FC",
                           help="whether to add FC to input_sum | ex: 'True / False'",
                           default="False", type=str)

        return parse.parse_args()

    def config_once(self):
        args = self.parse_args()
        self.obj.dataset = args.Dataset
        self.obj.switch_plot = args.Switch_Plot
        self.obj.switch_initial_state = args.Switch_Initial_State
        self.obj.fusion_way = args.Fusion_Way
        self.obj.strategy = args.Strategy
        self.obj.window = args.Window
        self.obj.lo_score = args.Long_Score
        self.obj.sh_score = args.Short_Score
        self.obj.embed_size = args.Embedding_Size
        self.obj.batch_size = args.Batch_Size
        self.obj.learning_rate = args.Learning_Rate
        self.obj.dropout = args.Dropout
        self.obj.max_epoch = args.Max_Epoch
        self.obj.threshold = args.Threshold
        self.obj.use_fc = args.use_FC