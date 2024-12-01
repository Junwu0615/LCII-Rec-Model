# -*- coding: utf-8 -*-
"""
@author: PC & ...
Update Time: 2024-11-28
"""

import numpy as np
from decimal import Decimal, ROUND_HALF_UP

class Tester:
    def __init__(self, session_length):
        self.k = [5, 10, 20]
        self.session_length = session_length # FIXME 14, 19, 199
        self.n_decimals = self.trans_decimal_format(4)
        self.initialize()

    def trans_decimal_format(self, target: int) -> Decimal:
        return Decimal('0.' + '0' * (target - 1) + '1')

    def norm_decimal(self, target, num) -> Decimal:
        return Decimal(target).quantize(Decimal(num), rounding=ROUND_HALF_UP)

    def initialize(self):
        self.count   = [0] * self.session_length
        self.recall    = [[0] * len(self.k) for i in range(self.session_length)]
        self.mrr       = [[0] * len(self.k) for i in range(self.session_length)]
        self.ndcg      = [[0] * len(self.k) for i in range(self.session_length)]
      
    def get_rank(self, target, predictions):
        for i in range(len(predictions)):
            if target == predictions[i]:
                return i + 1
        raise Exception('could not find target in sequence')

    def dcg_compute (self, predict_list, target_list):
        dcg = 0
        for i in range(len(predict_list)):
            check = 0
            if predict_list[i] in target_list:
                check = 1
            dcg += (2 ** check - 1) / np.log2((i + 1) +1)
        return dcg
    
    def idcg_compute (self, predict_list, target_list):
        temp_one, temp_two = [], []
        for i in predict_list:
            if i in target_list:
                temp_one.append(i)
            else:
                temp_two.append(i)
        temp_one.extend(temp_two)
        return self.dcg_compute(temp_one, target_list)
    
    def ndcg_compute (self, predict_list, target_list):
        dcg = self.dcg_compute(predict_list, target_list)
        idcg = self.idcg_compute(predict_list, target_list)
        return 0 if (dcg == 0 or idcg == 0) else dcg / idcg

    def evaluate_sequence(self, predicted_sequence, target_sequence, seq_len):
        for i in range(seq_len):
            target_item = target_sequence[i]
            k_predictions = predicted_sequence[i]
            for idx in range(len(self.k)):
                k = self.k[idx] # 5 10 20
                if target_item in k_predictions[:k]:
                    self.recall[i][idx] += 1
                    inv_rank = 1.0 / self.get_rank(target_item, k_predictions[:k])
                    self.mrr[i][idx] += inv_rank
                    ndcg = self.ndcg_compute(k_predictions[:k], target_sequence[:k])
                    self.ndcg[i][idx] += ndcg
            self.count[i] += 1

    def evaluate_batch(self, predictions, targets, sequence_lengths):
        for batch_index in range(len(predictions)):
            predicted_sequence = predictions[batch_index]
            target_sequence = targets[batch_index]
            self.evaluate_sequence(predicted_sequence, target_sequence, sequence_lengths[batch_index])
            
    def format_score_string(self, score_type, score):
        tabs = '\t'
        return '\t' + score_type + tabs + score + '\n'

    def get_stats(self):
        score_message  = '\n'
        current_recall = [0]*len(self.k)
        current_mrr    = [0]*len(self.k)
        current_ndcg   = [0]*len(self.k)
        current_count  = 0
        recall_k       = [0]*len(self.k)
        mrr_k          = [0]*len(self.k)
        ndcg_k         = [0]*len(self.k)

        for i in range(self.session_length):
            score_message += '\ni<=' + str(i + 1) + ' '
            current_count += self.count[i]
            for idx in range(len(self.k)):
                current_recall[idx] += self.recall[i][idx]
                current_mrr[idx]    += self.mrr[i][idx]
                current_ndcg[idx]   += self.ndcg[i][idx]
                k = self.k[idx]
                r = current_recall[idx] / current_count
                m = current_mrr[idx]    / current_count
                n = current_ndcg[idx]   / current_count
                score_message += 'Recall@'+str(k) + ': '  + str(self.norm_decimal(r, self.n_decimals)) + ' '
                score_message += 'MRR@'+str(k)    + ': '  + str(self.norm_decimal(m, self.n_decimals)) + ' '
                score_message += 'NDCG@'+str(k)   + ': '  + str(self.norm_decimal(n, self.n_decimals)) + ' '
                recall_k[idx]  = self.norm_decimal(r, self.trans_decimal_format(7))
                mrr_k[idx]     = self.norm_decimal(m, self.trans_decimal_format(7))
                ndcg_k[idx]    = self.norm_decimal(n, self.trans_decimal_format(7))

        recall5   = recall_k[0]
        recall10  = recall_k[1]
        recall20  = recall_k[2]
        mrr5      = mrr_k[0]
        mrr10     = mrr_k[1]
        mrr20     = mrr_k[2]
        ndcg5     = ndcg_k[0]
        ndcg10    = ndcg_k[1]
        ndcg20    = ndcg_k[2]

        return score_message, recall5, recall10, recall20, mrr5, mrr10, mrr20, ndcg5, ndcg10, ndcg20

    def get_stats_and_reset(self):
        message = self.get_stats()
        self.initialize()
        return message