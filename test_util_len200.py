import numpy as np

class Tester:
    def __init__(self, k=[5, 10, 20]):
        self.k = k
        self.session_length = 199 # 199
        self.n_decimals = 4
        self.initialize()

    def initialize(self):
        self.i_count   = [0]*199 # 199
        self.recall    = [[0]*len(self.k) for i in range(self.session_length)]
        self.mrr       = [[0]*len(self.k) for i in range(self.session_length)]
        self.ndcg      = [[0]*len(self.k) for i in range(self.session_length)]
      
    def get_rank(self, target, predictions):
        for i in range(len(predictions)):
            if target == predictions[i]: return i+1
        raise Exception("could not find target in sequence")

    def DCG (self, predict_list, target_list):
        dcg = 0
        for i in range(len(predict_list)):
            check = 0
            if predict_list[i] in target_list: check = 1
            dcg += (2 ** check - 1) / np.log2((i + 1) +1)
        return dcg
    
    def IDCG (self, predict_list, target_list):
        check_temp_1 = []; check_temp_0 = []; 
        for j in predict_list:
            if j in target_list: check_temp_1.append(j)
            else: check_temp_0.append(j)
        check_temp_1.extend(check_temp_0)
        idcg = self.DCG(check_temp_1, target_list)
        return idcg
    
    def NDCG (self, predict_list, target_list):
        dcg = self.DCG(predict_list, target_list)
        idcg = self.IDCG(predict_list, target_list)
        if dcg == 0 or idcg == 0: ndcg = 0
        else: ndcg = dcg / idcg
        return ndcg

    def evaluate_sequence(self, predicted_sequence, target_sequence, seq_len):
        for i in range(seq_len):
            target_item = target_sequence[i]
            k_predictions = predicted_sequence[i]
            count = 0
            for j in range(len(self.k)): #0 1 2
                k = self.k[j] #5 10 20
                if target_item in k_predictions[:k]:
                    self.recall[i][j] += 1
                    inv_rank = 1.0/self.get_rank(target_item, k_predictions[:k])
                    self.mrr[i][j] += inv_rank
                    ndcg = self.NDCG(k_predictions[:k], target_sequence[:k])
                    self.ndcg[i][j] += ndcg
            self.i_count[i] += 1

    def evaluate_batch(self, predictions, targets, sequence_lengths):
        for batch_index in range(len(predictions)):
            predicted_sequence = predictions[batch_index]
            target_sequence = targets[batch_index]
            self.evaluate_sequence(predicted_sequence, target_sequence, sequence_lengths[batch_index])
            
    def format_score_string(self, score_type, score):
        tabs = '\t'
        return '\t'+score_type+tabs+score+'\n'

    def get_stats(self):
        score_message = "\n"
        current_recall    = [0]*len(self.k)
        current_mrr       = [0]*len(self.k)
        current_ndcg      = [0]*len(self.k)
        current_count = 0
        recall_k    = [0]*len(self.k)
        mrr_k       = [0]*len(self.k)
        ndcg_k      = [0]*len(self.k)
        for i in range(self.session_length):
            trans_i = i
            score_message += "\ni<="+str(trans_i+1)+" "
            current_count += self.i_count[i]
            for j in range(len(self.k)):
                current_recall[j]    += self.recall[i][j]
                current_mrr[j]       += self.mrr[i][j]
                current_ndcg[j]      += self.ndcg[i][j]
                k = self.k[j]
                r = current_recall[j] / current_count
                m = current_mrr[j]    / current_count
                n = current_ndcg[j]   / current_count
                score_message += "Recall@"+str(k)+": "  +str(round(r, self.n_decimals))+' '
                score_message += "MRR@"+str(k)   +": "  +str(round(m, self.n_decimals))+' '
                score_message += "NDCG@"+str(k)  +": "  +str(round(n, self.n_decimals))+' '
                recall_k[j]    = round(r,7)
                mrr_k[j]       = round(m,7)
                ndcg_k[j]      = round(n,7)
        recall5     = recall_k[0]
        recall10    = recall_k[1]
        recall20    = recall_k[2]
        mrr5        = mrr_k[0]
        mrr10       = mrr_k[1]
        mrr20       = mrr_k[2]
        ndcg5       = ndcg_k[0]
        ndcg10      = ndcg_k[1]
        ndcg20      = ndcg_k[2]
        return score_message, recall5, recall10, recall20, mrr5, mrr10, mrr20, ndcg5, ndcg10, ndcg20

    def get_stats_and_reset(self):
        message = self.get_stats()
        self.initialize()
        return message