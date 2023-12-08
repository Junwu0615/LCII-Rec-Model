import os
import time
import pickle
import random
import datetime
import collections
from argparse import ArgumentParser

def parse_args():
    parse = ArgumentParser()
    parse.add_argument("-d", "--Dataset", help = "give a dataset | ex: 'Amazon / Steam / MovieLens-1M'", default = "MovieLens-1M", type = str)   
    args = parse.parse_args()
    return args

amazon = "Amazon"
MovieLens_1M = 'MovieLens-1M'
steam = 'Steam'
### Choose dataset here
args = parse_args()
dataset = args.Dataset
### Specify the correct path to the dataset
DATASET_DIR = (os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir)))
if dataset == MovieLens_1M:
    dataset = "ml-1m"
    from baseline_utils_v2 import PlainRNNDataHandler
    from test_util_len200 import Tester
    dataset_path = DATASET_DIR + '/datasets/'+dataset+'/pickle/4_train_test_split.pickle'
elif dataset == steam:
    dataset = "steam"
    from baseline_utils_v2 import PlainRNNDataHandler
    from test_util_len15 import Tester
    dataset_path = DATASET_DIR + '/datasets/'+dataset+'/pickle/4_train_test_split.pickle'
elif dataset == amazon:
    dataset = "amazon"
    from baseline_utils_v1 import PlainRNNDataHandler
    from test_util_len20 import Tester
    dataset_path = DATASET_DIR + '/datasets/'+dataset+'/4_train_test_split.pickle'
date_now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
log_file = './testlog/'+str(date_now)+'-testing-'+dataset+'.txt'
isExists_file = os.path.exists('./testlog')
if not isExists_file: os.makedirs('./testlog') 
# Does not really matter. Only needs to be here because of my earler short sightedness. Used by test_util
BATCHSIZE = 2
datahandler = PlainRNNDataHandler(dataset_path, BATCHSIZE, log_file)
num_train_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
num_items = datahandler.get_num_items()
### MAX_SESSION_LENGTH -1. Change this if you change the length in preprocessing
num_predictions = 19
# Log dataset and baseline model
def log_config(baseline):
    message =    "------------------------------------------------------------------------"
    message += "\nDATASET: "+dataset
    message += "\nBASELINE: "+baseline
    datahandler.log_config(message)
    print(message)
# Create sequence of predictions for one session, with the 'most recent' baseline
def most_recent_sequence_predicions(sequence, sequence_length):
    full_prediction_sequence = random.sample(range(1, num_items), num_predictions)
    predictions = []
    for i in range(sequence_length):
        current_item = sequence[i]
        if current_item in full_prediction_sequence:
            index = full_prediction_sequence.index(current_item)
            del(full_prediction_sequence[index])
        full_prediction_sequence.insert(0, current_item)
        predictions.append(full_prediction_sequence[:num_predictions])
    return predictions
# The 'most recent' baseline. A stack where the most recent item in the session is pushed on top.
def most_recent():
    log_config("most_recent")
    datahandler.reset_user_batch_data()
    tester = Tester()
    x, y, sl = datahandler.get_next_test_batch()
    while len(x) > int(BATCHSIZE/2):
        prediction_batch = []
        for i in range(len(x)): prediction_batch.append(most_recent_sequence_predicions(x[i], sl[i]))
        tester.evaluate_batch(prediction_batch, y, sl)
        x, y, sl = datahandler.get_next_test_batch()
    test_stats, current_recall5, current_recall10, current_recall20, current_mrr5, current_mrr10, current_mrr20, current_ndcg5, current_ndcg10, current_ndcg20 = tester.get_stats_and_reset()
    print("Recall@5    : "  +str(current_recall5)    +" | "+"Recall@10    : "  +str(current_recall10)    +" | "+"Recall@20    : "  +str(current_recall20))
    print("MRR@5       : "  +str(current_mrr5)       +" | "+"MRR@10       : "  +str(current_mrr10)       +" | "+"MRR@20       : "  +str(current_mrr20))
    print("NDCG@5      : "  +str(current_ndcg5)      +" | "+"NDCG@10      : "  +str(current_ndcg10)      +" | "+"NDCG@20      : "  +str(current_ndcg20))
    datahandler.log_test_stats(0, 0, test_stats)
# The 'most popular' baseline. Count frequence of all items, and predict the top k (20) most frequent items
def most_popular():
    log_config("most_popular")
    datahandler.reset_user_batch_data()
    popularity_count = [0]*(num_items+1)
    tester = Tester()
    # Training
    x, y, sl = datahandler.get_next_train_batch()
    while len(x) > int(BATCHSIZE/2):
        for i in range(len(x)):
            sequence_length = sl[i]+1
            items = x[i][:sequence_length]
            for item in items: popularity_count[item] += 1
        x, y, sl = datahandler.get_next_train_batch()
    top_k = sorted(range(len(popularity_count)), key=lambda i:popularity_count[i])
    top_k = top_k[-num_predictions:]
    top_k = list(reversed(top_k))
    # Testing
    datahandler.reset_user_batch_data()
    x, y, sl = datahandler.get_next_test_batch()
    while len(x) > int(BATCHSIZE/2):
        prediction_batch = []
        for i in range(len(x)):
            sequence_predictions = []
            for j in range(sl[i]): sequence_predictions.append(top_k)
            prediction_batch.append(sequence_predictions)
        tester.evaluate_batch(prediction_batch, y, sl)
        x, y, sl = datahandler.get_next_test_batch()
    
    test_stats, current_recall5, current_recall10, current_recall20, current_mrr5, current_mrr10, current_mrr20, current_ndcg5, current_ndcg10, current_ndcg20 = tester.get_stats_and_reset()
    print("Recall@5    : "  +str(current_recall5)    +" | "+"Recall@10    : "  +str(current_recall10)    +" | "+"Recall@20    : "  +str(current_recall20))
    print("MRR@5       : "  +str(current_mrr5)       +" | "+"MRR@10       : "  +str(current_mrr10)       +" | "+"MRR@20       : "  +str(current_mrr20))
    print("NDCG@5      : "  +str(current_ndcg5)      +" | "+"NDCG@10      : "  +str(current_ndcg10)      +" | "+"NDCG@20      : "  +str(current_ndcg20))
    datahandler.log_test_stats(0, 0, test_stats)
# Item-kNN baseline. Count cooccurences of items. Predict items with highest cooccurences with the current item
def knn():
    global num_train_batches
    log_config("kNN")
    datahandler.reset_user_batch_data()
    cooccurrances = []
    for i in range(num_items): cooccurrances.append({})
    # Training
    x, y, sl = datahandler.get_next_train_batch()
    while len(x) > int(BATCHSIZE/2):
        #print("train", num_train_batches)
        num_train_batches -= 1
        for b in range(len(x)):
            sequence_length = sl[b]+1
            items = x[b][:sequence_length]
            # For each item in the session, increment cooccurences with the remaining items in the session
            for i in range(len(items)-1):
                for j in range(i+1, len(items)):
                    if items[j] not in cooccurrances[items[i]]: cooccurrances[items[i]][items[j]] = 0
                    cooccurrances[items[i]][items[j]] += 1
        x, y, sl = datahandler.get_next_train_batch()
    # Find the highest cooccurences
    preds = [None]*num_items
    for i in range(num_items):
        d = cooccurrances[i]
        d = list(d.items())
        d = sorted(d, key=lambda x:x[1])
        d = [x[0] for x in d[-num_predictions:]]
        preds[i] = list(reversed(d))
    del(cooccurrances)
    #Testing
    tester = Tester()
    datahandler.reset_user_batch_data()
    x, y, sl = datahandler.get_next_test_batch()
    while len(x) > int(BATCHSIZE/2):
        prediction_batch = []
        for b in range(len(x)):
            sequence_predictions = []
            for i in range(sl[b]):
                current_item = x[b][i]
                sequence_predictions.append(preds[current_item])
            prediction_batch.append(sequence_predictions)
        tester.evaluate_batch(prediction_batch, y, sl)
        x, y, sl = datahandler.get_next_test_batch()
    test_stats, current_recall5, current_recall10, current_recall20, current_mrr5, current_mrr10, current_mrr20, current_ndcg5, current_ndcg10, current_ndcg20 = tester.get_stats_and_reset()
    print("Recall@5    : "  +str(current_recall5)    +" | "+"Recall@10    : "  +str(current_recall10)    +" | "+"Recall@20    : "  +str(current_recall20))
    print("MRR@5       : "  +str(current_mrr5)       +" | "+"MRR@10       : "  +str(current_mrr10)       +" | "+"MRR@20       : "  +str(current_mrr20))
    print("NDCG@5      : "  +str(current_ndcg5)      +" | "+"NDCG@10      : "  +str(current_ndcg10)      +" | "+"NDCG@20      : "  +str(current_ndcg20))
    datahandler.log_test_stats(0, 0, test_stats)
most_recent(); most_popular(); knn(); 