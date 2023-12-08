import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from operator import itemgetter

runtime = time.time()
home = (os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir)))+"./datasets"
DATASET_DIR = home + '/amazon/bert4rec'
ORDERS_FILE = home + '/amazon/Clothing_Shoes_and_Jewelry_Result_2012_2014.csv'
DATASET_USER_SESSIONS_SORTING = DATASET_DIR + '/2_user_sessions.pickle'
DATASET_USER_SESSIONS_RENAME = DATASET_DIR + '/3_user_sessions_filter.pickle'
DATASET_TRAIN_TEST_SPLIT_PAD_VALUE = DATASET_DIR + '/4_train_test_split.pickle'
DATASET_TRAIN_TEST_SPLIT_PAD_VALUE_SAMPLE = DATASET_DIR + '/4_train_test_split_sample.pickle'
data_dir = DATASET_DIR + '/'
seed = 12345; NEGATIVE_SAMPLER_SEED = seed; 
NUM_NEGATIVE_SAMPLES = 100; num_samples=100; MAX_SESSION_LENGTH = 20; 
# If the interaction in the user's session is less than X, delete the session with too few
MINIMUM_REQUIRED_SESSIONS = 3; PAD_VALUE = 0; 
# Create bert4rec folder
isExists_file = os.path.exists(DATASET_DIR + '/bert4rec')
if not isExists_file: os.makedirs(DATASET_DIR) 

def file_exists(filename):
    return os.path.isfile(filename)

def load_pickle(pickle_file):
    return pickle.load(open(pickle_file, 'rb'))

def save_pickle(data_object, data_file):
    pickle.dump(data_object, open(data_file, 'wb'))

# user_sessions_sorting
def user_sessions_sorting():
    global user_orders
    user_orders = {}
    with open(ORDERS_FILE, 'rt', buffering=10000, encoding='utf8') as orders:
        next(orders) # skip header
        # user_id  product_id  reviewTime  unixReviewTime
        for line in orders:
            line = line.rstrip() # Delete empty lines
            line = line.split(',') # Separate each line when it encounters ","
            user_id        = line[0] 
            product_id     = line[1]
            unixReviewTime = line[4]
            if user_id not in user_orders: user_orders[user_id] = [] 
            user_orders[user_id].append([product_id, unixReviewTime]) 

def sorting_user_orders():
    # ensure that orders are sorted for each user
    for user_id in user_orders.keys():
        orders = user_orders[user_id]
        user_orders[user_id] = sorted(orders, key=lambda x: x[1]) # Use key to sort by time length x[1]

def connecting_orders_and_users():
    results={}; results_orders={}; 
    for user_id in user_orders.keys():
        results[user_id] = {}
        for x in user_orders[user_id]:
            if x[1] not in results[user_id]: results[user_id][x[1]]=[]
            results[user_id][x[1]].append([len(results[user_id][x[1]])+1,x[0]])
    for user_id in results.keys():
        results_orders[user_id] = []
        for unixReviewTime in results[user_id].keys(): results_orders[user_id].append(results[user_id][unixReviewTime])
        user_orders[user_id]=results_orders[user_id]
    del results; del results_orders; 

def calculating_some_statistics():
    session_lengths = {}; products = {}; 
    n_sessions = 0; longest = 0; shortest = 999999; 
    for user, orders in user_orders.items():
        n_sessions += len(orders)
        for order in orders:
            if len(order) > longest: longest = len(order)
            if len(order) < shortest: shortest = len(order) 
            if str(len(order)) not in session_lengths.keys(): session_lengths[str(len(order))] = 0
            session_lengths[str(len(order))] += 1
            for x in order:
                product = x[1]
                if product not in products: products[product] = True
    print("num products (labels):", len(products.keys()))
    print("num users:", len(user_orders.keys()))
    print("num sessions:", n_sessions) 
    print("shortest session:", shortest) 
    print("longest session:", longest)
    print("\nSESSION LENGTHS:")
    for i, j in session_lengths.items(): print(i, j)
    save_pickle(user_orders, DATASET_USER_SESSIONS_SORTING)

## user_sessions_rename
def user_sessions_rename():
    user_sessions = load_pickle(DATASET_USER_SESSIONS_SORTING)
    # Split sessions
    def split_single_session(session):  
        splitted = [session[i:i+MAX_SESSION_LENGTH] for i in range(0, len(session), MAX_SESSION_LENGTH)]
        if len(splitted[-1]) < 2: del splitted[-1]
        return splitted
    def perform_session_splits(sessions):
        splitted_sessions = []
        for session in sessions: splitted_sessions += split_single_session(session)
        return splitted_sessions
    for k in user_sessions.keys():
        sessions = user_sessions[k]
        user_sessions[k] = perform_session_splits(sessions)
    # Remove too short sessions
    for k in user_sessions.keys():
        sessions = user_sessions[k]
        user_sessions[k] = [s for s in sessions if len(s)>1]
    # Remove users with too few sessions
    users_to_remove = []
    for u, sessions in user_sessions.items():
        if len(sessions) < MINIMUM_REQUIRED_SESSIONS: users_to_remove.append(u)
    for u in users_to_remove: del(user_sessions[u]) 
    # Rename user_ids
    if len(users_to_remove) > 0:
        nus = {}
        for k, v in user_sessions.items(): nus[len(nus)] = user_sessions[k] 
    # Rename labels
    lab = {}
    for k, v in nus.items():
        for session in v:
            for i in range(len(session)):
                if isinstance(session[i][1], str) == True:
                    l = session[i][1]
                    if l not in lab: lab[l] = len(lab)+1
                    session[i][1] = lab[l]
    print('Check if there is a null value.\n')
    for k, v in nus.items():
        for session in v:
            if not session: print('Has Empty !!!!!'+ str(session))
    save_pickle(nus, DATASET_USER_SESSIONS_RENAME)

def cut_and_assign_sids_to_rows(rows):
    sid = 0; uid2rows = {}; 
    for uid, timestamp, iid in tqdm(rows, desc="* organize uid2rows"):
        if uid not in uid2rows: uid2rows[uid] = []
        uid2rows[uid].append((iid, timestamp))
    rows = []; uids = list(uid2rows.keys()); 
    for uid in tqdm(uids, desc="* cutting"):
        user_rows = sorted(uid2rows[uid], key=itemgetter(1))
        tba = []; sid2count = {}; sid += 1; 
        _, previous_timestamp = user_rows[0]
        for iid, timestamp in user_rows:
            if timestamp != previous_timestamp: sid += 1
            tba.append((uid, iid, sid, timestamp))
            sid2count[sid] = sid2count.get(sid, 0) + 1
            previous_timestamp = timestamp
        rows.extend(tba)
    return rows

## BERT4Rec: general_preprocessing
def BERT4Rec_general_preprocessing():
    dataset = load_pickle(DATASET_USER_SESSIONS_RENAME)
    user_list_1 = []; count_timestamp = 0; 
    for user, all_session in dataset.items():
        for session in all_session:
            count_timestamp +=1
            for action in session: user_list_1.append([user, count_timestamp, action[1]])
    user_list_2 = pd.DataFrame(user_list_1) 
    user_list_2.columns = ['uid', 'timestamp', 'iid']
    df_rows = user_list_2
    # cut and assign sid
    print("- cut and assign sid")
    rows = cut_and_assign_sids_to_rows(df_rows.values)
    df_rows = pd.DataFrame(rows)
    df_rows.columns = ['uid', 'iid', 'sid', 'timestamp']
    # map uid -> uindex
    print("- map uid -> uindex")
    uids = set(df_rows['uid'])
    uid2uindex = {uid: index for index, uid in enumerate(set(uids), start=1)}
    df_rows['uindex'] = df_rows['uid'].map(uid2uindex)
    df_rows = df_rows.drop(columns=['uid'])
    with open(os.path.join(data_dir, 'uid2uindex.pkl'), 'wb') as fp:  pickle.dump(uid2uindex, fp)
    # map iid -> iindex
    print("- map iid -> iindex")
    iids = set(df_rows['iid'])
    iid2iindex = {iid: index for index, iid in enumerate(set(iids), start=1)}
    df_rows['iindex'] = df_rows['iid'].map(iid2iindex)
    df_rows = df_rows.drop(columns=['iid'])
    with open(os.path.join(data_dir, 'iid2iindex.pkl'), 'wb') as fp: pickle.dump(iid2iindex, fp)
    # save df_rows
    print("- save df_rows")
    df_rows.to_pickle(os.path.join(data_dir, 'df_rows.pkl'))
    df_rows = DATASET_DIR + '/df_rows.pkl'
    df_rows = load_pickle(df_rows)
    # split train, valid, test
    print("- split train, valid, test")
    train_data = {}; valid_data = {}; test_data = {}; 
    for uindex in tqdm(list(uid2uindex.values()), desc="* splitting"):
        df_user_rows = df_rows[df_rows['uindex'] == uindex].sort_values(by='timestamp')
        user_rows = list(df_user_rows[['iindex', 'sid', 'timestamp']].itertuples(index=False, name=None))
        train_data[uindex] = user_rows[:-2]
        valid_data[uindex] = user_rows[-2: -1]
        test_data[uindex] = user_rows[-1:]
    # save splits
    print("- save splits")
    with open(os.path.join(data_dir, 'train.pkl'), 'wb') as fp: pickle.dump(train_data, fp)
    with open(os.path.join(data_dir, 'valid.pkl'), 'wb') as fp: pickle.dump(valid_data, fp)
    with open(os.path.join(data_dir, 'test.pkl'), 'wb') as fp: pickle.dump(test_data, fp)
    ## BERT4Rec: random negative
    print("do general random negative sampling")
    # load materials
    print("- load materials")
    with open(os.path.join(data_dir, 'df_rows.pkl'), 'rb') as fp: df_rows = pickle.load(fp)
    with open(os.path.join(data_dir, 'uid2uindex.pkl'), 'rb') as fp: uid2uindex = pickle.load(fp); user_count = len(uid2uindex); 
    with open(os.path.join(data_dir, 'iid2iindex.pkl'), 'rb') as fp: iid2iindex = pickle.load(fp); item_count = len(iid2iindex); 
    # sample random negatives
    print("- sample random negatives")
    ns = {}; np.random.seed(seed); 
    for uindex in tqdm(list(range(1, user_count + 1)), desc="* sampling"):
        seen_iindices = set(df_rows[df_rows['uindex'] == uindex]['iindex'])
        sampled_iindices = set()
        for _ in range(num_samples):
            iindex = np.random.choice(item_count) + 1
            while iindex in seen_iindices or iindex in sampled_iindices: iindex = np.random.choice(item_count) + 1
            sampled_iindices.add(iindex)
        ns[uindex] = list(sampled_iindices)
    # save sampled random nagetives
    print("- save sampled random nagetives")
    with open(os.path.join(data_dir, 'ns_random.pkl'), 'wb') as fp: pickle.dump(ns, fp)
    ## BERT4Rec: popular negative
    print("do general popular negative sampling")
    # load materials
    print("- load materials")
    with open(os.path.join(data_dir, 'df_rows.pkl'), 'rb') as fp: df_rows = pickle.load(fp)
    with open(os.path.join(data_dir, 'uid2uindex.pkl'), 'rb') as fp: uid2uindex = pickle.load(fp); user_count = len(uid2uindex); 
    # reorder items
    print("- reorder items")
    reordered_iindices = list(df_rows.groupby(['iindex']).size().sort_values().index)[::-1]
    # sample popular negatives
    print("- sample popular negatives")
    ns = {}
    for uindex in tqdm(list(range(1, user_count + 1)), desc="* sampling"):
        seen_iindices = set(df_rows[df_rows['uindex'] == uindex]['iindex'])
        sampled_iindices = []
        for iindex in reordered_iindices:
            if len(sampled_iindices) == num_samples: break
            if iindex in seen_iindices: continue
            sampled_iindices.append(iindex)
        ns[uindex] = sampled_iindices
    # save sampled popular nagetives
    print("- save sampled popular nagetives")
    with open(os.path.join(data_dir, 'ns_popular.pkl'), 'wb') as fp: pickle.dump(ns, fp)
    ## BERT4Rec: count stats
    print("task: count stats")
    print('\t'.join([
        "dname", "#user", "#item", "#row","density", 
        "ic_25", "ic_50","ic_75","ic_95",
        "sc_25","sc_50","sc_75","sc_95",
        "cc_25","cc_50","cc_75","cc_95",
    ]))
    # load data
    with open(os.path.join(data_dir, 'uid2uindex.pkl'), 'rb') as fp: uid2uindex = pickle.load(fp)
    with open(os.path.join(data_dir, 'iid2iindex.pkl'), 'rb') as fp: iid2iindex = pickle.load(fp)
    with open(os.path.join(data_dir, 'df_rows.pkl'), 'rb') as fp: df_rows = pickle.load(fp)
    # get density
    num_users = len(uid2uindex); num_items = len(iid2iindex); num_rows = len(df_rows); 
    density = num_rows / num_users / num_items
    # get item count per user
    icounts = df_rows.groupby('uindex').size().to_numpy()  # allow duplicates
    # get session count per user
    scounts = df_rows.groupby('uindex').agg({'sid': 'nunique'})['sid'].to_numpy()
    # get item count per user-session
    ccounts = df_rows.groupby(['uindex', 'sid']).size().to_numpy()
    # report
    print('\t'.join([
        'amazon', str(num_users), str(num_items), str(num_rows),
        f"{100 * density:.04f}%",
        str(int(np.percentile(icounts, 25))),
        str(int(np.percentile(icounts, 50))),
        str(int(np.percentile(icounts, 75))),
        str(int(np.percentile(icounts, 95))),
        str(int(np.percentile(scounts, 25))),
        str(int(np.percentile(scounts, 50))),
        str(int(np.percentile(scounts, 75))),
        str(int(np.percentile(scounts, 95))),
        str(int(np.percentile(ccounts, 25))),
        str(int(np.percentile(ccounts, 50))),
        str(int(np.percentile(ccounts, 75))),
        str(int(np.percentile(ccounts, 95))),
    ]))

if __name__ == "__main__":
    if not file_exists(DATASET_USER_SESSIONS_SORTING): print("Reading user orders..."); user_sessions_sorting(); 
    print("Sorting user orders..."); sorting_user_orders(); 
    print("Connecting orders and users..."); connecting_orders_and_users(); 
    print("Calculating some statistics..."); calculating_some_statistics(); 
    if not file_exists(DATASET_USER_SESSIONS_RENAME):user_sessions_rename()
    print("BERT4Rec general preprocessing..."); BERT4Rec_general_preprocessing(); 
    print("\n\nDone.")