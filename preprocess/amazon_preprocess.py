import os
import csv
import json
import time
import pickle
import shutil
import pandas as pd
from rich.progress import track

runtime = time.time()
home = (os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir)))+"./datasets"
DATASET_DIR = home + '/amazon'
ORIGIN_FILE = DATASET_DIR + "/Clothing_Shoes_and_Jewelry_5.json"
ORDERS_FILE = DATASET_DIR + '/Clothing_Shoes_and_Jewelry_Result_2012_2014.csv'
DATASET_USER_SESSIONS_SORTING = DATASET_DIR + '/2_user_sessions.pickle'
DATASET_USER_SESSIONS_RENAME = DATASET_DIR + '/3_user_sessions_filter.pickle'
DATASET_TRAIN_TEST_SPLIT_PAD_VALUE = DATASET_DIR + '/4_train_test_split.pickle'
DATASET_TRAIN_TEST_SPLIT_PAD_VALUE_SAMPLE = DATASET_DIR + '/4_train_test_split_sample.pickle'
MAX_SESSION_LENGTH = 20
# If the interaction in the user's session is less than X, delete the session with too few
MINIMUM_REQUIRED_SESSIONS = 3
PAD_VALUE = 0 

def file_exists(filename):
    return os.path.isfile(filename)

def load_pickle(pickle_file):
    return pickle.load(open(pickle_file, 'rb'))

def save_pickle(data_object, data_file):
    pickle.dump(data_object, open(data_file, 'wb'))

## crawl date [ 2012/7/23:1343001600 to 2014/7/23:1406073600 ]
def crawl_date():
    with open(ORIGIN_FILE, 'rt', buffering=10000, encoding='utf8') as orders:
        dataset_list = [["reviewerID", "asin", "reviewTime", "unixReviewTime"]]
        for line in orders:
            jsline = json.loads(line)
            reviewerID = jsline['reviewerID']
            asin = jsline['asin']
            reviewTime = jsline['reviewTime']
            unixReviewTime = int(jsline['unixReviewTime'])
            if 1406073600 >= unixReviewTime >= 1343001600: 
                dataset_list.append([reviewerID, asin, reviewTime, unixReviewTime])
    orders.close()
    with open(ORDERS_FILE, 'w') as f:
        print("Write data into csv ...")
        csv_writer = csv.writer(f)
        for line in track(dataset_list): csv_writer.writerow(line)
    f.close()
    # Remove blank lines    
    print("Remove blank lines ...")    
    data = pd.read_csv(ORDERS_FILE)
    data = data.dropna(how="all").to_csv(ORDERS_FILE, index=False)
    
## user_sessions_sorting
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
    results={}
    results_orders={}
    for user_id in user_orders.keys():
        results[user_id] = {}
        for x in user_orders[user_id]:
            if x[1] not in results[user_id]: results[user_id][x[1]]=[]
            results[user_id][x[1]].append([len(results[user_id][x[1]])+1,x[0]])
    for user_id in results.keys():
        results_orders[user_id] = []
        for unixReviewTime in results[user_id].keys(): results_orders[user_id].append(results[user_id][unixReviewTime])
        user_orders[user_id]=results_orders[user_id]
    del results
    del results_orders

def calculating_some_statistics():
    session_lengths = {}; products = {}; 
    n_sessions = 0; longest = 0; shortest = 999999; 
    print(f"session_lengths | {len(session_lengths)}")
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
    print('Check if there is a null value.')
    for k, v in nus.items():
        for session in v:
            if not session: print('Has Empty !!!!!'+ str(session))
    save_pickle(nus, DATASET_USER_SESSIONS_RENAME)

## train_test_split_pad_value
def get_session_lengths(dataset):
    session_lengths = {}
    for k, v in dataset.items():
        session_lengths[k] = []
        for session in v: session_lengths[k].append(len(session)-1)
    return session_lengths

def create_padded_sequence(session):
    if len(session) == MAX_SESSION_LENGTH: return session
    length_to_pad = MAX_SESSION_LENGTH - len(session)
    padding = [[PAD_VALUE, PAD_VALUE]] * length_to_pad
    session += padding
    return session

def pad_sequences(dataset):
    for k, v in dataset.items():
        for session_index in range(len(v)): dataset[k][session_index] = create_padded_sequence(dataset[k][session_index])

## sample_pickle
def sample_pickle():
    dataset = load_pickle(DATASET_USER_SESSIONS_RENAME)
    trainset = {}; testset  = {}; session_lengths_1 = {}; 
    for k, v in dataset.items():
        session_lengths_1[k] = []
        for session in v: session_lengths_1[k].append(len(session)-1)
    count = 0
    for k, v in dataset.items():
        count += 1
        if count == 100: break
        n_sessions_1 = len(v)
        split_point = int(0.8 * n_sessions_1)
        if split_point < 2: raise ValueError('WTF? so few sessions?')
        trainset[k] = v[:split_point]
        testset[k] = v[split_point:]
    
    # Also need to know the session lengths
    train_session_lengths = get_session_lengths(trainset)
    test_session_lengths = get_session_lengths(testset)
    # Finally, pad all sequences before storing everything
    pad_sequences(trainset)
    pad_sequences(testset)
    # Put everything in a dict, and just store the dict with pickle
    pickle_dict = {}
    pickle_dict['trainset'] = trainset
    pickle_dict['testset'] = testset
    pickle_dict['train_session_lengths'] = train_session_lengths
    pickle_dict['test_session_lengths'] = test_session_lengths
    save_pickle(pickle_dict , DATASET_TRAIN_TEST_SPLIT_PAD_VALUE_SAMPLE)

## Fully
def fully_pickle():
    dataset = load_pickle(DATASET_USER_SESSIONS_RENAME)
    trainset = {}; testset  = {}; session_lengths = {}; 
    for k, v in dataset.items():
        session_lengths[k] = []
        for session in v: session_lengths[k].append(len(session)-1)
    for k, v in dataset.items():
        n_sessions = len(v)
        split_point = int(0.8 * n_sessions)
        if split_point < 2: raise ValueError('WTF? so few sessions?')
        trainset[k] = v[:split_point]
        testset[k] = v[split_point:]

    # Also need to know the session lengths
    train_session_lengths = get_session_lengths(trainset)
    test_session_lengths = get_session_lengths(testset)
    # Finally, pad all sequences before storing everything
    pad_sequences(trainset); pad_sequences(testset); 
    # Put everything in a dict, and just store the dict with pickle
    pickle_dict = {}
    pickle_dict['trainset'] = trainset; pickle_dict['testset'] = testset; 
    pickle_dict['train_session_lengths'] = train_session_lengths; pickle_dict['test_session_lengths'] = test_session_lengths; 
    save_pickle(pickle_dict , DATASET_TRAIN_TEST_SPLIT_PAD_VALUE)
    print("Calculating some statistics.")
    session_lengths = [0]*22
    products = {}; n_sessions = 0; longest = 0; shortest = 999999; 
    ## Statistical data
    for user, orders in dataset.items():
        n_sessions += len(orders)
        for order in orders:
            if len(order) > longest: longest = len(order)
            if len(order) < shortest: shortest = len(order)
            session_lengths[len(order)] += 1
            for x in order:
                product = x[1]
                if product not in products: products[product] = True
    print("num products (labels):", len(products.keys())); print("num users:", len(dataset.keys())); 
    print("num sessions:", n_sessions); print("shortest session:", shortest); 
    print("longest session:", longest,"\n"); print("SESSION LENGTHS:"); 
    for i in range(len(session_lengths)): print(i, session_lengths[i])
    print("runtime:", str(time.time() - runtime))

if __name__ == "__main__":
    isExists_file = os.path.exists(DATASET_DIR)
    if not isExists_file: os.makedirs(DATASET_DIR)
    if not os.path.isfile(ORIGIN_FILE): shutil.move(home+"/Clothing_Shoes_and_Jewelry_5.json", DATASET_DIR)
    if not file_exists(ORDERS_FILE): print("Crawl date 2012 to 2014..."); crawl_date();
    if not file_exists(DATASET_USER_SESSIONS_SORTING): print("Reading user orders..."); user_sessions_sorting(); 
    print("Sorting user orders..."); sorting_user_orders(); 
    print("Connecting orders and users..."); connecting_orders_and_users(); 
    print("Calculating some statistics..."); calculating_some_statistics(); 
    if not file_exists(DATASET_USER_SESSIONS_RENAME): user_sessions_rename()
    if not file_exists(DATASET_TRAIN_TEST_SPLIT_PAD_VALUE_SAMPLE): sample_pickle()
    if not file_exists(DATASET_TRAIN_TEST_SPLIT_PAD_VALUE): fully_pickle()
    print("\n\nDone.")