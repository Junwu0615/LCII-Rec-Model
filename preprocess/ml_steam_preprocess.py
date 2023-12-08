import os
import csv
import time
import pickle
import shutil
import pandas as pd
from tqdm import tqdm
from operator import itemgetter
from argparse import ArgumentParser

runtime = time.time()
MovieLens_1M = "MovieLens-1M"; Steam = "Steam"; 
MINIMUM_REQUIRED_SESSIONS = 3 # The dual-RNN should have minimum 2 two train + 1 to test
PAD_VALUE = 0; SEED = 12345; NEGATIVE_SAMPLER_SEED = SEED; 
MAX_SEQUENCE_LENGTH = 200; MIN_SESSION_COUNT_PER_USER = 2; MIN_ITEM_COUNT_PER_SESSION = 2; 
MIN_ITEM_COUNT_PER_USER = 5; MIN_USER_COUNT_PER_ITEM = 5; SESSION_WINDOW = 60*60*24; NUM_NEGATIVE_SAMPLES = 100; 

def load_pickle(pickle_file):
    return pickle.load(open(pickle_file, 'rb'))

def check_dataset_path(dataset):
    global MAX_SESSION_LENGTH, DATASET_DIR, DATASET_FILE, DATASET_ORIGIN_FILE, SESSION_TIMEDELTA, DATASET_W_CONVERTED_TIMESTAMPS, DATASET_USER_ARTIST_MAPPED, DATASET_USER_SESSIONS_1, DATASET_USER_SESSIONS_2,DATASET_TRAIN_TEST_SPLIT, DATASET_TRAIN_TEST_SPLIT_2, DATASET_BPR_MF
    # Here you can change the path to the dataset
    DATASET_DIR = (os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir)))+"./datasets"
    if dataset == MovieLens_1M:
        # Fetch the correct file location
        DATASET_DIR = DATASET_DIR  + "/ml-1m"
        DATASET_ORIGIN_FILE = DATASET_DIR + "/ratings.dat"
        DATASET_FILE = DATASET_DIR + "/ratings.csv" # ratings.csv
        MAX_SESSION_LENGTH = 200 # maximum number of events in a session
        
    elif dataset == Steam:
        # Create steam folder
        isExists_file = os.path.exists(DATASET_DIR + '/steam')
        if not isExists_file: os.makedirs(DATASET_DIR + '/steam')
        # Move file
        if not file_exists(DATASET_DIR + '/steam/steam_new.json'): shutil.move(DATASET_DIR + '/steam_new.json', DATASET_DIR + '/steam')
        # Fetch the correct file location
        DATASET_DIR = DATASET_DIR  + "/steam"
        DATASET_FILE = DATASET_DIR + '/steam_new.json'
        MAX_SESSION_LENGTH = 15 # maximum number of events in a session
        
    # Create pickle folder
    isExists_file = os.path.exists(DATASET_DIR + '/pickle')
    if not isExists_file: os.makedirs(DATASET_DIR + '/pickle') 
    DATASET_W_CONVERTED_TIMESTAMPS = DATASET_DIR + "/pickle" + '/1_converted_timestamps.pickle'
    DATASET_USER_ARTIST_MAPPED = DATASET_DIR + "/pickle" + '/2_user_artist_mapped.pickle'
    DATASET_USER_SESSIONS_1 = DATASET_DIR + "/pickle" + '/3_user_sessions_1.pickle'
    DATASET_USER_SESSIONS_2 = DATASET_DIR + "/pickle" + '/3_user_sessions_2.pickle'
    DATASET_TRAIN_TEST_SPLIT = DATASET_DIR + "/pickle" + '/4_train_test_split.pickle'
    DATASET_TRAIN_TEST_SPLIT_2 = DATASET_DIR + "/pickle" + '/4_train_test_split_sample.pickle' # Sample name
    DATASET_BPR_MF = DATASET_DIR + "/pickle" + '/bpr-mf_train_test_split.pickle'
    if dataset == MovieLens_1M: SESSION_TIMEDELTA = 60*60*24 # 1 hours 60*30
    elif dataset == Steam: SESSION_TIMEDELTA = 60*60*24 # 1 hours 60*30

def file_exists(filename):
    return os.path.isfile(filename)

def save_pickle(data_object, data_file):
    pickle.dump(data_object, open(data_file, 'wb'))

def dataset_format_conversion():
    # Dat to CSV
    with open(DATASET_ORIGIN_FILE) as dat_file, open(DATASET_FILE, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for line in dat_file:
            row = [field.strip() for field in line.split('::')]
            csv_writer.writerow(row)
            # Remove unusual symbols (,)
            for i in range(0,len(row)): row[i] = row[i].replace(",","")
    # Remove blank lines        
    data = pd.read_csv(DATASET_FILE)
    data = data.dropna(how="all").to_csv(DATASET_FILE, index=False)

def convert_timestamps():
    dataset_list = []
    with open(DATASET_FILE, 'rt', buffering=10000, encoding='utf8') as datasets:
        for line in datasets:
            if dataset == MovieLens_1M:
                line = line.rstrip().strip("\n").split(",") 
                user_id = line[0]; item = line[1]; timestamp = float(line[3]); 
                dataset_list.append( [user_id, timestamp, item] )
            elif dataset == Steam:
                trans = eval(line)
                user_id = trans['username']; item = trans['product_id']; timestamp = trans['date']; 
                struct_time = time.strptime(timestamp, "%Y-%m-%d") # Convert to time tuple
                timestamp = int(time.mktime(struct_time)) # Convert to timestamp
                dataset_list.append( [user_id, timestamp, item] )
    save_pickle(dataset_list, DATASET_W_CONVERTED_TIMESTAMPS)
    
def map_user_and_artist_id_to_labels():
    dataset_list = load_pickle(DATASET_W_CONVERTED_TIMESTAMPS)
    artist_map = {}; user_map = {}; artist_id = ''; user_id = ''; 
    for i in range(len(dataset_list)):
        user_id = dataset_list[i][0]
        artist_id = dataset_list[i][2]
        if user_id not in user_map: user_map[user_id] = len(user_map)
        if artist_id not in artist_map: artist_map[artist_id] = len(artist_map)
        dataset_list[i][0] = user_map[user_id]
        dataset_list[i][2] = artist_map[artist_id]
    # list to dataframe 
    dataset_list = pd.DataFrame(dataset_list) 
    # Save to pickle file
    save_pickle(dataset_list, DATASET_USER_ARTIST_MAPPED)

def reduce_ineligible_interactions_and_history():
    start_time = time.time()
    # load data
    print("- load data")
    dataset = load_pickle(DATASET_USER_ARTIST_MAPPED)
    update_dataset = dataset
    update_dataset.columns = ['User_ID', 'timestamp', 'Item_ID']
    # filter out tiny items
    print("- filter out tiny items")
    df_iid2ucount = update_dataset.groupby('Item_ID').size()
    survived_iids = df_iid2ucount.index[df_iid2ucount >= MIN_USER_COUNT_PER_ITEM]
    update_dataset = update_dataset[update_dataset['Item_ID'].isin(survived_iids)]
    # filter out tiny users
    print("- filter out tiny users")
    df_uid2icount = update_dataset.groupby('User_ID').size()
    survived_uids = df_uid2icount.index[df_uid2icount >= MIN_ITEM_COUNT_PER_USER]
    update_dataset = update_dataset[update_dataset['User_ID'].isin(survived_uids)]
    end_time = (time.time() - start_time)/60
    end_time = "%.2f" % end_time
    print("Complete | Before: "+str(len(dataset))+" pieces of data | After: "+str(len(update_dataset))+" pieces of data | Runtime Spend: "+str(end_time)+" min")
    save_pickle(update_dataset, DATASET_USER_SESSIONS_1)

def cut_and_assign_sids_to_rows(rows):
    sid = 0; uid2rows = {}; 
    for uid, timestamp, iid in tqdm(rows, desc="* organize uid2rows"):
        if uid not in uid2rows: uid2rows[uid] = []
        uid2rows[uid].append((iid, timestamp))
    rows = []
    uids = list(uid2rows.keys())
    for uid in tqdm(uids, desc="* cutting"):
        user_rows = sorted(uid2rows[uid], key=itemgetter(1))
        tba = []; sid2count = {}; 
        if MAX_SEQUENCE_LENGTH: user_rows = user_rows[-MAX_SEQUENCE_LENGTH:]
        sid += 1
        _, previous_timestamp = user_rows[0]
        for iid, timestamp in user_rows:
            if timestamp - previous_timestamp > SESSION_WINDOW: sid += 1
            tba.append((uid, timestamp, iid))
            sid2count[sid] = sid2count.get(sid, 0) + 1
            previous_timestamp = timestamp
        if MIN_SESSION_COUNT_PER_USER and len(sid2count) < MIN_SESSION_COUNT_PER_USER: continue
        if MIN_ITEM_COUNT_PER_SESSION and min(sid2count.values()) < MIN_ITEM_COUNT_PER_SESSION: continue
        rows.extend(tba)
    return rows

def split_single_session(session):
    splitted = [session[i:i+MAX_SESSION_LENGTH] for i in range(0, len(session), MAX_SESSION_LENGTH)]
    return splitted

def perform_session_splits(sessions):
    splitted_sessions = []
    for session in sessions: splitted_sessions += split_single_session(session)
    return splitted_sessions

def split_long_sessions(user_sessions):
    for k, v in user_sessions.items(): user_sessions[k] = perform_session_splits(v)

def sort_and_split_user_sessions():
    # load data
    print("- load data"); dataset = load_pickle(DATASET_USER_SESSIONS_1); 
    # cut and assign sid
    print("- cut and assign sid"); rows = cut_and_assign_sids_to_rows(dataset.values); 
    dataset = pd.DataFrame(rows)
    dataset.columns = ['User_ID', 'timestamp', 'Item_ID']
    # map uid -> uindex
    print("- map uid -> uindex"); uids = set(dataset['User_ID']); 
    uid2uindex = {uid: index for index, uid in enumerate(set(uids), start=1)}
    dataset['uindex'] = dataset['User_ID'].map(uid2uindex)
    dataset = dataset.drop(columns=['User_ID'])
    with open(os.path.join(DATASET_DIR + "/pickle", 'uid2uindex.pkl'), 'wb') as fp: pickle.dump(uid2uindex, fp)
    # map iid -> iindex
    print("- map iid -> iindex"); iids = set(dataset['Item_ID']); 
    iid2iindex = {iid: index for index, iid in enumerate(set(iids), start=1)}
    dataset['iindex'] = dataset['Item_ID'].map(iid2iindex)
    dataset = dataset.drop(columns=['Item_ID'])
    with open(os.path.join(DATASET_DIR + "/pickle", 'iid2iindex.pkl'), 'wb') as fp: pickle.dump(iid2iindex, fp)
    # df to lsit
    time = list(dataset['timestamp']); user = list(dataset['uindex']); item = list(dataset['iindex']); 
    temp_datalist = []
    for i in range(len(dataset)): temp_datalist.append([user[i], time[i], item[i]])
    # list to dict
    user_sessions = {}; current_session = []; 
    for event in temp_datalist:
        user_id = event[0]; timestamp = event[1]; artist = event[2]; 
        new_event = [timestamp, artist]
        # if new user -> new session
        if user_id not in user_sessions:
            user_sessions[user_id] = []; current_session = []; 
            user_sessions[user_id].append(current_session)
            current_session.append(new_event)
            continue
        # it is an existing user: is it a new session?
        # we also know that the current session contains at least one event
        # NB: Dataset is presorted from newest to oldest events
        last_event = current_session[-1]
        last_timestamp = last_event[0]
        timedelta = timestamp - last_timestamp
        if timedelta < SESSION_TIMEDELTA: current_session.append(new_event) # new event belongs to current session
        # new event belongs to new session
        else: current_session = [new_event]; user_sessions[user_id].append(current_session); 
    split_long_sessions(user_sessions)
    save_pickle(user_sessions, DATASET_USER_SESSIONS_2)

def get_session_lengths(dataset):
    session_lengths = {}
    for k, v in dataset.items():
        session_lengths[k] = []
        for session in v: session_lengths[k].append(len(session)-1)
    return session_lengths

def create_padded_sequence(session):
    if len(session) == MAX_SESSION_LENGTH: return session
    dummy_timestamp = 0; dummy_label = 0; 
    length_to_pad = MAX_SESSION_LENGTH - len(session)
    padding = [[dummy_timestamp, dummy_label]] * length_to_pad
    session += padding
    return session

def pad_sequences(dataset):
    for k, v in dataset.items():
        for session_index in range(len(v)): dataset[k][session_index] = create_padded_sequence(dataset[k][session_index])

# Splits the dataset into a training and a testing set, by extracting the last
# sessions of each user into the test set
def split_to_training_and_testing():
    dataset = load_pickle(DATASET_USER_SESSIONS_2)
    trainset = {}; testset = {}; 
    for k, v in dataset.items():
        n_sessions = len(v)
        split_point = int(0.8*n_sessions)
        # runtime check to ensure that we have enough sessions for training and testing
        if split_point < 1: #2
            raise ValueError('User '+str(k)+' with '+str(n_sessions)+""" sessions, 
                resulted in split_point: '+str(split_point)+' which gives too 
                few training sessions. Please check that data and preprocessing 
                is correct.""")
        trainset[k] = v[:split_point]; testset[k] = v[split_point:]; 
    # Also need to know session lengths for train- and testset
    train_session_lengths = get_session_lengths(trainset)
    test_session_lengths = get_session_lengths(testset)
    # Finally, pad all sequences before storing everything
    pad_sequences(trainset); pad_sequences(testset); 
    # Put everything we want to store in a dict, and just store the dict with pickle
    pickle_dict = {}
    pickle_dict['trainset'] = trainset; pickle_dict['testset'] = testset; 
    pickle_dict['train_session_lengths'] = train_session_lengths; pickle_dict['test_session_lengths'] = test_session_lengths; 
    save_pickle(pickle_dict , DATASET_TRAIN_TEST_SPLIT)

def split_to_training_and_testing_sample():
    dataset = load_pickle(DATASET_USER_SESSIONS_2)
    count = 0; split_sample = {}; 
    for k, v in dataset.items():
        split_sample[k] = v; count += 1; 
        if count == 100: break
    trainset = {}; testset = {}; 
    for k, v in split_sample.items():
        n_sessions = len(v)
        split_point = int(0.8*n_sessions)
        # runtime check to ensure that we have enough sessions for training and testing
        if split_point < 1: #2
            raise ValueError('User '+str(k)+' with '+str(n_sessions)+""" sessions, 
                resulted in split_point: '+str(split_point)+' which gives too 
                few training sessions. Please check that data and preprocessing 
                is correct.""")
        trainset[k] = v[:split_point]; testset[k] = v[split_point:]; 
    # Also need to know session lengths for train- and testset
    train_session_lengths = get_session_lengths(trainset); 
    test_session_lengths = get_session_lengths(testset); 
    # Finally, pad all sequences before storing everything
    pad_sequences(trainset); pad_sequences(testset)
    # Put everything we want to store in a dict, and just store the dict with pickle
    pickle_dict = {}
    pickle_dict['trainset'] = trainset; pickle_dict['testset'] = testset; 
    pickle_dict['train_session_lengths'] = train_session_lengths; pickle_dict['test_session_lengths'] = test_session_lengths; 
    save_pickle(pickle_dict , DATASET_TRAIN_TEST_SPLIT_2)

def create_bpr_mf_sets():
    p = load_pickle(DATASET_TRAIN_TEST_SPLIT)
    train = p['trainset']; train_sl = p['train_session_lengths']; 
    test = p['testset']; test_sl = p['test_session_lengths']; 
    for user in train.keys():
        extension = test[user][:-1]
        train[user].extend(extension)
        extension = test_sl[user][:-1]
        train_sl[user].extend(extension)
    for user in test.keys():
        test[user] = [test[user][-1]]
        test_sl[user] = [test_sl[user][-1]]
    pickle_dict = {}
    pickle_dict['trainset'] = train; pickle_dict['testset'] = test; 
    pickle_dict['train_session_lengths'] = train_sl; pickle_dict['test_session_lengths'] = test_sl; 
    save_pickle(pickle_dict , DATASET_BPR_MF)

def parse_args():
    parse = ArgumentParser()
    parse.add_argument("-d", "--Dataset", help = "give a dataset | ex: 'Steam / MovieLens_1M / MovieLens_20M'", default = "Steam", type = str)
    args = parse.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args() # Get input
    dataset = args.Dataset # Get dataset
    check_dataset_path(dataset)
    if not file_exists(DATASET_FILE): print("Dataset format conversion."); dataset_format_conversion(); 
    if not file_exists(DATASET_W_CONVERTED_TIMESTAMPS): print("Converting timestamps."); convert_timestamps(); 
    if not file_exists(DATASET_USER_ARTIST_MAPPED): print("Mapping user and artist IDs to labels."); map_user_and_artist_id_to_labels(); 
    if not file_exists(DATASET_USER_SESSIONS_1): print("Reduce ineligible interactions and history."); reduce_ineligible_interactions_and_history();         
    if not file_exists(DATASET_USER_SESSIONS_2): print("Sorting sessions to users."); sort_and_split_user_sessions(); 
    if not file_exists(DATASET_TRAIN_TEST_SPLIT): print("Splitting dataset into training and testing sets."); split_to_training_and_testing(); 
    if not file_exists(DATASET_TRAIN_TEST_SPLIT_2): print("Splitting dataset into training and testing sets to Sample Data."); split_to_training_and_testing_sample(); 
    if not file_exists(DATASET_BPR_MF): print("Creating dataset for BPR-MF."); create_bpr_mf_sets(); 
    end_time = (time.time() - runtime)/60
    end_time = "%.2f" % end_time
    print("Runtime Spend:", str(end_time), "min")