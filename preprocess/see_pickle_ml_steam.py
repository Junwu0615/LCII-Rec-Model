import os
import time
import pickle
from argparse import ArgumentParser

def load_pickle(pickle_file):
    return pickle.load(open(pickle_file, 'rb'))

def parse_args():
    parse = ArgumentParser()
    parse.add_argument("-d", "--Dataset", help = "give a dataset | ex: 'Steam / MovieLens-1M'", 
                       default = "Steam", type = str)
    args = parse.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args() # Get input
    dataset = args.Dataset # Get dataset
    home = (os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir)))+"./datasets"
    if dataset == 'MovieLens-1M': dataset = "ml-1m"; DATASET_FILE = home + "/" + dataset + '/ratings.csv'; 
    elif dataset == 'Steam': DATASET_FILE = home + "/" + dataset + '/steam_new.json' # steam_reviews.json
    Dataset_Name = dataset
    DATASET_DIR = home + "/" + dataset + "/pickle"
    pickle_converted = DATASET_DIR + '/1_converted_timestamps.pickle' 
    pickle_mapped = DATASET_DIR + '/2_user_artist_mapped.pickle' 
    pickle_sessions_1 = DATASET_DIR + '/3_user_sessions_1.pickle'
    pickle_sessions_2 = DATASET_DIR + '/3_user_sessions_2.pickle' 
    pickle_train_test_split = DATASET_DIR + '/4_train_test_split.pickle' 
    pickle_train_test_split_smaple = DATASET_DIR + '/4_train_test_split_sample.pickle' 
    pickle_converted = load_pickle(pickle_converted)
    pickle_mapped = load_pickle(pickle_mapped)
    pickle_sessions_1 = load_pickle(pickle_sessions_1)
    pickle_sessions_2 = load_pickle(pickle_sessions_2)
    pickle_train_test_split = load_pickle(pickle_train_test_split)
    pickle_train_test_split_sample = load_pickle(pickle_train_test_split_smaple)
    
    with open(DATASET_FILE, 'rt', buffering=10000, encoding='utf8') as dataset:
        count = 0
        for i in dataset:
            print(i); count += 1; 
            if count == 1: print(); break; 

    ## pickle_converted (list)
    type(pickle_converted)
    count = 0
    for i in pickle_converted:
        print(i); count += 1; 
        if count == 1: print(); break; 

    ## pickle_mapped (DataFrame)
    type(pickle_mapped)
    pickle_mapped
    
    ## pickle_sessions_1 (DataFrame)
    type(pickle_sessions_1)
    len(pickle_sessions_1)
    pickle_sessions_1
    item = pickle_sessions_1['Item_ID'].values
    time = pickle_sessions_1['timestamp'].values
    user = pickle_sessions_1['User_ID'].values
    dict_ = {}
    for i in range(len(pickle_sessions_1)):
        if user[i] not in dict_: dict_[user[i]]=[]
        dict_[user[i]].append([time[i], item[i]])
    u_count = 0; i_count = 0; 
    for k, v in dict_.items():
        u_count +=1; i_count = i_count + len(v); 
    print("user : %.2f" %u_count)
    print("items: %.2f" %i_count)

    ## pickle_sessions_2 (Dict)
    type(pickle_sessions_2)
    u_count = 0; i_count = 0; 
    for k, v in pickle_sessions_2.items():
        u_count +=1; i_count = i_count + len(v); 
    print("user : %.2f" %u_count)
    print("items: %.2f" %i_count)
    print(list(pickle_sessions_2.items())[0:1])
    count = 0
    for a, b in pickle_sessions_2.items():
        for bb in b:
            print(bb); count += 1; 
            if count == 1: break
        break

    print("\nInput data status")
    count_action_to_user = []; count_sess_to_user = []; count_act_to_sess = []; 
    for user, session in pickle_sessions_2.items():
        count_sess_to_user.append(len(session))
        temp_c = 0
        for action in session:
            temp_c += len(action)
            count_act_to_sess.append(len(action))
        count_action_to_user.append(temp_c)
    print("Data length of each user has x action (Should be consistent with the num of users):",len(count_action_to_user))
    print("Data length of each user has x session (Should be consistent with the num of users):",len(count_sess_to_user))
    print("Data length of each session has x action (Should be consistent with the num of sessions):",len(count_act_to_sess))
    print("\nInterquartile Range")
    count_action_to_user = sorted(count_action_to_user, reverse = False)
    Q1 = int(len(count_action_to_user)/4*1); Q2 = int(len(count_action_to_user)/4*2); Q3 = int(len(count_action_to_user)/4*3); 
    print("Each user has x action | Q1:",count_action_to_user[Q1],"Q2:",count_action_to_user[Q2],"Q3:",count_action_to_user[Q3])
    count_sess_to_user = sorted(count_sess_to_user, reverse = False)
    Q1 = int(len(count_sess_to_user)/4*1); Q2 = int(len(count_sess_to_user)/4*2); Q3 = int(len(count_sess_to_user)/4*3); 
    print("Each user has x session | Q1:",count_sess_to_user[Q1],"Q2:",count_sess_to_user[Q2],"Q3:",count_sess_to_user[Q3])
    count_act_to_sess = sorted(count_act_to_sess, reverse = False)
    Q1 = int(len(count_act_to_sess)/4*1); Q2 = int(len(count_act_to_sess)/4*2); Q3 = int(len(count_act_to_sess)/4*3); 
    print("Each session has x action | Q1:",count_act_to_sess[Q1],"Q2:",count_act_to_sess[Q2],"Q3:",count_act_to_sess[Q3])
    
    ## Fully data
    trainset = pickle_train_test_split['trainset']
    testset = pickle_train_test_split['testset']
    train_session_lengths = pickle_train_test_split['train_session_lengths']
    test_session_lengths = pickle_train_test_split['test_session_lengths']
    user_count = 0; items_in_sess = 0; session_nums_train = 0; session_nums_test = 0; pading_count = 0; 
    User = {}; Session = {}; item_type = {}; 
    for user, user_sess in trainset.items():
        if user not in User: User[user] = 0; user_count +=1; 
        User[user] += 1
        count_sess_len = len(user_sess)  
        if count_sess_len not in Session: Session[count_sess_len] = 0; 
        Session[count_sess_len] += 1
        session_nums_train = session_nums_train + len(user_sess)
        for each_session in user_sess:
            items_in_sess = items_in_sess + len(each_session)
            for i in each_session:
                if i[1] not in item_type: item_type[i[1]] = True; 
                if i == [0, 0]: pading_count += 1; 
    for user, user_sess in testset.items():
        if user not in User: User[user] = 0; user_count +=1; 
        User[user] += 1
        count_sess_len = len(user_sess)  
        if count_sess_len not in Session: Session[count_sess_len] = 0
        Session[count_sess_len] += 1
        session_nums_test = session_nums_test + len(user_sess)
        for each_session in user_sess:
            items_in_sess = items_in_sess + len(each_session)
            for i in each_session:
                if i[1] not in item_type: item_type[i[1]] = True
                if i == [0, 0]: pading_count += 1

    session_nums = session_nums_train+session_nums_test
    pading_proportion = pading_count/items_in_sess*100
    pading_proportion = "%.2f" %pading_proportion
    print("\n\n --  The following data contains <" + Dataset_Name +">: train & test of full statistics")
    print(" --  Session length distribution in the data set [Session length : num]:  " + str(Session))
    print(" ------------------------------------------------------------------------------------------------------ ")
    print(" --  Total user | "+str(user_count))
    print(" --  session num of train | "+str(session_nums_train))
    print(" --  session num of test | "+str(session_nums_test))
    print(" --  Total session | "+str(session_nums))
    print(" --  Total interaction types | "+str(len(item_type)))
    print(" --  Total interaction records | "+str(items_in_sess))
    print(" --  Total interaction records (Deduction of missing value) | "+str(items_in_sess-pading_count))
    print(" --  Each user has x item | "+str(round(items_in_sess/user_count,2)))
    print(" --  Each user has x item (Deduction of missing value) | "+str(round((items_in_sess-pading_count)/user_count,2)))
    print(" --  Each user has x session | "+str(round((session_nums/user_count),2)))
    print(" --  Each session has x item | "+str(round((items_in_sess-pading_count)/session_nums,2)))
    print(" --  Each session has x actions | "+str(items_in_sess/session_nums))
    print(" --  Total num of missing value [0,0] | "+str(pading_count))
    print(" --  The proportion of missing values ​​in the overall item | "+ str(pading_proportion) +"%")
    
    ## sample
    trainset = pickle_train_test_split_sample['trainset']
    testset = pickle_train_test_split_sample['testset']
    train_session_lengths = pickle_train_test_split_sample['train_session_lengths']
    test_session_lengths = pickle_train_test_split_sample['test_session_lengths']
    user_count = 0; items_in_sess = 0; session_nums_train = 0; session_nums_test = 0; pading_count = 0; 
    User = {}; Session = {}; item_type = {}
    for user, user_sess in trainset.items():
        if user not in User: User[user] = 0; user_count +=1; 
        User[user] += 1
        count_sess_len = len(user_sess)  
        if count_sess_len not in Session: Session[count_sess_len] = 0
        Session[count_sess_len] += 1
        session_nums_train = session_nums_train + len(user_sess)
        for each_session in user_sess:
            items_in_sess = items_in_sess + len(each_session)
            for i in each_session:
                if i[1] not in item_type: item_type[i[1]] = True
                if i == [0, 0]: pading_count += 1
    for user, user_sess in testset.items():
        if user not in User: User[user] = 0; user_count +=1; 
        User[user] += 1
        count_sess_len = len(user_sess)  
        if count_sess_len not in Session: Session[count_sess_len] = 0
        Session[count_sess_len] += 1
        session_nums_test = session_nums_test + len(user_sess)
        for each_session in user_sess:
            items_in_sess = items_in_sess + len(each_session)
            for i in each_session:
                if i[1] not in item_type: item_type[i[1]] = True
                if i == [0, 0]: pading_count += 1
    session_nums = session_nums_train+session_nums_test
    pading_proportion = pading_count/items_in_sess*100
    pading_proportion = "%.2f" %pading_proportion
    print("\n\n --  The following data contains <" + Dataset_Name +">: train & test of sample statistics")
    print(" --  Session length distribution in the data set [Session length : num]:  " + str(Session))
    print(" ------------------------------------------------------------------------------------------------------ ")
    print(" --  Total user | "+str(user_count))
    print(" --  session num of train | "+str(session_nums_train))
    print(" --  session num of test | "+str(session_nums_test))
    print(" --  Total session | "+str(session_nums))
    print(" --  Total interaction types | "+str(len(item_type)))
    print(" --  Total interaction records | "+str(items_in_sess))
    print(" --  Total interaction records (Deduction of missing value) | "+str(items_in_sess-pading_count))
    print(" --  Each user has x item | "+str(round(items_in_sess/user_count,2)))
    print(" --  Each user has x item (Deduction of missing value) | "+str(round((items_in_sess-pading_count)/user_count,2)))
    print(" --  Each user has x session | "+str(round((session_nums/user_count),2)))
    print(" --  Each session has x item | "+str(round((items_in_sess-pading_count)/session_nums,2)))
    print(" --  Each session has x actions | "+str(items_in_sess/session_nums))
    print(" --  Total num of missing value [0,0] | "+str(pading_count))
    print(" --  The proportion of missing values ​​in the overall item | "+ str(pading_proportion) +"%")