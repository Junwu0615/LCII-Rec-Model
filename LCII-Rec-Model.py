import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
#from tensorflow.contrib import rnn
#from tensorflow.contrib import layers
import tf_slim as layers
import os
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils_Integrated import IIRNNDataHandler

def parse_args():
    parse = ArgumentParser()
    parse.add_argument("-d", "--Dataset", help = "give a dataset | ex: 'Amazon / Steam / MovieLens-1M'", 
                       default = "MovieLens-1M", type = str)   
    parse.add_argument("-sp", "--Switch_Plot", help = "give a switch plot | ex: 'sum / dot / attention_gate_sum / attention_gate_dot'", 
                       default = "attention_gate_dot", type = str)
    parse.add_argument("-sis", "--Switch_Initial_State", help = "give a switch initial state | ex: 'True / False'", 
                       default = "True", type = str)
    parse.add_argument("-fw", "--Fusion_Way", help = "give a fusion way | ex: 'att / lp / fix / none'", 
                       default = "fix", type = str)
    parse.add_argument("-s", "--Strategy", help = "give a strategy | ex: 'pre-combine / post-combine / original'", 
                       default = "post-combine", type = str)
    parse.add_argument("-w", "--Window", help = "give a window number | ex: '0-100' / 'no_use'", 
                       default = 4, type = str)
    parse.add_argument("-ls", "--Long_Score", help = "give a long-score | ex: '0.0-1.0' / 'no_use'", 
                       default = 0.8, type = str)
    parse.add_argument("-ss", "--Short_Score", help = "give a short-score | ex: '0.0-1.0' / 'no_use'", 
                       default = 0.2, type = str)
    parse.add_argument("-es", "--Embedding_Size", help = "give a embedding-size | ex: '30 / 50 / 80 / 100 / 200 / 300 / 500 / 800 / 1000'", 
                       default = 80, type = int)
    parse.add_argument("-bs", "--Batch_Size", help = "give a batch-size | ex: '16 / 32 / 64 / 100 / 128 / 256 / 512'", 
                       default = 100, type = int)
    parse.add_argument("-lr", "--Learning_Rate", help = "give a learning-rate | ex: '0.001 / 0.01 / ...'", 
                       default = 0.01, type = float)
    parse.add_argument("-dr", "--Dropout", help = "give a dropout | ex: '0.8'", 
                       default = 0.8, type = float)
    parse.add_argument("-me", "--Max_Epoch", help = "give a max-epoch | ex: '100 / 200 / ...'", 
                       default = 200, type = int)
    parse.add_argument("-t", "--Threshold", help = "give a threshold | ex: '98'", 
                       default = 98, type = int)
    parse.add_argument("-add", "--use_FC", help = "whether to add FC to input_sum | ex: 'True / False'", 
                       default = "False", type = str)
    args = parse.parse_args()
    return args

# Generate Trend Chart Data
def png(df, epoch, value_1, value_2, value_3, value_4, value_5):
    fig = plt.figure(figsize=(6,4))
    fig.patch.set_facecolor('white')
    plt.rcParams['font.sans-serif'] = ['Yu Gothic']; plt.grid(True); 
    if epoch == "loss_epoch":
        plt.plot(df[epoch], df[value_1])
        plt.plot(df[epoch], df[value_2])
        plt.legend([value_1, value_2], loc="upper left")
    elif epoch == "train_epoch":
        plt.plot(df[epoch], df[value_1])
        plt.plot(df[epoch], df[value_2])
        plt.plot(df[epoch], df[value_3])
        plt.legend([value_1, value_2, value_3], loc="upper left")
    elif epoch == "test_epoch":  
        plt.plot(df[epoch], df[value_1])
        plt.plot(df[epoch], df[value_2])
        plt.plot(df[epoch], df[value_3])
        plt.legend([value_1, value_2, value_3], loc="upper left")
    plt.xlabel("Epoch")
    if epoch == "loss_epoch": plt.ylabel("loss")
    else: plt.ylabel("%")
    plt.title(value_5); plt.savefig("./testlog/"+ value_4 + ".png"); plt.clf(); 
    
def integrated_computing(switch_plot, input_1, input_2):
    if switch_plot == "sum": input_sum = input_1 + input_2
    elif switch_plot == "dot": input_sum = input_1 * input_2
    elif switch_plot == "attention_gate_sum":
        a = tf.math.tanh(input_1); b = tf.math.tanh(input_2); 
        a = tf.nn.softmax(a); b = tf.nn.softmax(b); 
        input_sum = (a/(a+b)*input_1) + (b/(a+b)*input_2)
    elif switch_plot == "attention_gate_dot":
        w_input_1 = tf.math.tanh(input_1); w_input_2 = tf.math.tanh(input_2); 
        w_input_1 = tf.nn.softmax(w_input_1); w_input_2 = tf.nn.softmax(w_input_2); 
        w_input_sum = w_input_1+w_input_2
        input_sum = ((w_input_1/w_input_sum)*input_1) * ((w_input_2/w_input_sum)*input_2)
    return input_sum

if __name__ == "__main__":
    args = parse_args() # Get input
    print("==========================================\n")
    code_runtime_now = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    runtime = time.time()
    tf.reset_default_graph()
    print(f"LCII Program Starts Running | Current Time: {code_runtime_now}")
    dataset = args.Dataset; switch_plot = args.Switch_Plot; switch_initial_state = args.Switch_Initial_State; 
    if switch_initial_state == "True": switch_initial_state = True 
    else: switch_initial_state = False
    fusion_way = args.Fusion_Way; strategy = args.Strategy; 
    window = args.Window; 
    if window != 'no_use': window = float(window)
    long_score = args.Long_Score; 
    if long_score != 'no_use': long_score = float(long_score)
    short_score = args.Short_Score; 
    if short_score != 'no_use': short_score = float(short_score)
    EMBEDDING_SIZE = INNER_RNN_SIZE = OUTER_RNN_SIZE = args.Embedding_Size
    BATCHSIZE = args.Batch_Size # How many pieces of data to read at one time
    learning_rate = args.Learning_Rate; dropout_pkeep = args.Dropout; MAX_EPOCHS = args.Max_Epoch; 
    MAX_SESSION_REPRESENTATIONS = 0 # Won't use it
    Threshold = args.Threshold # If recall@5 > Threshold, it will be determined as overfitting.
    use_FC = args.use_FC; temp_loss = 1000000000000000 # Prevent the loss from rising for the first time
    if use_FC == "True": use_FC = True 
    else: use_FC = False
    # Give name to log_file_name
    if fusion_way == 'none': log_file_name = (f"LCII_ver_{strategy}_dropout_{dropout_pkeep}")
    else:
        if fusion_way == 'fix':  log_file_name = (f"LCII_ver_{strategy}_w-{window}_{fusion_way}_L-{long_score}_S-{short_score}_dropout_{dropout_pkeep}")
        else: log_file_name = (f"LCII_ver_{strategy}_w-{window}_{fusion_way}_dropout_{dropout_pkeep}")
    if use_FC == True: log_file_name = (f"{log_file_name}_FC")
    do_training = True
    save_best = True
    isExists_file = os.path.exists('./testlog')
    if not isExists_file: os.makedirs('./testlog') 
    if dataset == "Amazon":
        dataset = "amazon"
        dataset_path = './datasets/'+dataset+'/4_train_test_split.pickle'
        #dataset_path = './datasets/'+dataset+'/4_train_test_split_sample.pickle'
        from test_util_len20 import Tester; SEQLEN = 20-1; 

    elif dataset == "MovieLens-1M":
        dataset = "ml-1m"
        dataset_path = './datasets/'+dataset+'/pickle/4_train_test_split.pickle'
        #dataset_path = './datasets/'+dataset+'/pickle/4_train_test_split_sample.pickle'
        from test_util_len200 import Tester; SEQLEN = 200-1; 
        
    elif dataset == "Steam":
        dataset = "steam"
        dataset_path = './datasets/'+dataset+'/pickle/4_train_test_split.pickle'
        #dataset_path = './datasets/'+dataset+'/pickle/4_train_test_split_sample.pickle'
        from test_util_len15 import Tester; SEQLEN = 15-1; 
        
    epoch_file = './epoch_file-'+log_file_name+'-'+dataset+'.pickle'
    checkpoint_file = './checkpoints/'+log_file_name+'-'+dataset+'-'
    checkpoint_file_ending = '.ckpt'
    date_now = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
    log_file = (f"./Testlog/{date_now}-testing-Integrated_log_preference.txt")
    seed = 0; tf.set_random_seed(seed)
    N_ITEMS  = -1; TOP_K = 20; 
    N_LAYERS = 1 # number of layers in the rnn

    # Load training data
    datahandler = IIRNNDataHandler(dataset_path, BATCHSIZE, log_file, MAX_SESSION_REPRESENTATIONS, INNER_RNN_SIZE)
    N_ITEMS = datahandler.get_num_items() 
    N_SESSIONS = datahandler.get_num_training_sessions()
    message = "------------------------------------------\n"
    message += (f"DATASET: {dataset} | MODEL: {log_file_name}\n")
    message += (f"SWITCH_INITIAL_STATE: {switch_initial_state} | USE_FC_IN_INPUT-SUM: {use_FC}\n")
    message += (f"STRATEGY: {strategy} | FUSION_WAY: {fusion_way} | SWITCH_PLOT: {switch_plot}\n")
    message += (f"ADJACENT_WINDOW_RATIO(0-100%): {window} | LONG_TERM_SCORE: {long_score} | SHORT_TERM_SCORE: {short_score}\n")
    message += (f"DROPOUT: {dropout_pkeep} | LEARNING_RATE: {learning_rate} | BATCHSIZE: {BATCHSIZE} | SEED: {seed}\n")
    message += (f"OUTER_RNN_SIZE: {OUTER_RNN_SIZE} | INNER_RNN_SIZE: {INNER_RNN_SIZE} | EMBEDDING_SIZE: {EMBEDDING_SIZE}\n")
    message += (f"TRAIN_N_SESSIONS: {N_SESSIONS} | N_ITEMS: {N_ITEMS} | N_LAYERS: {N_LAYERS} | SEQLEN: {SEQLEN} | MAX_EPOCHS: {MAX_EPOCHS}\n")
    message += ("MAX_SESSION_REPRESENTATIONS: {MAX_SESSION_REPRESENTATIONS}\n")
    message += "------------------------------------------\n"
    datahandler.log_config(message)
    if not do_training: print("\nOBS!!! Training is turned off!\n")
    ##
    ## The model
    ##
    print("Creating model")
    # Inputs
    X = tf.placeholder(tf.int32, [None, None], name='X') # [ BATCHSIZE, SEQLEN ]
    Y_ = tf.placeholder(tf.int32, [None, None], name='Y_') # [ BATCHSIZE, SEQLEN ]
    # Embeddings. W_embed = all embeddings. X_embed = retrieved embeddings 
    # from W_embed, corresponding to the items in the current batch
    W_embed = tf.Variable(tf.random_uniform([N_ITEMS, EMBEDDING_SIZE], -1.0, 1.0), name='embeddings')
    X_embed = tf.nn.embedding_lookup(W_embed, X) # [BATCHSIZE, SEQLEN, EMBEDDING_SIZE]
    # Length of sesssions (not considering padding)
    seq_len = tf.placeholder(tf.int32, [None], name='seqlen')
    batchsize = tf.placeholder(tf.int32, name='batchsize')
    lr = tf.placeholder(tf.float32, name='lr') # learning rate
    pkeep = tf.placeholder(tf.float32, name='pkeep') # dropout parameter

    # Inner_RNN
    Inner_cell = tf.nn.rnn_cell.GRUCell(INNER_RNN_SIZE)
    Inner_dropcell = tf.nn.rnn_cell.DropoutWrapper(Inner_cell, input_keep_prob=pkeep, output_keep_prob=pkeep)
    Inner_outputs, Inner_states = tf.nn.dynamic_rnn(Inner_dropcell, X_embed, sequence_length=seq_len, dtype=tf.float32)

    if strategy == 'original': input_sum = integrated_computing(switch_plot, X_embed, Inner_outputs)
    else:
        # Long and short term processing
        # h_list.append(tf.nn.relu(Inner_outputs))#Relu
        if int((SEQLEN+1) * window * 0.01) < 1: capture_len_h_list = -1
        else: capture_len_h_list = int((SEQLEN+1) * -window * 0.01)  
        short_sum = Inner_outputs[:, capture_len_h_list:]
        long_sum  = Inner_outputs[:, :capture_len_h_list]   
        if fusion_way == "lp":
            # Learn long and short term weights
            W_embed_Lp_long = tf.Variable(tf.random_uniform([1, SEQLEN+capture_len_h_list, OUTER_RNN_SIZE], -1.0, 1.0), name='embeddings_Lp_long')
            W_embed_Lp_short = tf.Variable(tf.random_uniform([1, -capture_len_h_list, OUTER_RNN_SIZE], -1.0, 1.0), name='embeddings_Lp_short')
            sess_rep = tf.concat([(long_sum*W_embed_Lp_long), (short_sum*W_embed_Lp_short)], 1) #long先再來Short    
        elif fusion_way == "fix": sess_rep = tf.concat([(long_sum*long_score), (short_sum*short_score)], 1)
        elif fusion_way == "att":
            short_x = tf.gather_nd(short_sum, tf.stack([tf.range(batchsize), seq_len-1], axis=1)) 
            long_x = tf.gather_nd(long_sum, tf.stack([tf.range(batchsize), seq_len-1], axis=1)) 
            short_x = tf.math.tanh(short_x); long_x  = tf.math.tanh(long_x); 
            short_x = tf.nn.softmax(short_x); long_x  = tf.nn.softmax(long_x); 
            short_long_sum = short_x+long_x
            max_short = short_x/short_long_sum; max_long  = long_x/short_long_sum; 
            max_short_2 = tf.argmax((short_x/short_long_sum), 1); max_long_2  = tf.argmax((long_x/short_long_sum), 1); 
            max_short_2 = tf.cast(max_short_2[0],dtype=tf.float32); max_long_2 = tf.cast(max_long_2[0],dtype=tf.float32); 
            sess_rep = tf.concat([(long_sum*max_long_2), (short_sum*max_short_2)], 1)
        See_long_short_term = sess_rep
        if strategy == 'pre-combine': input_sum = integrated_computing(switch_plot, X_embed, sess_rep)            
        elif strategy == 'post-combine': input_sum = integrated_computing(switch_plot, X_embed, Inner_outputs) * sess_rep
    if use_FC == True: input_sum = layers.linear(input_sum, EMBEDDING_SIZE) # Whether to finally add FC to input_sum
    last_Inner_output = tf.gather_nd(Inner_outputs, tf.stack([tf.range(batchsize), seq_len-1], axis=1)) 

    # Outer_RNN
    onecell = tf.nn.rnn_cell.GRUCell(OUTER_RNN_SIZE)
    dropcell = tf.nn.rnn_cell.DropoutWrapper(onecell, input_keep_prob=pkeep)
    multicell = tf.nn.rnn_cell.MultiRNNCell([dropcell]*N_LAYERS, state_is_tuple=False)
    multicell = tf.nn.rnn_cell.DropoutWrapper(multicell, output_keep_prob=pkeep)

    if switch_initial_state == True: Yr, H = tf.nn.dynamic_rnn(multicell, input_sum, sequence_length=seq_len, dtype=tf.float32, initial_state=last_Inner_output)
    else: Yr, H = tf.nn.dynamic_rnn(multicell, input_sum, sequence_length=seq_len, dtype=tf.float32)
    H = tf.identity(H, name='H') # just to give it a name
    # Apply softmax to the output
    # Flatten the RNN output first, to share weights across the unrolled time steps
    Yflat = tf.reshape(Yr, [-1, OUTER_RNN_SIZE]) # [ BATCHSIZE x SEQLEN, OUTER_RNN_SIZE ]
    # Change from internal size (from RNNCell) to N_ITEMS size
    Ylogits = layers.linear(Yflat, N_ITEMS) # [ BATCHSIZE x SEQLEN, N_ITEMS ]
    # Flatten expected outputs to match actual outputs
    Y_flat_target = tf.reshape(Y_, [-1]) # [ BATCHSIZE x SEQLEN ]
    # Calculate loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_flat_target) # [ BATCHSIZE x SEQLEN ]
    # Mask the losses (so we don't train in padded values)
    mask = tf.sign(tf.to_float(Y_flat_target))
    masked_loss = mask * loss
    # Unflatten loss
    loss = tf.reshape(masked_loss, [batchsize, -1]) # [ BATCHSIZE, SEQLEN ]
    # Get the index of the highest scoring prediction through Y
    Y = tf.argmax(Ylogits, 1)   # [ BATCHSIZE x SEQLEN ]
    Y = tf.reshape(Y, [batchsize, -1], name='Y') # [ BATCHSIZE, SEQLEN ]
    # Get prediction
    top_k_values, top_k_predictions = tf.nn.top_k(Ylogits, k=TOP_K) # [BATCHSIZE x SEQLEN, TOP_K]
    Y_prediction = tf.reshape(top_k_predictions, [batchsize, -1, TOP_K], name='YTopKPred')
    # Training
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    # Stats
    # Average sequence loss
    seqloss = tf.reduce_mean(loss, 1)
    # Average batchloss
    batchloss = tf.reduce_mean(seqloss)
    # Average number of correct predictions
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.int32)), tf.float32))
    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])
    # Init to save models
    if not os.path.exists("checkpoints"): os.mkdir("checkpoints")
    saver = tf.train.Saver(max_to_keep=1)
    # Initialization
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True # be nice and don't use more memory than necessary
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    ##
    ##  TRAINING
    ##
    print(message); print("\nStarting training.\n"); 
    epoch = datahandler.get_latest_epoch(epoch_file); print("|-Starting on epoch", epoch+1); 
    if epoch > 0:
        print("|--Restoring model.")
        save_file = checkpoint_file + checkpoint_file_ending
        saver.restore(sess, save_file)
    else: sess.run(init)
    epoch += 1; best_recall5 = -1; best_recall20 = -1; 
    num_training_batches = datahandler.get_num_training_batches()
    num_test_batches = datahandler.get_num_test_batches()
    # Generate Trend Chart Data
    data = [[], [], [], [], [], [], [], [], []]
    while epoch <= MAX_EPOCHS:
        print(f"\nStarting epoch #{epoch}")
        epoch_loss = 0
        # Generate Trend Chart Data
        data[0].append(epoch)
        datahandler.reset_user_batch_data()
        datahandler.reset_user_session_representations()
        if do_training:
            tester = Tester()
            _batch_number = 0
            xinput, targetvalues, sl, session_reps, sr_sl, user_list = datahandler.get_next_train_batch()
            while len(xinput) > int(BATCHSIZE/2):
                _batch_number += 1
                batch_start_time = time.time()
                feed_dict = {X: xinput, Y_: targetvalues, lr: learning_rate, pkeep: dropout_pkeep, batchsize: len(xinput), seq_len: sl}
                batch_predictions, _, bl, sess_rep = sess.run([Y_prediction, train_step, batchloss, H], feed_dict=feed_dict)
                # Evaluate predictions
                tester.evaluate_batch(batch_predictions, targetvalues, sl)
                batch_runtime = time.time() - batch_start_time
                epoch_loss += bl
                if _batch_number == 1: print(f"Train | Batch number: {_batch_number} / {num_training_batches}")
                if _batch_number%100 == 0:
                    print("Train | Batch number:", str(_batch_number), "/", str(num_training_batches), "| Batch time:", "%.2f" % batch_runtime, " seconds", end='')
                    print(" | Batch loss:", bl, end='')
                    eta = (batch_runtime*(num_training_batches-_batch_number))/60; eta = "%.2f" % eta; 
                    print(f" | ETA: {eta} minutes.")
                xinput, targetvalues, sl, session_reps, sr_sl, user_list = datahandler.get_next_train_batch()
            print("Epoch", epoch, "finished")
            print("|- Train Loss:", epoch_loss)
            # Generate Trend Chart Data
            data[7].append(epoch_loss) # epoch_loss_train
            if epoch_loss > temp_loss: print("The loss increases, and the best solution in the whole domain may be found !\n")
            temp_loss = epoch_loss
            # Print final test stats for epoch
            test_stats, current_recall5, current_recall10, current_recall20, current_mrr5, current_mrr10, current_mrr20, current_ndcg5, current_ndcg10, current_ndcg20 = tester.get_stats_and_reset()
            temp_recall5 = current_recall5
            print(f"Recall@5 : {current_recall5} | Recall@10 : {current_recall10} | Recall@20 : {current_recall20}")
            print(f"MRR@5    : {current_mrr5} | MRR@10    : {current_mrr10} | MRR@20    : {current_mrr20}")
            print(f"NDCG@5   : {current_ndcg5} | NDCG@10   : {current_ndcg10} | NDCG@20   : {current_ndcg20}")
            print("--------------------------------------------------------------------------------------")
            # Generate Trend Chart Data
            data[1].append(current_recall20); data[2].append(current_mrr20); data[3].append(current_ndcg20); 
        ##
        ## TESTING
        ##
        print("Starting testing")
        tester = Tester()
        datahandler.reset_user_batch_data()
        _batch_number = 0; epoch_loss_test = 0; 
        xinput, targetvalues, sl, session_reps, sr_sl, user_list = datahandler.get_next_test_batch()
        while len(xinput) > int(BATCHSIZE/2):
            batch_start_time = time.time()
            _batch_number += 1
            feed_dict = {X: xinput, Y_: targetvalues, pkeep: 1.0, batchsize: len(xinput), seq_len: sl}
            #1: Predicted item(y) 2: Preference representation(h)
            bl_t, batch_predictions, sess_rep = sess.run([batchloss, Y_prediction, H], feed_dict=feed_dict)
            # Evaluate predictions
            tester.evaluate_batch(batch_predictions, targetvalues, sl)
            # Print some stats during testing
            batch_runtime = time.time() - batch_start_time
            epoch_loss_test += bl_t
            if _batch_number%100 == 0:
                print("Test | Batch number:", str(_batch_number), "/", str(num_test_batches), "| Batch time:", "%.2f" % batch_runtime, " seconds", end='')
                eta = (batch_runtime*(num_test_batches-_batch_number))/60
                eta = "%.2f" % eta
                print(" ETA:", eta, "minutes.")
            xinput, targetvalues, sl, session_reps, sr_sl, user_list = datahandler.get_next_test_batch()
        print("|- Test Loss:", epoch_loss_test)
        # Print final test stats for epoch
        test_stats, current_recall5, current_recall10, current_recall20, current_mrr5, current_mrr10, current_mrr20, current_ndcg5, current_ndcg10, current_ndcg20 = tester.get_stats_and_reset()
        print(f"Recall@5 : {current_recall5} | Recall@10 : {current_recall10} | Recall@20 : {current_recall20}")
        print(f"MRR@5    : {current_mrr5} | MRR@10    : {current_mrr10} | MRR@20    : {current_mrr20}")
        print(f"NDCG@5   : {current_ndcg5} | NDCG@10   : {current_ndcg10} | NDCG@20   : {current_ndcg20}")
        # Generate Trend Chart Data
        data[8].append(epoch_loss_test); data[4].append(current_recall20); data[5].append(current_mrr20); data[6].append(current_ndcg20); 
        lgr = temp_recall5*100
        if Threshold < lgr and epoch > 1: print("\n<<<<<<<<<<<<<<<<<<<<<<< Suspected Overfitting ! >>>>>>>>>>>>>>>>>>>>>>>\n") # recall@5 > Threshold and epoch > 1
        if save_best:
            if current_recall5 > best_recall5 and Threshold > lgr: #if recall@5 better than in the past， and recall@5 <Threshold%
                # Save the model
                print("Saving model.")
                save_file = checkpoint_file + checkpoint_file_ending
                save_path = saver.save(sess, save_file)
                print(f"|- Model saved in file: {save_path}")
                best_recall5 = current_recall5
                #save best once
                temp_list = []
                temp_list.append([epoch, current_recall5, current_recall10, current_recall20, current_mrr5, current_mrr10, current_mrr20, current_ndcg5, current_ndcg10, current_ndcg20, epoch_loss, test_stats])
                datahandler.store_current_epoch(epoch, epoch_file)
        epoch += 1
    print(f"\nBEST Epoch | {temp_list[0][0]}")
    print(f"Recall@5 : {temp_list[0][1]} | Recall@10 : {temp_list[0][2]} | Recall@20 : {temp_list[0][3]}")
    print(f"MRR@5    : {temp_list[0][4]} | MRR@10    : {temp_list[0][5]} | MRR@20    : {temp_list[0][6]}")
    print(f"NDCG@5   : {temp_list[0][7]} | NDCG@10   : {temp_list[0][8]} | NDCG@20   : {temp_list[0][9]}\n")
    # Excerpt the best score and write it in log
    datahandler.log_test_stats(temp_list[0][0], temp_list[0][10], temp_list[0][11]) 
    end_time_1 = "%.2f" %((time.time() - runtime)/60)
    end_time_2 = "%.2f" %((time.time() - runtime)/60/60)
    print(f"Runtime Spend: {end_time_1} min | {end_time_2} hr\n")
    # Generate Trend Chart Data
    df_train = pd.DataFrame({"train_epoch": data[0], "Recall@20": data[1], "MRR@20": data[2], "NDCG@20": data[3]})
    df_test = pd.DataFrame({"test_epoch": data[0], "Recall@20": data[4], "MRR@20": data[5], "NDCG@20": data[6]})
    df_loss = pd.DataFrame({"loss_epoch": data[0], "train_loss": data[7], "test_loss": data[8]})
    png(df_train, "train_epoch", "Recall@20", "MRR@20", "NDCG@20", dataset+"-"+log_file_name+"-k@20 Evaluation indicator trend chart train", "k@20 Evaluation indicator trend chart train")
    png(df_test, "test_epoch", "Recall@20", "MRR@20", "NDCG@20", dataset+"-"+log_file_name+"-k@20 Evaluation indicator trend chart test", "k@20 Evaluation indicator trend chart test")
    png(df_loss, "loss_epoch", "train_loss", "test_loss", "none", dataset+"-"+log_file_name+"-Loss Trend", "Loss Trend")
    datahandler.close_log() # close log