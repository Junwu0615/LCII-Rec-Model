# -*- coding: utf-8 -*-
"""
@author: PC
Update Time: 2024-12-01
"""

import time

import pandas as pd
import tf_slim as layers

import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

from depend.test_utils import Tester
from depend.baselogic import BaseLogic
from depend.integrated_utils import IIRNNDataHandler

class LatentContextIIGRU(BaseLogic):
    def __init__(self, obj):
        self.dataset = obj.dataset
        self.switch_plot = obj.switch_plot
        self.fusion_way = obj.fusion_way
        self.strategy = obj.strategy
        self.window = float(obj.window) if obj.window != 'no_use' else obj.window
        self.lo_score = float(obj.lo_score) if obj.lo_score != 'no_use' else obj.lo_score
        self.sh_score = float(obj.sh_score) if obj.sh_score != 'no_use' else obj.sh_score
        self.embed_size = obj.embed_size
        self.batch_size = obj.batch_size
        self.learning_rate = obj.learning_rate
        self.dropout = obj.dropout
        self.max_epoch = obj.max_epoch
        self.threshold = obj.threshold # If recall@5 > Threshold, it will be determined as overfitting.
        self.use_fc = True if obj.use_fc == "True" else False
        self.switch_initial_state = True if obj.switch_initial_state == "True" else False
        self.seq_len = None
        self.dataset_path = ''
        self.top_k = 20
        self.n_layers = 1  # number of layers in the rnn
        self.max_session_rep = 0  # Won't use it
        self.temp_loss = 1000000000000000  # Prevent the loss from rising for the first time
        self.seed = 888
        self.save_best = True

    def rename_log(self) -> str:
        # FIXME Give name to log_file_name
        base = f"LCII_ver_{self.strategy}_dropout_{self.dropout}"
        if self.fusion_way == 'none':
            rename = base
        else:
            if self.fusion_way == 'fix':
                rename = f"{base}_w-{self.window}_{self.fusion_way}_L-{self.lo_score}_S-{self.sh_score}"
            else:
                rename = f"{base}_w-{self.window}_{self.fusion_way}"

        return f"{rename}_fc" if self.use_fc else rename

    def check_test_utils(self) -> Tester:
        if self.dataset == "Amazon":
            self.dataset_path = './Datasets/' + self.dataset.lower() + '/4_train_test_split.pickle'
            # dataset_path = './Datasets/' + self.dataset.lower() + '/4_train_test_split_sample.pickle'
            self.seq_len = 20 - 1

        elif self.dataset == "MovieLens-1M":
            self.dataset = "ml-1m"
            self.dataset_path = './Datasets/' + self.dataset.lower() + '/pickle' + '/4_train_test_split.pickle'
            # dataset_path = './Datasets/' + self.dataset.lower() + '/pickle' + '/4_train_test_split_sample.pickle'
            self.seq_len = 200 - 1

        elif self.dataset == "Steam":
            self.dataset_path = './Datasets/' + self.dataset.lower() + '/pickle' + '/4_train_test_split.pickle'
            # dataset_path = './Datasets/' + self.dataset.lower() + '/pickle' + '/4_train_test_split_sample.pickle'
            self.seq_len = 15 - 1

        tester = Tester(session_length=self.seq_len)
        return tester

    def generate_chart(self, df, epoch, value_1, value_2, value_3, value_4, value_5):
        # FIXME Generate Trend Chart Data
        fig = plt.figure(figsize=(6, 4))
        fig.patch.set_facecolor('white')
        plt.rcParams['font.sans-serif'] = ['Yu Gothic']
        plt.grid(True)

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
        y_string = "loss" if epoch == "loss_epoch" else "%"
        plt.ylabel(y_string)
        plt.title(value_5)
        plt.savefig("./testlog/" + value_4 + ".png")
        plt.clf()

    def integrated_computing(self, input_1, input_2):
        input_sum = None
        if self.switch_plot == "sum":
            input_sum = input_1 + input_2

        elif self.switch_plot == "dot":
            input_sum = input_1 * input_2

        elif self.switch_plot == "attention_gate_sum":
            a = tf.math.tanh(input_1)
            b = tf.math.tanh(input_2)
            a = tf.nn.softmax(a)
            b = tf.nn.softmax(b)
            input_sum = (a / (a + b) * input_1) + (b / (a + b) * input_2)

        elif self.switch_plot == "attention_gate_dot":
            w_input_1 = tf.math.tanh(input_1)
            w_input_2 = tf.math.tanh(input_2)
            w_input_1 = tf.nn.softmax(w_input_1)
            w_input_2 = tf.nn.softmax(w_input_2)
            w_input_sum = w_input_1 + w_input_2
            input_sum = ((w_input_1 / w_input_sum) * input_1) * ((w_input_2 / w_input_sum) * input_2)

        return input_sum

    def make_neural_map(self):
        # Inputs
        self.x = tf.placeholder(tf.int32, [None, None], name='x')  # [ batch_size, seq_len ]
        self.y_ = tf.placeholder(tf.int32, [None, None], name='y_')  # [ batch_size, seq_len ]

        # Embeddings. w_embed = all embeddings. x_embed = retrieved embeddings
        # from w_embed, corresponding to the items in the current batch
        w_embed = tf.Variable(tf.random_uniform([self.n_items, self.embed_size], -1.0, 1.0), name='embeddings')
        x_embed = tf.nn.embedding_lookup(w_embed, self.x)  # [batch_size, seq_len, EMBEDDING_SIZE]

        # Length of sesssions (not considering padding)
        self.seq_len = tf.placeholder(tf.int32, [None], name='seqlen')
        self.batchsize = tf.placeholder(tf.int32, name='batchsize')
        self.lr = tf.placeholder(tf.float32, name='lr')  # learning rate
        self.pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter

        # Inner_RNN
        Inner_cell = tf.nn.rnn_cell.GRUCell(self.embed_size)
        Inner_dropcell = tf.nn.rnn_cell.DropoutWrapper(Inner_cell,
                                                       input_keep_prob=self.pkeep, output_keep_prob=self.pkeep)
        Inner_outputs, Inner_states = tf.nn.dynamic_rnn(Inner_dropcell, x_embed,
                                                        sequence_length=self.seq_len,dtype=tf.float32)

        if self.strategy == 'original':
            input_sum = self.integrated_computing(x_embed, Inner_outputs)
        else:
            # Long and short term processing
            # h_list.append(tf.nn.relu(Inner_outputs))#Relu
            if int((self.seq_len + 1) * self.window * 0.01) < 1:
                capture_len_h_list = -1
            else:
                capture_len_h_list = int((self.seq_len + 1) * - self.window * 0.01)
            short_sum = Inner_outputs[:, capture_len_h_list:]
            long_sum = Inner_outputs[:, :capture_len_h_list]

            input_sum, sess_rep = None, None
            if self.fusion_way == "lp":
                # Learn long and short term weights
                W_embed_Lp_long = (
                    tf.Variable(tf.random_uniform([1, self.seq_len + capture_len_h_list, self.embed_size], -1.0, 1.0),
                                name='embeddings_Lp_long'))
                W_embed_Lp_short = (
                    tf.Variable(tf.random_uniform([1, -capture_len_h_list, self.embed_size], -1.0, 1.0),
                                name='embeddings_Lp_short'))
                sess_rep = tf.concat([(long_sum * W_embed_Lp_long), (short_sum * W_embed_Lp_short)],1)
            elif self.fusion_way == "fix":
                sess_rep = tf.concat([(long_sum * self.lo_score), (short_sum * self.sh_score)], 1)

            elif self.fusion_way == "att":
                short_x = tf.gather_nd(short_sum, tf.stack([tf.range(self.batchsize), self.seq_len - 1], axis=1))
                long_x = tf.gather_nd(long_sum, tf.stack([tf.range(self.batchsize), self.seq_len - 1], axis=1))
                short_x = tf.math.tanh(short_x)
                long_x = tf.math.tanh(long_x)
                short_x = tf.nn.softmax(short_x)
                long_x = tf.nn.softmax(long_x)
                short_long_sum = short_x + long_x
                max_short = short_x / short_long_sum
                max_long = long_x / short_long_sum
                max_short_2 = tf.argmax((short_x / short_long_sum), 1)
                max_long_2 = tf.argmax((long_x / short_long_sum), 1)
                max_short_2 = tf.cast(max_short_2[0], dtype=tf.float32)
                max_long_2 = tf.cast(max_long_2[0], dtype=tf.float32)
                sess_rep = tf.concat([(long_sum * max_long_2), (short_sum * max_short_2)], 1)

            if self.strategy == 'pre-combine':
                input_sum = self.integrated_computing(x_embed, sess_rep)

            elif self.strategy == 'post-combine':
                input_sum = self.integrated_computing(x_embed, Inner_outputs) * sess_rep

        if self.use_fc:
            input_sum = layers.linear(input_sum, self.embed_size)  # Whether to finally add FC to input_sum
        last_Inner_output = tf.gather_nd(Inner_outputs, tf.stack([tf.range(self.batchsize), self.seq_len - 1], axis=1))

        # Outer_RNN
        onecell = tf.nn.rnn_cell.GRUCell(self.embed_size)
        dropcell = tf.nn.rnn_cell.DropoutWrapper(onecell, input_keep_prob=self.pkeep)
        multicell = tf.nn.rnn_cell.MultiRNNCell([dropcell] * self.n_layers, state_is_tuple=False)
        multicell = tf.nn.rnn_cell.DropoutWrapper(multicell, output_keep_prob=self.pkeep)

        if self.switch_initial_state:
            Yr, self.H = tf.nn.dynamic_rnn(multicell, input_sum, sequence_length=self.seq_len, dtype=tf.float32,
                                      initial_state=last_Inner_output)
        else:
            Yr, self.H = tf.nn.dynamic_rnn(multicell, input_sum, sequence_length=self.seq_len, dtype=tf.float32)
        self.H = tf.identity(self.H, name='H')  # just to give it a name
        # Apply softmax to the output
        # Flatten the RNN output first, to share weights across the unrolled time steps
        Yflat = tf.reshape(Yr, [-1, self.embed_size])  # [ batch_size x seq_len, OUTER_RNN_SIZE ]
        # Change from internal size (from RNNCell) to n_items size
        Ylogits = layers.linear(Yflat, self.n_items)  # [ batch_size x seq_len, n_items ]
        # Flatten expected outputs to match actual outputs
        Y_flat_target = tf.reshape(self.y_, [-1])  # [ batch_size x seq_len ]
        # Calculate loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Ylogits,
                                                              labels=Y_flat_target)  # [ batch_size x seq_len ]
        # Mask the losses (so we don't train in padded values)
        mask = tf.sign(tf.to_float(Y_flat_target))
        masked_loss = mask * loss

        # Unflatten loss
        loss = tf.reshape(masked_loss, [self.batchsize, -1])  # [ batch_size, seq_len ]

        # Get the index of the highest scoring prediction through Y
        self.Y = tf.argmax(Ylogits, 1)  # [ batch_size x seq_len ]
        self.Y = tf.reshape(self.Y, [self.batchsize, -1], name='Y')  # [ batch_size, seq_len ]

        # Get prediction
        top_k_values, top_k_predictions = tf.nn.top_k(Ylogits, k=self.top_k)  # [batch_size x seq_len, top_k]

        self.Y_prediction = tf.reshape(top_k_predictions, [self.batchsize, -1, self.top_k], name='YTopKPred')

        # Training
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)

        # Stats
        # Average sequence loss
        seqloss = tf.reduce_mean(loss, 1)

        # Average batchloss
        self.batchloss = tf.reduce_mean(seqloss)

        # Average number of correct predictions
        accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_, tf.cast(self.Y, tf.int32)), tf.float32))
        loss_summary = tf.summary.scalar("batch_loss", self.batchloss)
        acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
        summaries = tf.summary.merge([loss_summary, acc_summary])

        # Init to save models
        saver = tf.train.Saver(max_to_keep=1)

        # Initialization
        self.init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True  # be nice and don't use more memory than necessary
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()

    def log_initial_settings(self, log_file_name):
        print("==========================================\n")
        print(f"LCII Program Starting | Current Time: {self.time_now()}")

        # Load training data
        self.epoch_file = './epoch_file-' + log_file_name + '-' + self.dataset + '.pickle'
        self.checkpoint_file = './checkpoints/' + log_file_name + '-' + self.dataset + '-'
        self.checkpoint_file_ending = '.ckpt'
        log_file = f"./testlog/{str(self.time_now())[:10]}_testing_preference.txt"

        datahandler = IIRNNDataHandler(self.dataset_path, self.batch_size, log_file, self.max_session_rep, self.embed_size)
        self.n_items = datahandler.get_num_items()
        self.n_sessions = datahandler.get_num_training_sessions()
        message = "------------------------------------------\n"
        message += (f"DATASET: {self.dataset} | MODEL: {log_file_name}\n")
        message += (f"SWITCH_INITIAL_STATE: {self.switch_initial_state} | USE_FC_IN_INPUT-SUM: {self.use_fc}\n")
        message += (f"STRATEGY: {self.strategy} | FUSION_WAY: {self.fusion_way} | SWITCH_PLOT: {self.switch_plot}\n")
        message += (f"ADJACENT_WINDOW_RATIO(0-100%): {self.window} | ")
        message += (f"LONG_TERM_SCORE: {self.lo_score} | SHORT_TERM_SCORE: {self.sh_score}\n")
        message += (f"DROPOUT: {self.dropout} | LEARNING_RATE: {self.learning_rate} | ")
        message += (f"BATCHSIZE: {self.batch_size} | SEED: {self.seed}\n")
        message += (f"OUTER_RNN_SIZE: {self.embed_size} | "
                    f"INNER_RNN_SIZE: {self.embed_size} | "
                    f"EMBEDDING_SIZE: {self.embed_size}\n")
        message += (f"TRAIN_N_SESSIONS: {self.n_sessions} | N_ITEMS: {self.n_items} | N_LAYERS: {self.n_layers} | ")
        message += (f"SEQLEN: {self.seq_len} | MAX_EPOCHS: {self.max_epoch}\n")
        message += (f"MAX_SESSION_REPRESENTATIONS: {self.max_session_rep}\n")
        message += "------------------------------------------\n"
        datahandler.log_config(message)
        return datahandler, message

    def main(self):
        do_training = True

        # FIXME Initial Settings
        runtime = time.time()
        BaseLogic.check_folder('./testlog')
        BaseLogic.check_folder('./checkpoints')
        BaseLogic.settings_seed(seed=self.seed)
        BaseLogic.settings_gpu_threshold()
        tf.reset_default_graph()
        log_file_name = self.rename_log()
        tester = self.check_test_utils()
        datahandler, message = self.log_initial_settings(log_file_name)
        self.make_neural_map()

        if not do_training:
            print("\nOBS!!! Training is turned off!\n")

        ##  TRAINING
        print(message)
        print("\nStarting training.\n")
        epoch = datahandler.get_latest_epoch(self.epoch_file)
        print("|-Starting on epoch", epoch + 1)
        if epoch > 0:
            print("|--Restoring model.")
            save_file = self.checkpoint_file + self.checkpoint_file_ending
            self.saver.restore(self.sess, save_file)
        else:
            self.sess.run(self.init)

        epoch += 1
        best_recall5 = -1
        best_recall20 = -1
        num_training_batches = datahandler.get_num_training_batches()
        num_test_batches = datahandler.get_num_test_batches()
        data = [[], [], [], [], [], [], [], [], []] # Generate Trend Chart Data
        while epoch <= self.max_epoch:
            print(f"\nStarting epoch #{epoch}")
            epoch_loss = 0
            data[0].append(epoch) # Generate Trend Chart Data
            datahandler.reset_user_batch_data()
            datahandler.reset_user_session_representations()
            if do_training:
                _batch_number = 0
                xinput, targetvalues, sl, session_reps, sr_sl, user_list = datahandler.get_next_train_batch()
                while len(xinput) > int(self.batch_size / 2):
                    _batch_number += 1
                    batch_start_time = time.time()
                    feed_dict = {self.x: xinput, self.y_: targetvalues, self.lr: self.learning_rate, self.pkeep: self.dropout,
                                 self.batchsize: len(xinput), self.seq_len: sl}
                    batch_predictions, _, bl, sess_rep = self.sess.run(
                        [self.Y_prediction, self.train_step, self.batchloss, self.H], feed_dict=feed_dict)

                    # Evaluate predictions
                    tester.evaluate_batch(batch_predictions, targetvalues, sl)
                    batch_runtime = time.time() - batch_start_time
                    epoch_loss += bl
                    if _batch_number == 1: print(f"Train | Batch number: {_batch_number} / {num_training_batches}")
                    if _batch_number % 100 == 0:
                        print("Train | Batch number:", str(_batch_number), "/", str(num_training_batches),
                              "| Batch time:", "%.2f" % batch_runtime, " seconds", end='')
                        print(" | Batch loss:", bl, end='')
                        eta = (batch_runtime * (num_training_batches - _batch_number)) / 60
                        eta = "%.2f" % eta
                        print(f" | ETA: {eta} minutes.")
                    xinput, targetvalues, sl, session_reps, sr_sl, user_list = datahandler.get_next_train_batch()

                print("Epoch", epoch, "finished")
                print("|- Train Loss:", epoch_loss)

                # Generate Trend Chart Data
                data[7].append(epoch_loss)  # epoch_loss_train

                if epoch_loss > self.temp_loss:
                    print("The loss increases, and the best solution in the whole domain may be found !\n")
                self.temp_loss = epoch_loss

                # Print final test stats for epoch
                (test_stats, current_recall5, current_recall10, current_recall20,
                 current_mrr5, current_mrr10, current_mrr20,
                 current_ndcg5, current_ndcg10, current_ndcg20) = tester.get_stats_and_reset()

                temp_recall5 = current_recall5
                print(f"Recall@5 : {current_recall5} | Recall@10 : {current_recall10} | Recall@20 : {current_recall20}")
                print(f"MRR@5    : {current_mrr5} | MRR@10    : {current_mrr10} | MRR@20    : {current_mrr20}")
                print(f"NDCG@5   : {current_ndcg5} | NDCG@10   : {current_ndcg10} | NDCG@20   : {current_ndcg20}")
                print("--------------------------------------------------------------------------------------")

                # Generate Trend Chart Data
                data[1].append(current_recall20)
                data[2].append(current_mrr20)
                data[3].append(current_ndcg20)

            ## TESTING
            print("Starting testing")
            datahandler.reset_user_batch_data()
            _batch_number = 0
            epoch_loss_test = 0
            xinput, targetvalues, sl, session_reps, sr_sl, user_list = datahandler.get_next_test_batch()
            while len(xinput) > int(self.batch_size / 2):
                batch_start_time = time.time()
                _batch_number += 1
                feed_dict = {self.x: xinput, self.y_: targetvalues, self.pkeep: 1.0,
                             self.batchsize: len(xinput), self.seq_len: sl}

                # 1: Predicted item(y) 2: Preference representation(h)
                bl_t, batch_predictions, sess_rep = self.sess.run(
                    [self.batchloss, self.Y_prediction, self.H], feed_dict=feed_dict)

                # Evaluate predictions
                tester.evaluate_batch(batch_predictions, targetvalues, sl)

                # Print some stats during testing
                batch_runtime = time.time() - batch_start_time
                epoch_loss_test += bl_t
                if _batch_number % 100 == 0:
                    print("Test | Batch number:", str(_batch_number), "/", str(num_test_batches), "| Batch time:",
                          "%.2f" % batch_runtime, " seconds", end='')
                    eta = (batch_runtime * (num_test_batches - _batch_number)) / 60
                    eta = "%.2f" % eta
                    print(" ETA:", eta, "minutes.")

                xinput, targetvalues, sl, session_reps, sr_sl, user_list = datahandler.get_next_test_batch()

            print("|- Test Loss:", epoch_loss_test)

            # Print final test stats for epoch
            (test_stats, current_recall5, current_recall10, current_recall20, current_mrr5, current_mrr10,
             current_mrr20, current_ndcg5, current_ndcg10, current_ndcg20) = tester.get_stats_and_reset()
            print(f"Recall@5 : {current_recall5} | Recall@10 : {current_recall10} | Recall@20 : {current_recall20}")
            print(f"MRR@5    : {current_mrr5} | MRR@10    : {current_mrr10} | MRR@20    : {current_mrr20}")
            print(f"NDCG@5   : {current_ndcg5} | NDCG@10   : {current_ndcg10} | NDCG@20   : {current_ndcg20}")

            # Generate Trend Chart Data
            data[8].append(epoch_loss_test)
            data[4].append(current_recall20)
            data[5].append(current_mrr20)
            data[6].append(current_ndcg20)
            lgr = temp_recall5 * 100
            if self.threshold < lgr and epoch > 1: print(
                "\n<<<<<<<<<<<<<<<<<<<<<<< Suspected Overfitting ! >>>>>>>>>>>>>>>>>>>>>>>\n")  # recall@5 > Threshold and epoch > 1
            if self.save_best:
                if current_recall5 > best_recall5 and self.threshold > lgr:  # if recall@5 better than in the past， and recall@5 <Threshold%
                    # Save the model
                    print("Saving model.")
                    save_file = self.checkpoint_file + self.checkpoint_file_ending
                    save_path = self.saver.save(self.sess, save_file)
                    print(f"|- Model saved in file: {save_path}")
                    best_recall5 = current_recall5
                    # save best once
                    temp_list = []
                    temp_list.append(
                        [epoch, current_recall5, current_recall10, current_recall20, current_mrr5, current_mrr10,
                         current_mrr20, current_ndcg5, current_ndcg10, current_ndcg20, epoch_loss, test_stats])
                    datahandler.store_current_epoch(epoch, self.epoch_file)
            epoch += 1
        print(f"\nBEST Epoch | {temp_list[0][0]}")
        print(f"Recall@5 : {temp_list[0][1]} | Recall@10 : {temp_list[0][2]} | Recall@20 : {temp_list[0][3]}")
        print(f"MRR@5    : {temp_list[0][4]} | MRR@10    : {temp_list[0][5]} | MRR@20    : {temp_list[0][6]}")
        print(f"NDCG@5   : {temp_list[0][7]} | NDCG@10   : {temp_list[0][8]} | NDCG@20   : {temp_list[0][9]}\n")

        # Excerpt the best score and write it in log
        datahandler.log_test_stats(temp_list[0][0], temp_list[0][10], temp_list[0][11])
        end_time_1 = "%.2f" % ((time.time() - runtime) / 60)
        end_time_2 = "%.2f" % ((time.time() - runtime) / 60 / 60)
        print(f"Runtime Spend: {end_time_1} min | {end_time_2} hr\n")

        # Generate Trend Chart Data
        df_train = pd.DataFrame({"train_epoch": data[0], "Recall@20": data[1], "MRR@20": data[2], "NDCG@20": data[3]})
        df_test = pd.DataFrame({"test_epoch": data[0], "Recall@20": data[4], "MRR@20": data[5], "NDCG@20": data[6]})
        df_loss = pd.DataFrame({"loss_epoch": data[0], "train_loss": data[7], "test_loss": data[8]})

        self.generate_chart(df_train, "train_epoch", "Recall@20", "MRR@20", "NDCG@20",
            self.dataset + "-" + log_file_name + "-k@20 Evaluation indicator trend chart train",
            "k@20 Evaluation indicator trend chart train")

        self.generate_chart(df_test, "test_epoch", "Recall@20", "MRR@20", "NDCG@20",
            self.dataset + "-" + log_file_name + "-k@20 Evaluation indicator trend chart test",
            "k@20 Evaluation indicator trend chart test")

        self.generate_chart(df_loss, "loss_epoch", "train_loss", "test_loss", "none",
                            self.dataset + "-" + log_file_name + "-Loss Trend",
            "Loss Trend")
        datahandler.close_log()  # close log


