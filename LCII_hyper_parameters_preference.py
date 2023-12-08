class hyper_parameters:
    
    dataset = { 
    'reddit': "subreddit",
    'lastfm': "lastfm",
    'instacart':"instacart",
    'tmall': "tmall_6",
    'amazon': "amazon",
    'MovieLens_1M': 'MovieLens-1M',
    'MovieLens_20M': 'MovieLens-20M',
    'steam': 'steam' 
    }

    switch_plot = {
    'sum': 'sum',
    'dot': 'dot',
    'attention_gate_sum': 'attention_gate_sum',
    'attention_gate_dot': 'attention_gate_dot'
    }

    switch_initial_state = {
    'True': True,
    'False': False
    }

    fusion_way = {
    'att': 'att',
    'lp' : 'lp',
    'fix': 'fix',
    'none': 'none'
    }

    strategy = {
    'pre-combine':'pre-combine',
    'post-combine':'post-combine',
    'original': 'original'
    }

    run_list = [
    [dataset['MovieLens_1M'], #reddit / lastfm / instacart / tmall / steam / MovieLens_1M / MovieLens_20M
     switch_plot['attention_gate_dot'], #sum / dot / attention_gate_sum / attention_gate_dot
     switch_initial_state['True'], #True / False
     fusion_way['none'], #att / lp / fix / none
     strategy['original'], #pre-combine / post-combine / original
     0,  #window: 0-100
     'no_use', #long_score: 0.0-1.0 / 'no_use'
     'no_use', #short_score: 0.0-1.0 / 'no_use'
     80,  #embedding size: 30 / 50 / 80 / 100 / 200 / 300 / 500 / 800 / 1000
     100, #batch size: 16 / 32 / 64 / 100 / 128 / 256 / 512
     0.01, #learning_rate: 0.001 / 0.01 / ...
     0, #dropout_pkeep: 0.8
     200, #Max epoch: 100 / 200 / ...
     '', #temp space #don't key log_name #log_file_name 
     98, #Threshold: 98 recall@5如果超過 x 就判定overfitting
     False, #是否要在input_sum 加入FC: True / False
    ],
    [dataset['MovieLens_1M'], #reddit / lastfm / instacart / tmall / steam / MovieLens_1M / MovieLens_20M
     switch_plot['attention_gate_dot'], #sum / dot / attention_gate_sum / attention_gate_dot
     switch_initial_state['True'], #True / False
     fusion_way['fix'], #att / lp / fix / none
     strategy['pre-combine'], #pre-combine / post-combine / original
     30,  #window: 0-100
     0.8, #long_score: 0.0-1.0 / 'no_use'
     0.2, #short_score: 0.0-1.0 / 'no_use'
     80,  #embedding size: 30 / 50 / 80 / 100 / 200 / 300 / 500 / 800 / 1000
     100, #batch size: 16 / 32 / 64 / 100 / 128 / 256 / 512
     0.01, #learning_rate: 0.001 / 0.01 / ...
     0, #dropout_pkeep: 0.8
     200, #Max epoch: 100 / 200 / ...
     '', #temp space #don't key log_name #log_file_name 
     98, #Threshold: 98 recall@5如果超過 x 就判定overfitting
     False, #是否要在input_sum 加入FC: True / False
    ]
    ]
    
    model_num = len(run_list)
    for each in range(model_num):
        if run_list[each][3] == 'none':
            run_list[each][13] = 'LCII_ver_'+run_list[each][4]+'_dropout_'+str(run_list[each][11])
        else:
            if run_list[each][3] == 'fix': 
                run_list[each][13] = 'LCII_ver_'+run_list[each][4]+'_w-'+str(run_list[each][5])+'_'+run_list[each][3]+'_L-'+str(run_list[each][6])+'_S-'+str(run_list[each][7])+'_dropout_'+str(run_list[each][11])
            else:
                run_list[each][13] = 'LCII_ver_'+run_list[each][4]+'_w-'+str(run_list[each][5])+'_'+run_list[each][3]+'_dropout_'+str(run_list[each][11])

        if run_list[each][15] == True:
            run_list[each][13] = run_list[each][13]+'_FC'