import torch
import argparse
import pandas as pd
from DCA.utils.data import Interactions, Categories
from DCA.config import get_parameters, get_logger, ACC_KPI, DIV_KPI, AGG_KPI
from DCA.utils.model_selection import fold_out, train_test_split, handle_adj, build_graph
from DCA.utils.dataset import NARMDataset, NARMDataset_Cat, SRGNNDataset, SRGNNDataset_Cat, GRU4RECDataset, ConventionDataset, GCEDataset, GCEDataset_Cat
from DCA.utils.metrics import accuracy_calculator, diversity_calculator, performance_calculator
from DCA.model.ca_narm import CA_NARM
from DCA.model.dca_narm import DCA_NARM
from DCA.model.ca_gcegnn import CA_CombineGraph
from DCA.model.ca_gcsan import CA_GCSAN
from DCA.model.dca_gcegnn import DCA_CombineGraph
from DCA.model.dca_gcsan import DCA_GCSAN
from DCA.model.ca_stamp import CA_STAMP
from DCA.model.ca_stamp_concat import CA_STAMP_CONCAT
from DCA.model.dca_stamp import DCA_STAMP
from DCA.utils.functions import reindex
from DCA.config import TUNE_PATH
from hyperopt import hp, tpe, fmin
import json
import os
import numpy as np
import time

def best_func(param_dict, args, f):
    # specify certain parameter according to algo_name
    # common hyperparameters
    hypers_dict = dict()
    hypers_dict['item_embedding_dim'] = int(param_dict['item_embedding_dim']) if 'item_embedding_dim' in param_dict.keys() else args.item_embedding_dim
    hypers_dict['lr'] = param_dict['lr'] if 'lr' in param_dict.keys() else args.lr
    hypers_dict['batch_size'] = int(param_dict['batch_size']) if 'batch_size' in param_dict.keys() else args.batch_size
    hypers_dict['epochs'] = int(param_dict['epochs']) if 'epochs' in param_dict.keys() else args.epochs
    
    # narm&gru4rec addition
    hypers_dict['hidden_size'] = int(param_dict['hidden_size']) if 'hidden_size' in param_dict.keys() else args.hidden_size
    hypers_dict['n_layers'] = int(param_dict['n_layers']) if 'n_layers' in param_dict.keys() else args.n_layers
    
    # gru4rec addition
    hypers_dict['dropout_input'] = param_dict['dropout_input'] if 'dropout_input' in param_dict.keys() else args.dropout_input
    hypers_dict['dropout_hidden'] = param_dict['dropout_hidden'] if 'dropout_hidden' in param_dict.keys() else args.dropout_hidden
    
    # srgnn&gcsan addition
    hypers_dict['step'] = int(param_dict['step']) if 'step' in param_dict.keys() else args.step
    
    # itemknn addition
    hypers_dict['alpha'] = param_dict['alpha'] if 'alpha' in param_dict.keys() else args.alpha
    
    # gcegnn addition
    hypers_dict['n_iter'] = int(param_dict['n_iter']) if 'n_iter' in param_dict.keys() else args.n_iter # [1, 2]
    hypers_dict['dropout_gcn'] = param_dict['dropout_gcn'] if 'dropout_gcn' in param_dict.keys() else args.dropout_gcn # [0, 0.2, 0.4, 0.6, 0.8]
    hypers_dict['dropout_local'] = param_dict['dropout_local'] if 'dropout_local' in param_dict.keys() else args.dropout_local # [0, 0.5]


    hypers_dict['weight'] = param_dict['weight'] if 'weight' in param_dict.keys() else args.weight
    hypers_dict['blocks'] = int(param_dict['blocks']) if 'blocks' in param_dict.keys() else args.blocks
    hypers_dict['tau'] = param_dict['tau'] if 'tau' in param_dict.keys() else args.tau
    hypers_dict['purposes'] = param_dict['purposes'] if 'purposes' in param_dict.keys() else args.purposes
    
    
    # reset hyperparmeters in args using hyperopt param_dict value
    args.item_embedding_dim = hypers_dict['item_embedding_dim']
    args.lr = hypers_dict['lr']
    args.batch_size = hypers_dict['batch_size']
    args.epochs = hypers_dict['epochs']
    # narm&gru4rec addition
    args.hidden_size = hypers_dict['hidden_size']
    args.n_layers = hypers_dict['n_layers']
    # gru4rec addition
    args.dropout_input = hypers_dict['dropout_input']
    args.dropout_hidden = hypers_dict['dropout_hidden']
    # srgnn&gcsan addition
    args.step = hypers_dict['step']
    
    args.alpha = hypers_dict['alpha']
    
    # gcegnn addition
    args.n_iter = hypers_dict['n_iter']
    args.dropout_gcn = hypers_dict['dropout_gcn']
    args.dropout_local = hypers_dict['dropout_local']

    args.weight = hypers_dict['weight']
    args.blocks = hypers_dict['blocks']
    args.tau = hypers_dict['tau']
    args.purposes = hypers_dict['purposes']
    
#    print(args.alpha)
    # new conf and model_conf
    conf, model_conf = get_parameters(args)

    logger = get_logger(__file__.split('.')[0] + f'_{conf["description"]}')

    ds = Interactions(conf, logger)
    if conf['category_key'] in ds.df.columns:
        ds.df = ds.df.drop(columns=conf['category_key'])
    train, test = train_test_split(ds.df, conf, logger, n_days=ds.n_days)#fold_out(ds.df, conf) #[:5000]
    train, item_id_map, id_item_map = reindex(train, conf['item_key'], None, start_from_zero=False)
    test = reindex(test, conf['item_key'], item_id_map) # code from 1

    # for category information
    cats = Categories(item_id_map, conf, logger)
    train = pd.merge(train, cats.df, how='left', on=conf['item_key'])
    test = pd.merge(test, cats.df, how='left', on=conf['item_key'])

    if conf['model'] in ['ca_narm', 'dca_narm', 'ca_stamp', 'dca_stamp', 'ca_stamp_concat']:
        train_dataset = NARMDataset_Cat(train, conf)
        test_dataset = NARMDataset_Cat(test, conf)
        train_loader = train_dataset.get_loader(model_conf, shuffle=True)
        test_loader = test_dataset.get_loader(model_conf, shuffle=False)

#        model = CA_NARM(cats.item_num, cats.n_cates, model_conf, logger).to(device) if conf['model']=='ca_narm' else DCA_NARM(cats.item_num, cats.n_cates, cats.item_cate_matrix, model_conf, logger).to(device)
        if conf['model'].split('_')[-1] == 'concat':
            if conf['model'].split('_')[1] == 'narm':
                model = CA_NARM(cats.item_num, cats.n_cates, model_conf, logger).to(device) if conf['model']=='ca_narm' else DCA_NARM(cats.item_num, cats.n_cates, cats.item_cate_matrix, model_conf, logger).to(device)
            if conf['model'].split('_')[1] == 'stamp':
                model = CA_STAMP_CONCAT(cats.item_num, cats.n_cates, model_conf, logger).to(device) #if conf['model']=='ca_stamp' else DCA_STAMP(cats.item_num, cats.n_cates, cats.item_cate_matrix, model_conf, logger).to(device)
        else:
            if conf['model'].split('_')[1] == 'narm':
                model = CA_NARM(cats.item_num, cats.n_cates, model_conf, logger).to(device) if conf['model']=='ca_narm' else DCA_NARM(cats.item_num, cats.n_cates, cats.item_cate_matrix, model_conf, logger).to(device)
            if conf['model'].split('_')[1] == 'stamp':
                model = CA_STAMP(cats.item_num, cats.n_cates, model_conf, logger).to(device) if conf['model']=='ca_stamp' else DCA_STAMP(cats.item_num, cats.n_cates, cats.item_cate_matrix, model_conf, logger).to(device)
            
        start = time.time()
        model.fit(train_loader)
        end = time.time()
        training_time_epoch = (end-start)/args.epochs
        hours, rem = divmod(training_time_epoch, 3600)
        minutes, seconds = divmod(rem, 60)
        
        f1 = open(TUNE_PATH+'Time_DCA.txt','a',encoding='utf-8')
        f1.write(str(args.dataset) + " NARM Training Time (epoch) : {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds) + "\n")
    
        start = time.time()
        preds, truth = model.predict(test_loader, conf['topk'])
        end = time.time()
        
        inference_time = end-start
        hours, rem = divmod(inference_time, 3600)
        minutes, seconds = divmod(rem, 60)
        f1.write(str(args.dataset) + " NARM Inference Time : {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds) + "\n")
        f1.close()
    elif conf['model'] in ['ca_gcegnn', 'dca_gcegnn']:
        # adj, num
        adj, num = build_graph(train, conf, model_conf)
        train_dataset = GCEDataset_Cat(train, conf)
        test_dataset = GCEDataset_Cat(test, conf)
        num_node = cats.item_num#train[conf['item_key']].nunique() + 1
        num_category = cats.n_cates
        adj, num = handle_adj(adj, num_node, model_conf['n_sample_all'], num)
        model = CA_CombineGraph(model_conf, num_node, num_category, adj, num, logger) if conf['model']=='ca_gcegnn' else DCA_CombineGraph(model_conf, num_node, num_category, cats.item_cate_matrix, adj, num, logger)
        model.fit(train_dataset)#, valid_dataset)
        preds, truth = model.predict(test_dataset, conf['topk'])
        
    elif conf['model'] in ['ca_gcsan', 'dca_gcsan']:
        train_dataset = SRGNNDataset_Cat(train, conf, shuffle=True)
        test_dataset = SRGNNDataset_Cat(test, conf, shuffle=False)
        model = CA_GCSAN(cats.item_num, cats.n_cates, model_conf, logger) if conf['model']=='ca_gcsan' else DCA_GCSAN(cats.item_num, cats.n_cates, cats.item_cate_matrix, model_conf, logger)
        model.fit(train_dataset)#, valid_dataset)
        preds, truth = model.predict(test_dataset, conf['topk'])
        
    logger.info(f"Finish predicting, start calculating {conf['model']}'s KPI...")

#    # Top20
#    metrics, metrics_div, metrics_agg = performance_calculator(preds[:,:20], truth, cats.item_cate_matrix, ACC_KPI, DIV_KPI, AGG_KPI)
#
#    foo = ['%.4f'%(metrics[i]) for i in range(len(ACC_KPI))]
#    div_foo = ['%.4f'%(metrics_div[i]) for i in range(len(DIV_KPI))]
#    agg_foo = ['%.4f'%(metrics_agg[i]) for i in range(len(AGG_KPI))]
#
#    f.write(str(args.model)+','+str(args.dataset)+', top20, '+','.join(foo) +','+','.join(div_foo)+','+','.join(agg_foo) + '\n')
#
#    # Top10
#    metrics, metrics_div, metrics_agg = performance_calculator(preds[:,:10], truth, cats.item_cate_matrix, ACC_KPI, DIV_KPI, AGG_KPI)
#
#    foo = ['%.4f'%(metrics[i]) for i in range(len(ACC_KPI))]
#    div_foo = ['%.4f'%(metrics_div[i]) for i in range(len(DIV_KPI))]
#    agg_foo = ['%.4f'%(metrics_agg[i]) for i in range(len(AGG_KPI))]
#
#    f.write(str(args.model)+','+str(args.dataset)+', top10, '+','.join(foo) +','+','.join(div_foo)+','+','.join(agg_foo) + '\n')
#
#    # Top5
#    metrics, metrics_div, metrics_agg = performance_calculator(preds[:,:5], truth, cats.item_cate_matrix, ACC_KPI, DIV_KPI, AGG_KPI)
#
#    foo = ['%.4f'%(metrics[i]) for i in range(len(ACC_KPI))]
#    div_foo = ['%.4f'%(metrics_div[i]) for i in range(len(DIV_KPI))]
#    agg_foo = ['%.4f'%(metrics_agg[i]) for i in range(len(AGG_KPI))]
#
#    f.write(str(args.model)+','+str(args.dataset)+', top5, '+','.join(foo) +','+','.join(div_foo)+','+','.join(agg_foo) + '\n')
#
#    f.flush()

    return None

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gru4rec", type=str)
    parser.add_argument("--user_key", default="user_id", type=str)
    parser.add_argument("--item_key", default="item_id", type=str)
    parser.add_argument("--category_key", default="category_id", type=str)
    parser.add_argument("--session_key", default="session_id", type=str)
    parser.add_argument("--time_key", default="timestamp", type=str)
    parser.add_argument("--dataset", default="ml-100k", type=str)
    parser.add_argument("--desc", default="nothing", type=str)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("-seed", type=int, default=22, help="Seed for random initialization") #Random seed setting

    parser.add_argument('--batch_size', type=int, default=128, help='batch size for loader')
    parser.add_argument('--item_embedding_dim', type=int, default=100, help='dimension of item embedding')
    parser.add_argument('--category_embedding_dim', type=int, default=100, help='dimension of category embedding')
    parser.add_argument('--hidden_size', type=int, default=100, help='dimension of linear layer')
    parser.add_argument('--epochs', type=int, default=20, help='training epochs number')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2/BPR penalty')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--n_layers', type=int, default=1, help='the number of gru layers')
    parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
    parser.add_argument("--sigma", type=float, default=None, help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]") # weight initialization [-sigma sigma] in literature
    parser.add_argument('--dropout_input', default=0, type=float) #0.5 for TOP and 0.3 for BPR
    parser.add_argument('--dropout_hidden', default=0, type=float) #0.5 for TOP and 0.3 for BPR
    parser.add_argument('--optimizer', default='Adagrad', type=str)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--momentum', default=0.1, type=float)
    parser.add_argument('--eps', default=1e-6, type=float) #not used
    parser.add_argument('--final_act', default='tanh', type=str)
    parser.add_argument('--loss_type', default='BPR-max', type=str) #type of loss function TOP1 / BPR for GRU4REC, TOP1-max / BPR-max for GRU4REC+
    parser.add_argument('--pop_n', type=int, default=100, help='top popular N items')
    parser.add_argument('--n_sims', type=int, default=100, help='non-zero scores to the N most similar items given back')
    parser.add_argument('--lmbd', type=float, default=20, help='Regularization. Discounts the similarity of rare items')
    parser.add_argument('--alpha', type=float, default=0.5, help='Balance between normalizing with the supports of the two items')
    parser.add_argument('--lambda_session', type=float, default=0, help='session embedding penalty')
    parser.add_argument('--lambda_item', type=float, default=0, help='item embedding penalty')
    # add for gcegnn
    parser.add_argument('--activate', type=str, default='relu')
    parser.add_argument('--n_sample_all', type=int, default=12)
    parser.add_argument('--n_sample', type=int, default=12)
    parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
    parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
    parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
    parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
    # add for gcsan
    parser.add_argument('--weight', type=float, default=0.4, help='weight in final session embedding')     # [0.4, 0.8]
    parser.add_argument('--blocks', type=int, default=1)                                    # [1,2,3,4]

    # add for mcprn
    parser.add_argument('--tau', type=float, default=0.1, help='tau in softmax')     # [0.4, 0.8]
    parser.add_argument('--purposes', type=int, default=1, help='#purposes in mcprn')
    args = parser.parse_args()

    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    # begin tuning here
    tune_log_path = TUNE_PATH
    if not os.path.exists(tune_log_path):
        os.makedirs(tune_log_path)

#    if args.model in ['ca_narm', 'ca_gcegnn', 'ca_gcsan']:
#        seren_tune_log_path = '/nfsshare/home/yinqing/seren-main/tune_log/'
#        param_dict = json.load(open(seren_tune_log_path +f'hypers/'+args.model.split('_')[1]+'_'+ args.dataset+'.json', 'r'))
    if args.model[0] == 'd':
        param_dict = json.load(open(tune_log_path +f'hypers/'+args.model[1:]+'_'+ args.dataset+'.json', 'r'))
    else:
        param_dict = json.load(open(tune_log_path +f'hypers/'+args.model+'_'+ args.dataset+'.json', 'r'))
#        print(param_dict)
    acc_names = ['%s'%key for key in ACC_KPI]
    div_names = ['%s'%key for key in DIV_KPI]
    agg_names = ['%s'%key for key in AGG_KPI]
    f = open(tune_log_path+str(args.dataset)+'.txt','a',encoding='utf-8')
    f.write('model,dataset,'+','.join(acc_names)+','+','.join(div_names)+','+','.join(agg_names)+'\n')
    f.flush()
    
#    best = fmin(opt_func, param_dict, algo=tpe.suggest, max_evals=2) #20
    for i in range(1):
        best_func(param_dict, args, f)
        
    f.close()
