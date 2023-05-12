import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import normal_
epsilon = 1e-4
from torch.distributions import Categorical
from scipy.stats import entropy

class STAMP_DL(nn.Module):
    def __init__(self, n_items, item_cate_matrix, params, logger):
        '''

        '''        
        super(STAMP_DL, self).__init__()
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.logger = logger
        # parameters
        self.n_items = n_items # already with padding
        self.item_cate_matrix = item_cate_matrix
        self.embedding_size = params['item_embedding_dim']
        # Embedding layer
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.w1 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w2 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w3 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w0 = nn.Linear(self.embedding_size, 1, bias=False)
        self.b_a = nn.Parameter(torch.zeros(self.embedding_size), requires_grad=True)
        self.mlp_a = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.mlp_b = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.sf = nn.Softmax(dim=1) #nn.LogSoftmax(dim=1)
        
        self.loss_function = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=params['lr_dc_step'], gamma=params['lr_dc'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.002)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.05)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
                
    def count_alpha(self, context, aspect, output):
        r"""This is a function that count the attention weights
        Args:
            context(torch.FloatTensor): Item list embedding matrix, shape of [batch_size, time_steps, emb]
            aspect(torch.FloatTensor): The embedding matrix of the last click item, shape of [batch_size, emb]
            output(torch.FloatTensor): The average of the context, shape of [batch_size, emb]
        Returns:
            torch.Tensor:attention weights, shape of [batch_size, time_steps]
        """
        timesteps = context.size(1)
        aspect_3dim = aspect.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)
        output_3dim = output.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)
        res_ctx = self.w1(context)
        res_asp = self.w2(aspect_3dim)
        res_output = self.w3(output_3dim)
        res_sum = res_ctx + res_asp + res_output + self.b_a
        res_act = self.w0(self.sigmoid(res_sum))
        alpha = res_act.squeeze(2)
        return alpha
        
    def forward(self, seq, lengths):
        batch_size = seq.size(1)
        seq = seq.transpose(0,1)
        item_seq_emb = self.item_embedding(seq) # [b, seq_len, emb]

        lengths = torch.Tensor(lengths).to(self.device)
        item_last_click_index = lengths - 1
        item_last_click = torch.gather(seq, dim=1, index=item_last_click_index.unsqueeze(1).long()) # [b, 1]
        last_inputs = self.item_embedding(item_last_click.squeeze())# [b, emb]
        org_memory = item_seq_emb # [b, seq_len, emb]
        ms = torch.div(torch.sum(org_memory, dim=1), lengths.unsqueeze(1).float())# [b, emb]
        alpha = self.count_alpha(org_memory, last_inputs, ms) # [b, seq_len]
        vec = torch.matmul(alpha.unsqueeze(1), org_memory) # [b, 1, emb]
        ma = vec.squeeze(1) + ms # [b, emb]
        hs = self.tanh(self.mlp_a(ma))
        ht = self.tanh(self.mlp_b(last_inputs))
        seq_output = hs * ht
        item_embs = self.item_embedding(torch.arange(self.n_items).to(self.device))
        scores = torch.matmul(seq_output, item_embs.permute(1, 0))
        item_scores = self.sf(scores)
        return item_scores

    def fit(self, train_loader, validation_loader=None):
        self.cuda() if torch.cuda.is_available() else self.cpu()
        self.logger.info('Start training...')
        
        for epoch in range(1, self.epochs + 1):  
            self.train()          
            total_loss = []
            for i, (seq, seq_cate, target, target_cate, lens) in enumerate(train_loader):
                self.optimizer.zero_grad()
                scores = self.forward(seq.to(self.device), lens)
                loss_ce = self.loss_function(torch.log(scores.clamp(min=1e-9)), target.squeeze().to(self.device))
                
                category_scores_consideringNum = scores.matmul(self.item_cate_matrix.to(self.device))
                Entropy_pre = Categorical(probs = category_scores_consideringNum[:,1:]).entropy()
                loss_dl = Entropy_pre.mean()
                
                loss = loss_ce - loss_dl
                
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())



    def predict(self, test_loader, k=15):
        self.eval()  
        preds, last_item = torch.tensor([]), torch.tensor([])
        for _, (seq, seq_cate, target, target_cate, lens) in enumerate(test_loader):
            scores = self.forward(seq.to(self.device), lens)
            rank_list = (torch.argsort(scores[:,1:], descending=True) + 1)[:,:k]  # why +1: +1 to represent the actual code of items

            preds = torch.cat((preds, rank_list.cpu()), 0)
            last_item = torch.cat((last_item, target), 0)

        return preds, last_item

        

