# DCA-SBRS

The official Pytorch code for 'A Simple Yet Effective Approach for Diversified Session-Based Recommendation'.

We provide our DCA framework, only Non-invasive Category-aware Attention (abbr. NCA) component, and only Model-agnostic Diversified Loss (abbr. MDL) component on SOTA SBRSs (i.e., NARM, STAMP, and GCE-GNN).

First, the folder '/DCA/model/' contains the above model's Pytorch implementation. 
* The naming rule for our DCA framework based on SOTA SBRS is 'DCA/model/dca_sbrs.py' (sbrs $\in$ [narm, stamp, gcegnn]). 
* The naming rule for only Non-invasive Category-aware Attention (abbr. NCA) component used with SOTA SBRS is 'DCA/model/ca_sbrs.py' (sbrs $\in$ [narm, stamp, gcegnn]). 
* The naming rule for only Model-agnostic Diversified Loss (abbr. MDL) component used with SOTA SBRS is 'DCA/model/ablation/DL/sbrs_dl.py' (sbrs $\in$ [narm, stamp, gcegnn]). 

Second, the folder '/DCA/utils/' contails the code for dataset pre-processing, Dataset Class definition, dataset processing for model input,  evaluation metrics' defination, and etc. 

Third, the folder '/DCA/tune_log/hypers' contains the optimal hyper-parameter setting for each SOTA SBRS on selected datasets. The dataset information is listed in '/DCA/tune_log/readme.txt'.



