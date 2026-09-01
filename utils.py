import numpy as np
import torch
import scipy
import math
from torch import nn
import torch.nn.functional as F
from model import SelfExpr, CommunityModel
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
import yaml
import pickle
import random
import os


def setseeds(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'




device = "cuda:0"



def dilute_adjacency(adj_matrix, sparsity):


    return



def enhance_sim_matrix(C, K, d, alpha):

    C = 0.5 * (C + C.T)
    r = min(d * K + 1, C.shape[0]-1)

    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))

    U = U[:,::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)
    L = np.abs(Z ** alpha)
    L = 0.5 * (L + L.T)
    L = L / L.max() 
    return L


def self_expressive_train(semodel, seoptimizer, x_train, config):
    alpha = 0.9
    x1 = x_train
    best_loss = 1e9

    for epoch in range(config['max_epoch_se']):
        semodel.train()
        seoptimizer.zero_grad()
        c, x2 = semodel(x1)
        se_loss = torch.norm(x1 - x2, p='fro')  
        reg_loss = torch.norm(c, p='fro')  
        loss = se_loss + alpha * reg_loss  
        loss.backward()
        seoptimizer.step()
        print('se_loss: {:.9f}'.format(se_loss.item()), 'reg_loss: {:.9f}'.format(reg_loss.item()), end=' ')
        print('full_loss: {:.9f}'.format(loss.item()), flush=True)
        if loss.item() < best_loss:
            if torch.cuda.is_available():
                best_c = c.cpu()
            else:
                best_c = c
            best_loss = loss.item()

    C = best_c
    C = C.cpu().detach().numpy()


    L = enhance_sim_matrix(C, config['n_class'], 3, 1)


    return L


def cluster_train(clustermodel, clusteroptimizer, x, from_list, to_list, val_list, config, num_rois, com_label):
    lambda2 = 0.1 
    best_loss = 1e9

    for epoch in range(config['max_epoch_cd']):
        epoch_loss = 0
        clustermodel.train()
        clusteroptimizer.zero_grad()
        for i in range(x.shape[0]):
            z_full = clustermodel(x[i]) 
            z_from = z_full[from_list[i]]
            z_to = z_full[to_list[i]]

            pred_similarity = torch.sum(z_from * z_to, dim=1)
            contri_loss = F.mse_loss(pred_similarity, torch.tensor(val_list[i]).to(device))
            prior_loss = F.cross_entropy(z_full, com_label)
            loss = contri_loss + lambda2 * prior_loss
            epoch_loss += loss

        epoch_loss.backward()
        clusteroptimizer.step()
        long_string = f"epoch {epoch + 1} train loss:{epoch_loss:.5f}"

        if epoch_loss.item() < best_loss:
            best_loss = epoch_loss.item()
            torch.save(clustermodel.state_dict(), "clustermodel" + str(num_rois))
            long_string += " --> Best model ever (stored)"

        print(long_string)

    return best_loss


def create_adj_incidence_matrix(dataset, num_rois, config, com_label):
    # Self-expressive model
    semodel = SelfExpr(num_rois).to(device)
    seoptimizer = torch.optim.Adam(semodel.parameters(), lr=config['se_lr'], weight_decay=5e-04)

    # Brain Module Dection Model
    clustermodel = CommunityModel(num_rois, 64, config['n_class'], 0.1).to(device)
    clusteroptimizer = torch.optim.Adam(clustermodel.parameters(), lr=config['cd_lr'], weight_decay=5e-04)

    num_samples = len(dataset)

    from_list_all = []
    to_list_all = []
    val_list_all = []

    A = []

    # ============== Self-Expressive Layer training Module ================#
    for i in range(num_samples):
        from_list = [] 
        to_list = [] 
        val_list = []  
        x_train = dataset[i]

        print("\n\n\nStarting self expressive train iteration:", i + 1)
        S = self_expressive_train(semodel, seoptimizer, x_train, config)  
        a = S
        A.append(a)

        threshold = config['threshold']
        for i in range(num_rois):
            for j in range(num_rois):
                if i == j:
                    continue
                if S[i, j] >= (1 - threshold) or (S[i, j] <= threshold and S[i, j] >= 0):
                    from_list.append(i)
                    to_list.append(j)
                    val_list.append(S[i, j])

        from_list_all.append(from_list)
        to_list_all.append(to_list)
        val_list_all.append(val_list)

    print("Self Expressive Layer training done...")


    # ============== Cluster training Module ================#
    print("\n\n\nStarting cluster training module")

    convergence_loss = cluster_train(clustermodel, clusteroptimizer, dataset, from_list_all, to_list_all, val_list_all, config, num_rois, com_label)

    print('convergence loss: {:.5f}'.format(convergence_loss))

    print("Cluster model training done...")

    # ============== Brain Module Detection ================#
    clustermodel.load_state_dict(torch.load("clustermodel" + str(num_rois)))
    clustermodel.eval()

    C = []
    for i in range(num_samples):
        c = clustermodel(dataset[i]).cpu().detach().numpy()
        C.append(c)


    adj_node = []
    adj_com = []
    for i in range(num_samples):
        adj_n = dilute_adjacency(A[i], config['sparsity_n'])
        adj_node.append(adj_n)

        adj_c = C[i].T @ A[i] @ C[i]  
        adj_c = dilute_adjacency(adj_c, config['sparsity_c'])
        adj_com.append(adj_c)

    return adj_node, adj_com, C
