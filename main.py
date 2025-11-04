import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from train import *
from test import *
from model import *
from utils import *
from dataloader import *
import warnings
from torch.utils.data import DataLoader, TensorDataset
import statistics

filename = 'config.yaml'
device ="cuda:0"

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    with open(filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    setseeds(config['seed'])


    split_num = config['split_num']
    skf = StratifiedKFold(n_splits=split_num, shuffle=True, random_state=config['seed'])

    dataset = config['dataset']
    task = config['task']


    data, label = get_networks(dataset, task)

    fold = 0
    Fold_metrics = []

    for train_index, test_index in skf.split(label, label):
        fold += 1
        print('------------------------------Fold: {}------------------------------'.format(fold))

        train_data = [data[i] for i in train_index]
        train_label = [label[i] for i in train_index]

        test_data = [data[i] for i in test_index]
        test_label = [label[i] for i in test_index]

        train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_label))
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_label))
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)


        model = Model(config).to(device)

        if config['test_only'] == False:

            train_model_process(model, train_dataloader, config, store_path="results/model" + str(fold) + ".pt")

        tn, fp, fn, tp, auc = test_model_process(model, test_dataloader, store_path="results/model" + str(fold) + ".pt")

        ACC = (tp + tn) / (tp + tn + fp + fn)
        print("Fold " + str(fold) + " ACC: " + str(ACC))
        Fold_metrics.append([tn, fp, fn, tp, auc])


    Mean_ACC, Mean_SEN, Mean_SPE, Mean_PRE, Mean_F1, Mean_AUC = [], [], [], [], [], []
    print("----------------------------------------------------------")
    for fold in range(config['split_num']):
        print("Fold:" + str(fold + 1) + " Testing Result")
        tn, fp, fn, tp, auc = Fold_metrics[fold]

        ACC = (tp + tn) / (tp + tn + fp + fn)
        Mean_ACC.append(ACC)

        SEN = tp / (tp + fn)
        Mean_SEN.append(SEN)

        SPE = tn / (tn + fp)
        Mean_SPE.append(SPE)

        PRE = tp / (tp + fp)
        Mean_PRE.append(PRE)

        F1 = 2 * PRE * SEN / (PRE + SEN)
        Mean_F1.append(F1)

        AUC = auc
        Mean_AUC.append(AUC)

        print("ACC: " + str(round(ACC * 100, 2)))
        print("SEN: " + str(round(SEN * 100, 2)))
        print("SPE: " + str(round(SPE * 100, 2)))
        print("PRE: " + str(round(PRE * 100, 2)))
        print("F1-score: " + str(round(F1 * 100, 2)))
        print("AUC: " + str(round(AUC * 100, 2)))
        print("----------------------------------------------------------")

    print("Mean ACC: " + str(round(sum(Mean_ACC) / config['split_num'] * 100, 2)) + "±" + str(
        round(np.std(Mean_ACC) * 100, 2)))
    print("Mean SEN: " + str(round(sum(Mean_SEN) / config['split_num'] * 100, 2)) + "±" + str(
        round(np.std(Mean_SEN) * 100, 2)))
    print("Mean SPE: " + str(round(sum(Mean_SPE) / config['split_num'] * 100, 2)) + "±" + str(
        round(np.std(Mean_SPE) * 100, 2)))
    print("Mean PRE: " + str(round(sum(Mean_PRE) / config['split_num'] * 100, 2)) + "±" + str(
        round(np.std(Mean_PRE) * 100, 2)))
    print("Mean F1-score: " + str(round(sum(Mean_F1) / config['split_num'] * 100, 2)) + "±" + str(
        round(np.std(Mean_F1) * 100, 2)))
    print("Mean AUC: " + str(round(sum(Mean_AUC) / config['split_num'] * 100, 2)) + "±" + str(
        round(np.std(Mean_AUC) * 100, 2)))
    print(Mean_ACC)

