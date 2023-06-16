import math
import random
import torch
from subprocess import check_output
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
import torch.utils.data as Data
import pickle
from deepforest import CascadeForestClassifier
import sys
import numpy as np

seed = 1
random.seed(seed)


def dataLoader(x, y, batch_size):
    xt = torch.Tensor(x).float()
    yt = torch.Tensor(y).float()
    data = Data.TensorDataset(xt, yt)
    loader = Data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=0)
    return xt, yt, loader


def fit(Xs, y, type, fea_type):
    if (fea_type == 'origin'):
        model_path = '../model/%s.model' % (type)
        scaler_path = '../model/%s.scaler' % (type)
        path_cutoff = '../cutoff/cutoff_%s.csv' % (type)
    elif ('norm_exp' in fea_type):
        model_path = '../model/norm_exp/%s/%s.model' % (fea_type[9:], type)
        scaler_path = '../model/norm_exp/%s/%s.scaler' % (fea_type[9:], type)
        path_cutoff = '../cutoff/norm_exp/%s/cutoff_%s.csv' % (fea_type[9:], type)
    else:
        model_path = '../model/%s/%s.model' % (fea_type, type)
        scaler_path = '../model/%s/%s.scaler' % (fea_type, type)
        path_cutoff = '../cutoff/%s/cutoff_%s.csv' % (fea_type, type)
    scaler = preprocessing.RobustScaler()
    X = scaler.fit_transform(Xs)
    fp = open(scaler_path, 'wb')
    # save the scaler, and then use it in test data
    pickle.dump(scaler, fp)
    fp.close()
    del scaler
    model = CascadeForestClassifier(n_estimators=3, random_state=0, n_trees=50)
    model.fit(X, y)
    fp = open(model_path, 'wb')
    # save the trained model
    pickle.dump(model, fp)
    fp.close()
    del model

    model = pickle.load(open(model_path, 'rb'))
    out = model.predict_proba(X)[:, -1]
    fpr, tpr, thresholds = roc_curve(y, out)
    optimal, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    optimal_th = optimal

    df_cutoff = pd.DataFrame([optimal_th], columns={'cutoff'})
    df_cutoff.to_csv(path_cutoff)


def predict(Xs, type, fea_type='origin'):
    if (fea_type == 'origin'):
        model_path = '../model/%s.model' % (type)
        scaler_path = '../model/%s.scaler' % (type)
    elif ('norm_exp' in fea_type):
        model_path = '../model/norm_exp/%s/%s.model' % (fea_type[9:], type)
        scaler_path = '../model/norm_exp/%s/%s.scaler' % (fea_type[9:], type)
    else:
        model_path = '../model/%s/%s.model' % (fea_type, type)
        scaler_path = '../model/%s/%s.scaler' % (fea_type, type)
    scaler = pickle.load(open(scaler_path, 'rb'))
    X = scaler.transform(Xs)
    model = pickle.load(open(model_path, 'rb'))
    pre_y = model.predict_proba(X)[:, -1]
    return pre_y


# use Youden index to find optimal threshold
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


# the Fisher's exact test
def fisher(gene_sig, gene_nsig, key):
    gene_sig_true = [item for item in gene_sig if item in key]
    gene_nsig_true = [item for item in gene_nsig if item in key]
    sig_true_len = len(gene_sig_true)
    sig_flase_len = len(gene_sig) - len(gene_sig_true)
    nsig_true_len = len(gene_nsig_true)
    number_cancer_1 = 20000 - len(gene_sig) - nsig_true_len
    p = fisher_ex(sig_true_len, sig_flase_len, nsig_true_len, number_cancer_1)
    return p


def fisher_ex(a, b, c, d):
    _, pvalue = fisher_exact([[a, b], [c, d]], 'greater')
    if pvalue < 1e-256:
        pvalue = 1e-256
    # if you want to obtain the data without log, please select the code below
    # p1 = pvalue
    p1 = -math.log10(pvalue)
    return p1


# use the Mann-Whitney U test to calculate the p-value
def mannwhitneyu_(probas_, yy):
    yy = np.array(yy).flatten()
    x1_all = []
    x2_all = []
    for i in probas_[yy == 0]:
        x1_all.append(i)
    for i in probas_[yy == 1]:
        x2_all.append(i)
    statistic, pvalue = mannwhitneyu(x1_all, x2_all, use_continuity=True, alternative='two-sided')
    return pvalue


def fit_part(Xs, y, type, fea_nums):
    model_path = '../model/%s_%s.model' % (type, fea_nums)

    scaler_path = '../model/%s_%s.scaler' % (type, fea_nums)
    scaler = preprocessing.RobustScaler()
    X = scaler.fit_transform(Xs)

    fp = open(scaler_path, 'wb')
    pickle.dump(scaler, fp)
    fp.close()
    del scaler

    model = CascadeForestClassifier(n_estimators=3, random_state=0, n_trees=20)
    model.fit(X, y)
    fp = open(model_path, 'wb')
    pickle.dump(model, fp)
    fp.close()
    del model

    model = pickle.load(open(model_path, 'rb'))
    out = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, out)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    optimal, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    optimal_th = optimal
    print(optimal_th)
    path_cutoff = '../cutoff/cutoff_%s_%s.csv' % (type, fea_nums)
    df_cutoff = pd.DataFrame([optimal_th], columns={'cutoff'})
    df_cutoff.to_csv(path_cutoff)


def predict_part(Xs, type, fea_nums='t5'):
    model_path = '../model/%s_%s.model' % (type, fea_nums)
    scaler_path = '../model/%s_%s.scaler' % (type, fea_nums)
    scaler = pickle.load(open(scaler_path, 'rb'))
    X = scaler.transform(Xs)
    model = pickle.load(open(model_path, 'rb'))
    pre_y = model.predict_proba(X)[:, 1]
    return pre_y
