import operator
import random
import pandas as pd
import numpy as np


# select positive samples and negative samples with the ratio of 1:'nb_imb'
def build_set(pos_key, neg_key, all_list, nb_imb=10):
    pos_ids = []
    neg_ids = []
    rand_dis = []
    for id in all_list:
        gene = id
        if gene in pos_key:
            pos_ids.append(id)
        elif gene in neg_key:
            neg_ids.append(id)
        else:
            rand_dis.append(id)
    rand_dis = random.sample(rand_dis, len(pos_ids) * nb_imb - len(neg_ids))
    neg_ids = list(set(rand_dis) | set(neg_ids))
    pos_ids.sort()
    neg_ids.sort()
    return pos_ids, neg_ids


# build the train set
def file2data(cancer_type, train_pos, train_neg, fea_type='origin'):
    if (fea_type == 'origin'):
        fea_one = '../feature/%s_train.csv' % (cancer_type)
    elif ('norm_exp' in fea_type):
        fea_one = '../feature/norm_exp/%s/%s_train.csv' % (fea_type[9:], cancer_type)
    else:
        fea_one = '../feature/%s/%s_train.csv' % (fea_type, cancer_type)
    df_one = pd.read_csv(fea_one, header=0, index_col=0, sep=',')
    feature_name = list(df_one.columns.values)
    mat_train_pos = df_one.loc[train_pos, ::]
    mat_train_neg = df_one.loc[train_neg, ::]
    gene_name = list(mat_train_pos.index) + list(mat_train_neg.index)
    mat_train_pos = mat_train_pos.values.astype(float)
    mat_train_neg = mat_train_neg.values.astype(float)
    X_train = np.concatenate([mat_train_pos, mat_train_neg])
    y_train = np.concatenate([np.ones((len(train_pos))), np.zeros((len(train_neg)))])
    return X_train, y_train, feature_name, gene_name


# get the train set to train model
def feature_input(cancer_type, nb_imb, fea_type='origin'):
    if (fea_type == 'origin'):
        # raw train dataset
        input = '../feature/%s_train.csv' % (cancer_type)
    elif ('norm_exp' in fea_type):
        input = '../feature/norm_exp/%s/%s_train.csv' % (fea_type[9:], cancer_type)
    else:
        input = '../feature/%s/%s_train.csv' % (fea_type, cancer_type)

    df_fea = pd.read_csv(input, header=0, index_col=0, sep=',')
    train_gene = list(df_fea.index)
    # gene_list contains genes with mutations
    df_gene = pd.read_csv('../input/gene.list', header=0, index_col=0, sep='\t')
    # intersection of sets, get the mutation genes from raw train dataset
    all_list = []
    for i in list(df_fea.index):
        if i in list(df_gene.index):
            all_list.append(i)
    # driver genes (oncogenes and tumor suppressor genes)
    key_2020 = '../input/train.txt'
    pd_key = pd.read_csv(key_2020, header=None, sep='\t')
    pd_key.columns = ['gene']
    pd_key = pd_key.drop_duplicates(subset=['gene'], keep='first')
    key_20 = pd_key['gene'].values.tolist()
    # identified passenger genes
    neg_key = ['CACNA1E', 'COL11A1', 'DST', 'TTN']
    # gene name of driver/passenger
    key_train_gene = set(key_20) & set(train_gene)
    neg_train_gene = set(neg_key) & set(train_gene)
    # select positive samples and negative samples with the ratio of 1:10
    pos, neg = build_set(key_train_gene, neg_train_gene, all_list, nb_imb)
    # build the train set
    X_train, y_train, feature_name, gene_name = file2data(cancer_type, pos, neg, fea_type)
    return X_train, y_train, feature_name, gene_name


# get the raw test data
def file2test(cancer_type, fea_type='origin'):
    if (fea_type == 'origin'):
        fea_test = '../feature/%s_test.csv' % (cancer_type)
    elif ('norm_exp' in fea_type):
        fea_test = '../feature/norm_exp/%s/%s_test.csv' % (fea_type[9:], cancer_type)
    else:
        fea_test = '../feature/%s/%s_test.csv' % (fea_type, cancer_type)
    df_test = pd.read_csv(fea_test, header=0, index_col=0, sep=',')
    feature_name = list(df_test.columns.values)
    gene_name = list(df_test.index)
    X = df_test.values.astype(float)
    return X, gene_name, df_test, feature_name


def feature_drop(x):
    fea = ['gene length', 'expression_CCLE', 'replication_time', 'HiC_compartment', 'gene_betweeness', 'gene_degree']
    for i in fea:
        x = x.drop(i, axis=1)
    return x


def feature_part(x, fea_nums):
    fea_b = ["cna_std", "5'UTR", "expression_CCLE", "lost start and stop", "replication_time", ]
    fea_t = ['recurrent missense', 'gene_degree', 'DEL', 'SNP', 'gene_betweeness']
    fea_cna = ['cna_mean', 'cna_std']
    fea_exp = ['exp_mean', 'exp_std']
    fea_methy = ['methy_mean', 'methy_std']
    fea_mut = ['silent', 'nonsense', 'splice site', 'missense', 'recurrent missense', 'frameshift indel',
               'inframe indel', 'lost start and stop',
               "3'UTR", "5'UTR", 'SNP', 'DEL', 'INS']
    fea_know = ['gene length', 'expression_CCLE', 'replication_time', 'HiC_compartment', 'gene_betweeness',
                'gene_degree']
    fea_mut_know = fea_mut + fea_know

    df_5 = x
    if fea_nums == 't5':
        df_5 = x[fea_t]
    elif fea_nums == 'b5':
        df_5 = x[fea_b]
    elif fea_nums == 'cna':
        df_5 = x[fea_cna]
    elif fea_nums == 'exp':
        df_5 = x[fea_exp]
    elif fea_nums == 'methy':
        df_5 = x[fea_methy]
    elif fea_nums == 'mut':
        df_5 = x[fea_mut]
    elif fea_nums == 'know':
        df_5 = x[fea_know]
    elif fea_nums == 'm_k':
        df_5 = x[fea_mut_know]

    return df_5


# pancan top5/ bottom5/ CNA/ gene expression...
def feature_input_part(cancer_type, nb_imb=1, fea_nums='t5'):
    input = '../feature/%s_train.csv' % (cancer_type)
    df_fea = pd.read_csv(input, header=0, index_col=0, sep=',')
    train_gene = list(df_fea.index)
    df_gene = pd.read_csv('../input/gene.list', header=0, index_col=0, sep='\t')
    all_list = []
    for i in list(df_fea.index):
        if i in list(df_gene.index):
            all_list.append(i)
    key_2020 = '../input/train.txt'
    pd_key = pd.read_csv(key_2020, header=None, sep='\t')
    pd_key.columns = ['gene']
    pd_key = pd_key.drop_duplicates(subset=['gene'], keep='first')
    key_20 = pd_key['gene'].values.tolist()
    neg_key = ['CACNA1E', 'COL11A1', 'DST', 'TTN']
    key_train_gene = set(key_20) & set(train_gene)
    neg_train_gene = set(neg_key) & set(train_gene)
    pos, neg = build_set(key_train_gene, neg_train_gene, all_list, nb_imb)
    X_train, y_train, feature_name, gene_name = file2data_part(cancer_type, pos, neg, fea_nums)
    return X_train, y_train, feature_name, gene_name


def file2data_part(cancer_type, train_pos, train_neg, fea_nums='t5'):
    X_train = []
    X = []
    fea_one = '../feature/%s_train.csv' % (cancer_type)
    df_one = pd.read_csv(fea_one, header=0, index_col=0, sep=',')
    df_one = feature_part(df_one, fea_nums)
    feature_name = list(df_one.columns.values)
    mat_train_pos = df_one.loc[train_pos, ::]
    mat_train_neg = df_one.loc[train_neg, ::]
    gene_name = list(mat_train_pos.index) + list(mat_train_neg.index)
    mat_train_pos = mat_train_pos.values.astype(float)
    mat_train_neg = mat_train_neg.values.astype(float)
    X_train.append(np.concatenate([mat_train_pos, mat_train_neg]))
    X.append(df_one.values.astype(float))
    y_train = np.concatenate([np.ones((len(train_pos))), np.zeros((len(train_neg)))])
    return X_train, y_train, feature_name, gene_name


def file2test_part(cancer_type, fea_nums):
    X = []
    fea_test = '../feature/%s_test.csv' % (cancer_type)
    df_test = pd.read_csv(fea_test, header=0, index_col=0, sep=',')
    df_test = feature_part(df_test, fea_nums)
    feature_name = list(df_test.columns.values)
    gene_name = list(df_test.index)
    X.append(df_test.values.astype(float))
    return X, gene_name, df_test, feature_name
