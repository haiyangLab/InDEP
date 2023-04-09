import argparse
import os
import sys
from matplotlib.pyplot import MultipleLocator
import matplotlib.colors as colors
from codes.base import fisher, mannwhitneyu_
from codes.pyvenn import venn
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
sys.path.append(r"/InDEP")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# other.py : drawing pictures showed in paper

# base1 base2: Filter and process the data of the comparison method according to the training set of your own model
# base1:process the data
# base2: after processing with base1, you can use base2 directly on the processed data
def base1(method, path):
    if method in ['2020plus', 'e-Driver', 'CompositeDriver','MutSig2CV', 'OncodriveFML', ]:
        df = pd.read_csv('../2018/%s/PANCAN.csv' % (method), sep=',')
    else:
        df = pd.read_csv('../2018/%s.csv' % (method), sep=',')
    df['in_2018'] = 0
    test_2018 = pd.read_csv(path, sep=',')
    test_2018_gene = test_2018['gene'].values.tolist()
    for index, row in df.iterrows():
        if df.loc[index, 'gene'] in test_2018_gene:
            df.loc[index, 'in_2018'] = 1
    newdf1 = df.loc[(df['in_2018'] == 1)]
    if method == '2020plus':
        newdf1 = newdf1.loc[(newdf1['info'] == 'TYPE=driver')]
    if '.' in newdf1['score'].tolist():
        print('yes')
        newdf1['score'] = newdf1[['pvalue']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    newdf1 = newdf1[['gene', 'score']]
    lab = newdf1['gene']
    if method != '2020plus':
        print(len(list(lab)) == len(set(lab)))
        newdf1 = newdf1.pivot_table(columns=["gene"])
        newdf1 = pd.DataFrame(newdf1.values.T, columns=newdf1.index, index=newdf1.columns)
    if method in ['2020plus', 'e-Driver', 'CompositeDriver',
                  'MutSig2CV', 'OncodriveFML',]:
        newdf1.to_csv('../2018/%s/PANCAN_1_10.csv' % method)
    else:
        newdf1.to_csv('../2018/%s_PANCAN_1_10.csv' % method)
    com = test_2018.loc[test_2018['gene'].isin(lab)]
    com = com[['gene', 'class']]
    com.columns = ['gene', 'class']
    c = pd.merge(newdf1, com, on='gene', how='outer')
    c['score'] = c['score'].astype('float64')
    list_x = c['score'].values.tolist()
    list_y = c['class'].values.tolist()
    return list_y, list_x

def base2(method, path):
    if method in ['2020plus',  'e-Driver', 'CompositeDriver','MutSig2CV', 'OncodriveFML' ]:
        df = pd.read_csv('../2018/%s/PANCAN_1_10.csv' % method, sep=',', header=0)
    elif method == 'InDEP':
        df = pd.read_csv('../PANCAN_test_performance.csv', sep=',', header=0)
    else:
        df = pd.read_csv('../2018/%s_PANCAN_1_10.csv' % method, sep=',', header=0)
    our_test = pd.read_csv(path, ',', header=0)
    our_test1 = our_test[['gene', 'class']]
    test_2018 = df[['gene', 'score']]
    c = pd.merge(our_test1, test_2018, on='gene', how='outer')
    driver_mean = c.loc[c['class'] == 1, 'score'].mean()
    non_driver_mean = c.loc[c['class'] == 0, 'score'].mean()
    m = c.loc[c['class'] == 0].fillna(value=float(non_driver_mean))
    n = c.loc[c['class'] == 1].fillna(value=float(driver_mean))
    p = pd.concat([m, n], ignore_index=True)
    list_x = p['score'].values.tolist()
    list_y = p['class'].values.tolist()
    return list_y, list_x

def draw_roc(list_y, list_x, method, color):
    # calculate roc
    fpr, tpr, thresholds1 = roc_curve(list_y, list_x)
    AUC = auc(fpr, tpr)
    title = 'CGC Roc Curve'
    print('cgcAUC %s:%0.3f' % (method, AUC))
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.grid(c='#E0E0E0')
    plt.plot(fpr, tpr, label='%s(AUC=%0.3f)' % (method, AUC), color=color)
    plt.title(title, y=-0.15)

def draw_pr(list_y, list_x, method, color):
    #  calculate pr
    precision, recall, thresholds = precision_recall_curve(list_y, list_x)
    AP = average_precision_score(list_y, list_x)
    title = 'CGC PR Curve'
    print('cgcPR %s:%0.3f' % (method, AP))
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.grid(c='#E0E0E0')
    plt.plot(recall, precision, label='%s(PR=%0.3f)' % (method, AP), color=color)
    plt.title(title, y=-0.15)

# draw pr curve and roc curve
def otherResults(path):
    methods = ['InDEP', '2020plus', 'DNsum', 'CompositeDriver', 'e-Driver', 'MutSig2CV', 'OncodriveFML', 'NetSig',
               'DriverML', 'WITER', ]
    colors = ['#4E659B', '#8A8CBF', '#B8A8CF', '#E7BCC6', '#FDCF9E', '#EFA484', '#B6766C', '#BEA6A1', '#A5D0C3',
              '#709A8D', ]
    save_path_auc = 'roc_pr/cgcAUC.jpg'
    save_path_pr = 'roc_pr/cgcPR.jpg'
    for m in range(len(methods)):
        i = methods[m]
        color = colors[m]
        # list_y, list_x = base1(i, path)
        list_y, list_x = base2(i, path)
        p_value = (mannwhitneyu_(np.array(list_x), np.array(list_y)))
        print(p_value)
        draw_roc(list_y, list_x, i, color)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path_auc, dpi=600)
    plt.show()
    for m in range(len(methods)):
        i = methods[m]
        color = colors[m]
        list_y, list_x = base2(i, path)
        draw_pr(list_y, list_x, i, color)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path_pr, dpi=600)
    plt.show()

# venn3 picture(InDEP, CGC and other metheds )
def drawVenn(A, B, C, method):
    A = set(A)
    B = set(B)
    C = set(C)
    labels = venn.get_labels([A, B, C], fill=['number'])
    venn.venn3(labels, names=['CGC', 'InDEP', '%s'%(method)],
                         colors=['#8A8CBFB3','#E7BCC6B3','#EFA484B3'],
                         edges=['black','black','black',], fontsize=18)
    plt.tight_layout()
    plt.savefig('pyvenn_%s.png'%method, dpi=600)
    plt.show()

# all methods' enrichment analysis result in PANCAN
def draw_pancan_bar(fisher, method):
    plt.figure(figsize=(4.7,5))
    plt.grid(ls='--', alpha=0.5)
    color = ['#104680','#6dadd1','#e9f1f4','#fbe3d5','#f8b191','#dc6e57',  '#6d021e']
    plt.xticks(rotation=90, fontsize=10, fontweight='bold')
    plt.bar(x=method, height=np.array(fisher), color=color,
                 edgecolor='black', zorder=100, label='value')
    plt.tight_layout()
    plt.savefig('pancan_fisher.png', dpi=600)
    plt.show()

# calculate percentage
def deal_percent(genes, methods):
    df = pd.DataFrame(index=['g1','g2','g3','g4'])
    for i in range(len(methods)):
        gene = genes[i]
        g1, g2, g3, g4 = 0, 0, 0, 0
        for g in gene:
            num = 0
            for m in genes:
                if g in m:
                    num += 1
            if num == 1:
                g1 += 1
            elif num == 2:
                g2 += 1
            elif num == 3:
                g3 += 1
            else:
                g4 += 1
            df[methods[i]] = [g1,g2,g3,g4]
    return df

# percentage column chart
def draw_percent_bar(data):
    x = data.index.tolist()
    y1 = data['g1'] / (data['g1'] + data['g2']+data['g3'] + data['g4'])
    y2 = data['g2'] / (data['g1'] + data['g2']+data['g3'] + data['g4'])
    y3 = data['g3'] / (data['g1'] + data['g2'] + data['g3'] + data['g4'])
    y4 = data['g4'] / (data['g1'] + data['g2'] + data['g3'] + data['g4'])
    colors= ['#1A2847','#3D8E86','#EBE1A9','#E67762']
    plt.figure(figsize=(7,5))
    plt.bar(x, y4,  label='predicted by 4-7 methods', color=colors[0], edgecolor='black', zorder=5)
    plt.bar(x, y3,  bottom=y4, label='predicted by 3 methods', color=colors[1], edgecolor='black', zorder=5)
    plt.bar(x, y2,  bottom=y3+y4, label='predicted by 2 methods', color=colors[2], edgecolor='black', zorder=5)
    plt.bar(x, y1,  bottom=1-y1, label='uniquely predicted', color=colors[3], edgecolor='black', zorder=5)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=90, fontsize=10, fontweight='bold',fontproperties='SimHei')
    plt.yticks( fontsize=10,)
    plt.grid(axis='y', alpha=0.5, ls='--')
    plt.legend(frameon=False, bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig('bar_num.png', dpi=600)
    plt.show()

# fisher -log10 P-values Box line diagram
def draw_box(fisher_dict, method):
    plt.figure(figsize=(6,3.5))
    box = []
    for i in method:
        box.append(fisher_dict[i])
    flierprops = dict(marker='o', markerfacecolor='black', markersize=2,
                      linestyle='none')
    facecolor = ['#6d021e','#b52330', '#dc6e57','#f8b191','#fbe3d5','#e9f1f4','#6dadd1','#317cb6','#104680',]
    f = plt.boxplot(box, labels=method, patch_artist=True,
                    vert=False, flierprops=flierprops, showcaps=False)
    for box, c in zip(f['boxes'], facecolor):
        box.set(color='black', linewidth=1)
        box.set(facecolor=c)
    for whisker in f['whiskers']:
        whisker.set(color='black', linewidth=1.2)
    for median in f['medians']:
        median.set(color='black', linewidth=1.2)
    plt.grid(axis='x', ls='--', alpha=0.5)
    my_x_ticks = np.arange(0, 60, 10)
    plt.xticks(my_x_ticks, fontsize=10, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('boxplot.png', dpi=600)
    plt.show()

# the number of candidate driver genes and enrichment analysis result of InDEP
def our_cgc(cancer):
    pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
    cgc_key = pd_cgc['Gene'].values.tolist()  # cgc
    res_file = '../score/InDEP/%s.score' % (cancer)
    df_score = pd.read_csv(res_file, index_col=0, header=0, sep=',')
    df = pd.read_csv('../cutoff/InDEP/cutoff_%s.csv' % (cancer), sep=',', index_col=0, header=0)
    cutoff = df['cutoff'].values.tolist()[0]
    gene_sig = df_score.loc[df_score['score'] > cutoff].index.tolist()
    gene_sig = set(gene_sig)
    gene_nsig = df_score.loc[df_score['score'] <= cutoff].index.tolist()
    gene_nsig = set(gene_nsig)
    b = fisher(gene_sig, gene_nsig, cgc_key)
    return cgc_key, gene_sig, b

# feature importance
def featureImportance(cancer):
    path = '../interpretation/InDEP/data/%s_shapValues.csv'%(cancer)
    all = pd.read_csv(path, index_col=0, header=0, sep=',')
    name = all.columns.tolist()
    a = dict.fromkeys(name, 0)
    for i in name:
        a[i] = all[i].abs().mean()
    b = dict(sorted(a.items(),key=lambda x:x[1]))
    return b

# draw feature importance in cancer level
def draw_cancer_importance_bar(cancer):
    b = featureImportance(cancer)
    key_name = list(b.keys())[::-1]
    key_value = list(b.values())[::-1]
    key_values = []
    sum_ = sum(key_value)
    for i in key_value:
        key_values.append(i/sum_)
    rgb = ([60,64,91],[94,108,117],[124,153,142],[130,178,154],[174,200,179],
           [189,210,188],[219,224,206],[244,241,222],[241,229,192],[236,216,172],
           [229,189,137],[221,167,120],[219,156,115],[216,142,107],[212,135,103], )
    rgb = np.array(rgb) / 255.0
    icmap = colors.ListedColormap(rgb, name='my_color')
    norm = plt.Normalize(min(key_values), max(key_values))
    norm_values = norm(key_values)
    color = icmap(norm_values)
    plt.figure(figsize=(6, 3))
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.bar(key_name, key_values, color=color,edgecolor='black')
    sm = cm.ScalarMappable(cmap=icmap, norm=norm)
    plt.colorbar(sm,shrink=0.7)
    plt.xticks(fontsize=9,rotation=90,fontweight='bold')
    plt.yticks(fontsize=9,fontweight='bold')
    plt.tight_layout()
    plt.savefig('importance/cancer_level/%s.png'%cancer, dpi=600, pad_inches=0.0)
    plt.show()

# draw feature importance in gene level(in paper)
def draw_gene_importance():
    sns.set()
    plt.rcParams['font.sans-serif'] = 'SimHei'
    np.random.seed(0)
    df = pd.read_excel('paper_gene.xlsx',header=0,index_col=0)
    fea_others = ['gene length', 'expression_CCLE', 'replication_time', 'HiC_compartment', 'gene_betweeness',
                  'gene_degree']
    for i in df.columns:
        if i in fea_others:
            df.drop(i,axis=1,inplace=True)
    f, ax = plt.subplots(figsize=( 18,5.5))
    sns.heatmap(df.T, ax=ax, vmin=-0.25, vmax=0.25, cmap='coolwarm',  linewidths=2, cbar=False)
    plt.xticks(fontsize=9, rotation=90,)
    plt.yticks(fontsize=9,)
    plt.tight_layout()
    plt.savefig('importance/gene_level/paper_gene.png', dpi=600, pad_inches=0.0)
    plt.show()

# draw feature importance in gene level(in Supplementary Material)
def draw_all_cancer_gene_importance(cancer,gene_name,feature,df):
    plt.figure(figsize=(10,10) )
    plt.title('%s'%cancer)
    plt.yticks(np.arange(len(gene_name)), gene_name, )
    plt.xticks(np.arange(len(feature)), feature, rotation=90)
    plt.imshow(df, cmap='coolwarm', vmin=-np.max(df.to_numpy()), vmax=np.max(df.to_numpy()))
    plt.title('%s feature importance (gene level)' % cancer)
    plt.colorbar(fraction=0.01)
    plt.tight_layout()
    plt.savefig('importance/gene_level/importance_%s.png'%cancer, dpi=600)
    plt.show()

def median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2

def draw_4_roc_pr(auc,pr,labels,num_data,colors):
    x= plt.figure(figsize=(8, 3))
    a = x.add_subplot(111)
    a.bar(x=labels, height=num_data, width=0.5, color=colors)
    b = a.twinx()
    b.plot(labels, auc, c='#313131', linewidth=1.4,marker='.')
    b.plot(labels, pr, c='#313131', linewidth=1.4,marker='x')
    y_major_locator = MultipleLocator(0.2)
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.yaxis.set_major_locator(y_major_locator)
    plt.legend(["A","B"],loc='upper left')
    plt.tight_layout()
    plt.savefig('omics/4_roc_pr.png' , dpi=600)
    plt.show()

def draw_4_fisher(labels,fisher_data,colors):
    plt.figure(figsize=(8, 3))
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.bar(x=labels, height=fisher_data, width=0.5, color=colors)
    plt.tight_layout()
    plt.savefig('omics/gene_fisher.png', dpi=600)
    plt.show()

def draw_4_genes(labels,num_data,colors):
    plt.figure(figsize=(8, 3))
    plt.bar(x=labels, height=num_data, width=0.5, color=colors)
    plt.tight_layout()
    plt.savefig('compare/gene_nums.png', dpi=600)
    plt.show()

# omic comparison (feature importance) bar
def draw_percent_bar_omics():
    x = ['PANCAN','BRCA','LUAD','GBM','UCEC',]
    # have calculated before, here for simplicity (see folder:analysis and run.py for detail)
    y1 = np.array([0.4736,0.3024,0.2602,0.1769,0.3298])
    y2 = np.array([0.0533,0.0699,0.0821,0.0784,0.0528])
    y3 = np.array([0.0509,0.0518,0.0739,0.0773,0.0774])
    y4 = np.array([0.0384,0.0390,0.0528,0.0551,0.0363])
    y5 = np.array([0.3837,0.5369,0.5310,0.6121,0.5038])
    colors=['#3D8E86', '#EBE1A9', '#EFA484', '#E67762', '#1A2847', ]
    plt.figure(figsize=(1.5,3))
    plt.bar(x, y4, label='CNA',width=0.5, color=colors[0], )
    plt.bar(x, y3,  bottom=y4,width=0.5,label='expression', color=colors[1], )
    plt.bar(x, y2,  bottom=y4+y3,width=0.5, label='Methylation', color=colors[2], )
    plt.bar(x, y1,  bottom=y3+y4+y2,width=0.5, label='mutation', color=colors[3], )
    plt.bar(x, y5,  bottom=1-y5,width=0.5, label='covariate', color=colors[4], )
    plt.ylim(0, 1.0)
    plt.xticks(rotation=90, fontsize=10, fontweight='bold',fontproperties='SimHei')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.yaxis.set_ticks_position('right')
    plt.yticks(fontsize=10, rotation=90)
    plt.tight_layout()
    plt.savefig('omics/cancer_percent.png', dpi=600)
    plt.show()

#omic comparison (feature importance) pie
def pie_cancer_feature_importance():
    cancer = ['PANCAN','GBM','UCEC','LUAD','BRCA']
    for c in cancer:
        df = pd.read_csv('../resultPic/GCF/data_feature_add/shap/all_genes/data/%s_shapValues.csv'%c,',',header=0,index_col=0)
        fea_cna = ['cna_mean', 'cna_std']
        fea_exp = ['exp_mean', 'exp_std']
        fea_methy = ['methy_mean', 'methy_std']
        fea_mut = ['silent', 'nonsense', 'splice site', 'missense', 'recurrent missense', 'frameshift indel',
                   'inframe indel', 'lost start and stop',
                   "3'UTR", "5'UTR", 'SNP', 'DEL', 'INS']
        fea_others = ['gene length', 'expression_CCLE', 'replication_time', 'HiC_compartment', 'gene_betweeness',
                    'gene_degree']
        columns_mean = df.abs().mean().to_frame().T
        f = []
        cna = columns_mean[fea_cna].loc[:, :].apply(lambda x: x.sum(), axis=1)[0]
        f.append(cna)
        exp = columns_mean[fea_exp].loc[:, :].apply(lambda x: x.sum(), axis=1)[0]
        f.append(exp)
        methy = columns_mean[fea_methy].loc[:, :].apply(lambda x: x.sum(), axis=1)[0]
        f.append(methy)
        mut = columns_mean[fea_mut].loc[:, :].apply(lambda x: x.sum(), axis=1)[0]
        f.append(mut)
        others = columns_mean[fea_others].loc[:, :].apply(lambda x: x.sum(), axis=1)[0]
        f.append(others)
        labels=['CNA ','Expression','Methylation','Mutation','Covariate']
        patchs,_,_=plt.pie(f, colors= ['#3D8E86','#EBE1A9','#EFA484','#E67762','#1A2847',],
                           autopct='%1.2f%%',pctdistance=1.2,textprops={'fontsize':14,'fontweight':'bold'} )
        if c == 'PANCAN':
            le = 'lower left'
        else:
            le = 'lower right'
        plt.legend(patchs,labels,loc=le)
        plt.title('%s'%c,fontsize=16,fontweight='bold')
        plt.tight_layout()
        plt.savefig('omics/pie_%s.png'%c, dpi=600,pad_inches=0.0)
        plt.show()

def main(argv=None):
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(description='InDEP')
    parser.add_argument("-m", dest='mode', default="paper_gene_importance", help="the picture you want to draw")
    args = parser.parse_args()

    # 1. ROC PR curve (fig.2a in paper)
    if args.mode == 'draw':
        test_path = '../PANCAN_test_performance.csv'
        cgc = '../input/cgc_somatic'
        otherResults(test_path)

    # 2. eachcancer's enrichment analysis result（Box line diagram）(fig.2b in paper)
    elif args.mode == 'fisher':
        pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
        cgc_key = pd_cgc['Gene'].values.tolist()  # cgc
        cutoff = 0.05
        method = ['NetSig', 'e-Driver', 'DriverML', 'CompositeDriver', 'MutSig2CV', 'WITER', 'OncodriveFML', '2020plus',
                  'InDEP', ]
        fisher_dict = {}
        gene_num_dict={}
        for i in method:
            print(i)
            gene_sig_list = []
            fisher_cgc = []
            gene_num = []
            cancers = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC',
                       'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'SARC', 'SKCM',
                       'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM', 'COAD', 'READ']
            if i == 'InDEP':
                for cancer in cancers:
                    _, gg, c = our_cgc(cancer)
                    fisher_cgc.append(c)
                    gene_num.append(gg)
            elif i in ['DriverML', 'NetSig']:
                path_driver = '../2018/%s/%s.xlsx' % (i, i)
                cancer_key = pd.read_excel(path_driver, sheet_name=None)
                cancer_key = list(cancer_key.keys())
                for cancer in cancers:
                    if cancer not in cancer_key:
                        continue
                    sheet = pd.read_excel(path_driver, sheet_name='%s' % cancer)
                    if i == 'DriverML':
                        drivergene = sheet[[i]].dropna()[i].tolist()
                    else:
                        drivergene = sheet['gene'].tolist()
                    file = '../2018/%s/%s.csv' % (i, cancer)
                    df = pd.read_csv(file, sep=',', index_col=0, header=0)
                    gene_all = df['6'].tolist()
                    gene_same_driver = set(drivergene) & set(gene_all)
                    gene_n = set(gene_all) - set(drivergene)
                    a = len(gene_same_driver)
                    c = fisher(gene_same_driver, gene_n, cgc_key)
                    gene_num.append(a)
                    fisher_cgc.append(c)

            elif i == 'WITER':
                sheet = pd.read_csv('../2018/WITER/WITER_driver.csv', header=0, index_col=0, sep=',')
                for cancer in cancers:
                    if cancer not in sheet.columns.tolist():
                        continue
                    cancer_df = sheet[[cancer]]
                    df_slice = cancer_df.loc[cancer_df[cancer] != '-']
                    drivergene = df_slice.index.tolist()
                    file = '../2018/%s/%s.csv' % (i, cancer)
                    df = pd.read_csv(file, sep=',', index_col=0, header=0)
                    gene_all = df['6'].tolist()
                    gene_same_driver = set(drivergene) & set(gene_all)
                    gene_n = set(gene_all) - set(drivergene)
                    a = len(gene_same_driver)
                    gene_sig_list.append(a)

                    c = fisher(gene_same_driver, gene_n, cgc_key)
                    gene_num.append(a)
                    fisher_cgc.append(c)
            else:
                if i in ['2020plus', 'CompositeDriver', 'e-Driver', ]:
                    cutoff = 0.05
                if i == 'MutSig2CV':
                    cutoff = 0.1
                if i == 'OncodriveFML':
                    cutoff = 0.25
                for cancer in cancers:
                    file = '../2018/%s/%s/%s.txt' % (i, i, cancer)
                    flag = os.path.exists(file)
                    if flag == False:
                        print(cancer)
                        continue
                    df = pd.read_csv(file, sep='\t')
                    gene_sig = df.loc[df['qvalue'] < cutoff]
                    gene_sig = gene_sig.drop_duplicates(['gene'])
                    gene_sig = gene_sig['gene'].values.tolist()

                    gene_nsig = df.loc[df['qvalue'] >= cutoff]
                    gene_nsig = gene_nsig.drop_duplicates(['gene'])
                    gene_nsig = gene_nsig['gene'].values.tolist()

                    a = len(gene_sig)
                    gene_sig_list.append(a)
                    c = fisher(gene_sig, gene_nsig, cgc_key)
                    fisher_cgc.append(c)
                    gene_num.append(a)
            medi = median(fisher_cgc)
            print(medi)
            fisher_dict[i] = fisher_cgc
            gene_num_dict[i] = gene_num
        draw_box(fisher_dict, method)

    # 3.venn3 picture (fig.S3)
    elif args.mode == 'venn':
        cancer = 'PANCAN'
        methods  = ['2020plus','MutSig2CV','OncodriveFML','e-Driver','CompositeDriver','NetSig']
        cgc_key, our_gene, _ = our_cgc(cancer)
        for method in methods:
            if method == 'NetSig':
                # NetSig provides driver gene
                path_driver = '../2018/%s/%s.xlsx' % (method, method)
                sheet = pd.read_excel(path_driver, sheet_name=cancer)
                gene_sig = sheet['gene'].tolist()

            else:
                # other methods provide threshold
                cutoff = 0.05
                if method == 'MutSig2CV':
                    cutoff = 0.1
                elif method  == 'OncodriveFML':
                    cutoff = 0.25
                file = '../2018/%s/%s/%s.txt' % (method, method, cancer)
                df = pd.read_csv(file, sep='\t')
                gene_sig = df.loc[df['qvalue'] < cutoff]
                gene_sig = gene_sig.drop_duplicates(['gene'])
                gene_sig = gene_sig['gene'].values.tolist()

            drawVenn(cgc_key, our_gene, gene_sig, method)

    # 4. enrichment result of InDEP in pancan (fig.2b in paper)
    elif args.mode == 'pan_fisher':
        pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
        cgc_key = pd_cgc['Gene'].values.tolist()  # cgc
        cutoff = 0.05
        method = ['InDEP', 'MutSig2CV', '2020plus', 'OncodriveFML', 'e-Driver', 'CompositeDriver', 'NetSig', ]
        cancer = 'PANCAN'
        fisher_cgc = []
        for i in method:
            print(i)
            if i == 'InDEP':
                _, _, c, = our_cgc(cancer)
                fisher_cgc.append(c)

            elif i == 'NetSig':
                path_driver = '../2018/%s/%s.xlsx' % (i, i)
                sheet = pd.read_excel(path_driver, sheet_name=cancer)
                drivergene = sheet['gene'].tolist()
                file = '../2018/%s.csv' % (i)
                df = pd.read_csv(file, sep=',', index_col=0, header=0)
                gene_all = df['gene'].tolist()
                gene_same_driver = set(drivergene) & set(gene_all)
                gene_n = set(gene_all) - set(drivergene)
                c = fisher(gene_same_driver, gene_n, cgc_key)
                fisher_cgc.append(c)

            else:
                if i in ['2020plus', 'CompositeDriver', 'e-Driver', ]:
                    cutoff = 0.05
                elif i == 'MutSig2CV':
                    cutoff = 0.1
                elif i == 'OncodriveFML':
                    cutoff = 0.25

                file = '../2018/%s/%s/%s.txt' % (i, i, cancer)
                flag = os.path.exists(file)
                if flag == False:
                    fisher_cgc.append(0)
                    continue
                df = pd.read_csv(file, sep='\t')
                gene_sig = df.loc[df['qvalue'] < cutoff]
                gene_sig = gene_sig.drop_duplicates(['gene'])
                gene_sig = gene_sig['gene'].values.tolist()
                gene_nsig = df.loc[df['qvalue'] >= cutoff]
                gene_nsig = gene_nsig.drop_duplicates(['gene'])
                gene_nsig = gene_nsig['gene'].values.tolist()
                c = fisher(gene_sig, gene_nsig, cgc_key)
                fisher_cgc.append(c)
        draw_pancan_bar(fisher_cgc, method)

    # 5. percentage column chart (fig.S2)
    elif args.mode == 'pan_gene':
        pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
        cgc_key = pd_cgc['Gene'].values.tolist()  # cgc
        cutoff = 0.05
        methods = ['InDEP','MutSig2CV','2020plus','OncodriveFML', 'e-Driver',  'CompositeDriver','NetSig', ]
        cancer = 'PANCAN'
        gene_methods = []

        for i in methods:
            print(i)
            if i == 'InDEP':
                _, c, _ = our_cgc(cancer)
                gene_methods.append(c)

            elif i == 'NetSig':
                path_driver = '../2018/%s/%s.xlsx' % (i, i)
                sheet = pd.read_excel(path_driver, sheet_name=cancer)
                drivergene = sheet['gene'].tolist()
                file = '../2018/%s.csv' % (i)
                df = pd.read_csv(file, sep=',', index_col=0, header=0)
                gene_all = df['gene'].tolist()
                gene_same_driver = set(drivergene) & set(gene_all)
                gene_methods.append(gene_same_driver)

            else:
                if i in ['2020plus', 'CompositeDriver', 'e-Driver', ]:
                    cutoff = 0.05
                elif i == 'MutSig2CV':
                    cutoff = 0.1
                elif i == 'OncodriveFML':
                    cutoff = 0.25
                file = '../2018/%s/%s/%s.txt' % (i, i, cancer)
                df = pd.read_csv(file, sep='\t')
                gene_sig = df.loc[df['qvalue'] < cutoff]
                gene_sig = gene_sig.drop_duplicates(['gene'])
                gene_sig = gene_sig['gene'].values.tolist()
                gene_methods.append(gene_sig)
        dff = deal_percent(gene_methods,methods)
        print(dff.T)
        draw_percent_bar(dff.T)

    # 6. note the differences between different methods (Table.S2)
    elif args.mode == 'methodsDriverNum':
        cutoff = 0.05
        method = ['InDEP', '2020plus', 'CompositeDriver', 'e-Driver', 'MutSig2CV', 'OncodriveFML', 'NetSig', 'DriverML',
                  'WITER']
        cancers = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC',
                   'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PANCAN', 'PCPG', 'PRAD', 'SARC', 'SKCM',
                   'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM', 'COAD', 'READ']
        cancers = ['PANCAN']
        writer = pd.ExcelWriter('../driverGeneInDifferentMethods111.xlsx')
        for cancer in cancers:
            gene_sig_dict = {}
            for i in method:
                print(i)
                if i == 'InDEP':
                    _, gene, _, = our_cgc(cancer)
                    gene_sig_dict[i] = gene

                elif i in ['DriverML', 'NetSig']:
                    path_driver = '../2018/%s/%s.xlsx' % (i, i)
                    cancer_key = pd.read_excel(path_driver, sheet_name=None)
                    cancer_key = list(cancer_key.keys())
                    if cancer not in cancer_key:
                        gene_sig_dict[i] = []
                        continue
                    sheet = pd.read_excel(path_driver, sheet_name='%s' % cancer)
                    if i == 'DriverML':
                        drivergene = sheet[[i]].dropna()[i].tolist()
                    else:
                        drivergene = sheet['gene'].tolist()

                    if cancer == 'PANCAN' and i=='NetSig':
                        file = '../2018/%s.csv' % (i)
                        df = pd.read_csv(file, sep=',', index_col=0, header=0)
                        gene_all = df['gene'].tolist()
                    else:
                        file = '../2018/%s/%s.csv' % (i, cancer)
                        df = pd.read_csv(file, sep=',', index_col=0, header=0)
                        gene_all = df['6'].tolist()
                    gene_same_driver = set(drivergene) & set(gene_all)
                    gene_sig_dict[i] = gene_same_driver

                elif i == 'WITER':
                    sheet = pd.read_csv('../2018/WITER/WITER_driver.csv', header=0, index_col=0, sep=',')
                    if cancer not in sheet.columns.tolist():
                        gene_sig_dict[i] = []
                        continue
                    cancer_df = sheet[[cancer]]
                    df_slice = cancer_df.loc[cancer_df[cancer] != '-']
                    drivergene = df_slice.index.tolist()
                    file = '../2018/%s/%s.csv' % (i, cancer)
                    df = pd.read_csv(file, sep=',', index_col=0, header=0)
                    gene_all = df['6'].tolist()
                    gene_same_driver = set(drivergene) & set(gene_all)
                    gene_sig_dict[i] = gene_same_driver

                else:
                    if i in ['2020plus', 'CompositeDriver', 'e-Driver', ]:
                        cutoff = 0.05
                    elif i == 'MutSig2CV':
                        cutoff = 0.1
                    elif i == 'OncodriveFML':
                        cutoff = 0.25

                    file = '../2018/%s/%s/%s.txt' % (i, i, cancer)
                    flag = os.path.exists(file)
                    if flag == False:
                        gene_sig_dict[i] = []
                        continue
                    df = pd.read_csv(file, sep='\t')
                    gene_sig = df.loc[df['qvalue'] < cutoff]
                    # print(list(gene_sig['gene']))
                    gene_sig = gene_sig.drop_duplicates(['gene'])
                    gene_sig = gene_sig['gene'].values.tolist()
                    gene_sig_dict[i] = gene_sig

            my_df = pd.DataFrame.from_dict(gene_sig_dict, orient='index').T
            my_df.to_excel(writer, sheet_name=cancer)
        writer.save()

    # 7. draw feature importance in cancer level (fig.4)
    elif args.mode == 'importance':
        draw_cancer_importance_bar('PANCAN')

    # 8. draw feature importance in gene level(in paper) (fig.3)
    elif args.mode == 'paper_gene_importance':
        draw_gene_importance()

    # 9. draw feature importance in gene level(fig.S4-S15)
    elif args.mode == 'all_cancer_gene_importance':
        cancers = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC',
                   'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP',
                   'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD',
                   'PANCAN', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM',
                   'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
        for cancer in cancers:
            if cancer in ['BRCA','PANCAN','GBM','LUAD','UCEC']:
                file = '../interpretation/InDEP/driver_gene/%s.csv'%cancer
            else:
                file = '../interpretation/InDEP/driver_gene/%s_shapValues.csv'%cancer
            df = pd.read_csv(file,index_col=0,header=0)
            if cancer in ['BRCA', 'PANCAN', 'GBM', 'LUAD', 'UCEC']:
                df.drop(['score'],axis=1,inplace=True)
            gene_name = df.index.tolist()
            feature = df.columns.tolist()
            draw_all_cancer_gene_importance(cancer,gene_name,feature,df)

    # 10. InDEP benefits from multi-omics (train model respectively) (fig.6)
    elif args.mode == 'train_model_omics_compare':
        labels = ['CNA ', 'Expression', 'Methylation', 'Mutation','Full Feature Set']
        colors = ['#3D8E86', '#EBE1A9', '#EFA484', '#E67762', '#1A2847']

        # CNA, Expression, Methylation, Mutation, Full feature set
        # have calculated before, here for simplicity (see folder:analysis and run.py for detail)
        auc_data = [0.531,0.605,0.572,0.724,0.805]
        pr_data = [0.112,0.130,0.113,0.336,0.460]
        num_data = [2548,4616,5664,253,183]
        fisher_data = [3.7,18.4,20.5,99.5,123.0]

        draw_4_roc_pr(auc_data,pr_data,labels,num_data,colors)
        draw_4_genes(labels,num_data,colors)
        draw_4_fisher(labels,fisher_data,colors)

    # 11. omic comparison (feature importance) (fig.6)
    elif args.mode == 'bar_omics_compare':
        draw_percent_bar_omics()

    # 12. omic comparison (feature importance) (fig.6)
    elif args.mode == 'pie_omics_compare':
        pie_cancer_feature_importance()

if __name__ == '__main__':
    main()


