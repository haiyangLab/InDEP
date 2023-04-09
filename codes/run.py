import sys
from sklearn.metrics import average_precision_score
from codes.base import *
from codes.interpret import *
import sys
import argparse
sys.path.append(r"/InDEP")

def main(argv=None):
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(description='ExpDriver v1.0')
    parser.add_argument('-c',dest='key',default='./key.csv',help='coding file')
    parser.add_argument('-s',dest='pos',default='./pos_2020.txt',help='coding file')
    parser.add_argument('-g',dest='neg',default='./neg_2020.txt',help='non_codong file')
    parser.add_argument('-m',dest='mode',default='train',help='mode') #train,score,eval,interpretation
    parser.add_argument('-l',dest='learn',default='InDEP',help='model type') #RF
    parser.add_argument('-t',dest='type',default='PANCAN',help='cancer type') #PANCAN
    parser.add_argument('-o',dest='out',default='../score/',help='coding file')
    parser.add_argument('-p',dest='threads_num',type=int,default=1,help='threads num')
    parser.add_argument('-imb', dest='imb', type=int, default=10, help='the ratio of positive and negative sample')
    parser.add_argument('-fn', dest='featureNums', default='t5', help='use part of features')  # t5 b5 cna mut methy exp know m_k
    args = parser.parse_args()

    if args.mode == 'train':
        # a positive and negative sample ratio of 1:10
        nb_imb = 10
        X_train, y_train, _, _ = feature_input(args.type,nb_imb)
        fit(X_train, y_train, args.type, args.learn)

    # Scoring on the test set
    elif args.mode == 'score':
        # get all genes with mutations
        genes = set(pd.read_csv('../input/gene.list', header=0, index_col=0, sep='\t').index.tolist())

        # X: raw test data(contains all genes)
        # ids: gene name of raw test data
        X, ids,_ ,_ = file2test(args.type)
        y_p = predict(X, args.type, method=args.learn)

        # predict: raw test data
        df_score = pd.DataFrame(np.nan, index=ids, columns=['score'])
        df_score.loc[ids, 'score'] = y_p
        # predict: genes with mutation in test data
        df_all = pd.DataFrame(np.nan, index=genes, columns=['score'])
        g_and = set(genes) & set(ids)
        df_all.loc[g_and, 'score'] = df_score.loc[g_and, 'score']
        # predict: genes with mutation not in test data marked 0
        df_all['score'].fillna(0, inplace=True)
        out_path = '%s%s/%s.score' % (args.out, args.learn, args.type)
        df_all = df_all.sort_values(by=['score'], ascending=[False])
        df_all.to_csv(out_path, header=True)

    # Result analysis
    elif args.mode == 'eval':
        input = "../feature/%s_test.csv" % (args.type)
        df_tmp = pd.read_csv(input, header=0, index_col=0, sep=',')
        row_list = df_tmp.index.tolist()
        pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
        # cgc
        cgc_key = pd_cgc['Gene'].values.tolist()
        res_file = '%s%s/%s.score' % (args.out, args.learn, args.type)
        # prediction of test data
        df_score = pd.read_csv(res_file, index_col=0,header=0, sep=',')
        score_gene = df_score.index.tolist()
        all_list = []
        for i in row_list:
            if i in score_gene:
                all_list.append(i)
        pos_cgc, neg_cgc = build_set(cgc_key, [], all_list, nb_imb=10)
        df_pos_cgc = df_score.loc[pos_cgc, ::]
        df_pos_cgc['score'].fillna((df_pos_cgc['score'].mean()), inplace=True)
        score_pos_cgc = df_pos_cgc['score'].values.tolist()
        df_neg_cgc = df_score.loc[neg_cgc, ::]
        df_neg_cgc['score'].fillna((df_neg_cgc['score'].mean()), inplace=True)
        score_neg_cgc = df_neg_cgc['score'].values.tolist()
        # true
        y_cgc = np.concatenate([np.ones((len(score_pos_cgc))), np.zeros((len(score_neg_cgc)))])
        # predict
        y_p_cgc = np.concatenate([score_pos_cgc, score_neg_cgc])

        if args.type == 'PANCAN':
            print('yes')
            gene_cgc = pos_cgc + neg_cgc
            df_PANCAN_gene = pd.DataFrame(gene_cgc,columns=['gene'])
            df_PANCAN_gene.set_index('gene',inplace=True)
            df_PANCAN_gene['score'] = y_p_cgc
            df_PANCAN_gene['class'] = y_cgc
            # use in drawing picture later
            df_PANCAN_gene.to_csv('../PANCAN_test_performance.csv')

        auprc = '%.3f'%(average_precision_score(y_cgc, y_p_cgc))
        fpr, tpr, thresholds = roc_curve(y_cgc, y_p_cgc)
        auroc = '%.3f' % auc(fpr, tpr)
        print('performance(pr,auc):',auprc, auroc)
        p_value = (mannwhitneyu_(y_p_cgc,y_cgc))
        print('p_value', p_value)

        # fisher
        path_cutoff = '../cutoff/%s/cutoff_%s.csv' % (args.learn,args.type)
        df = pd.read_csv(path_cutoff,sep=',',index_col=0,header=0)
        cutoff = df['cutoff'].values.tolist()[0]

        gene_sig = df_score.loc[df_score['score'] > cutoff].index.tolist()
        gene_sig = set(gene_sig)
        gene_nsig = df_score.loc[df_score['score'] <= cutoff].index.tolist()
        gene_nsig = set(gene_nsig)

        a = len(gene_sig)
        print('the number of candidate driver genes: %d,'% a)
        b = fisher(gene_sig, gene_nsig, cgc_key)
        print('fisher:', '%0.1f' % b)

    # interpretation
    elif args.mode == 'interpretation':
        interpretion_shap(type=args.type, method=args.learn)

    # if you want to get the data in Table 1 or fig.5/6,  you can use the codes below
    elif args.mode == 'train_part':
        nb_imb = 10
        X_train, y_train, featureName, _ = feature_input_part(args.type,nb_imb,args.featureNums)
        fit_part(X_train, y_train, args.type, args.learn, fea_nums=args.featureNums)

    elif args.mode == 'score_part':
        genes = set(pd.read_csv('../input/gene.list', header=0, index_col=0, sep='\t').index.tolist())
        X, ids,_ ,_ = file2test_part(args.type,fea_nums=args.featureNums)
        y_p = predict_part(X, args.type, method=args.learn,fea_nums=args.featureNums)
        df_score = pd.DataFrame(np.nan, index=ids, columns=['score'])
        df_score.loc[ids, 'score'] = y_p
        df_all = pd.DataFrame(np.nan, index=genes, columns=['score'])
        g_and = set(genes) & set(ids)
        df_all.loc[g_and, 'score'] = df_score.loc[g_and, 'score']
        df_all['score'].fillna(0, inplace=True)
        out_path = '%s%s/%s_%s.score' % (args.out, args.learn, args.type,args.featureNums)
        if args.featureInput == 'less':
            out_path = '%s%s/%s_less.score' % (args.out, args.learn, args.type)
        df_all = df_all.sort_values(by=['score'], ascending=[False])
        df_all.to_csv(out_path, header=True)

    elif args.mode == 'eval_part':
        input = "../feature/%s_test.csv" % (args.type)
        df_tmp = pd.read_csv(input, header=0, index_col=0, sep=',')
        all_list1 = df_tmp.index.tolist()
        pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
        cgc_key = pd_cgc['Gene'].values.tolist()  # cgc
        res_file = '%s%s/%s_%s.score' % (args.out, args.learn, args.type, args.featureNums)
        df_score = pd.read_csv(res_file, index_col=0,header=0, sep=',')
        score_gene = df_score.index.tolist()
        all_list = []
        for i in all_list1:
            if i in score_gene:
                all_list.append(i)
        pos_cgc, neg_cgc = build_set(cgc_key, [], all_list, nb_imb=10)
        df_pos_cgc = df_score.loc[pos_cgc, ::]
        df_pos_cgc['score'].fillna((df_pos_cgc['score'].mean()), inplace=True)
        score_pos_cgc = df_pos_cgc['score'].values.tolist()
        df_neg_cgc = df_score.loc[neg_cgc, ::]
        df_neg_cgc['score'].fillna((df_neg_cgc['score'].mean()), inplace=True)
        score_neg_cgc = df_neg_cgc['score'].values.tolist()
        y_cgc = np.concatenate([np.ones((len(score_pos_cgc))), np.zeros((len(score_neg_cgc)))])
        y_p_cgc = np.concatenate([score_pos_cgc, score_neg_cgc])

        if args.type == 'PANCAN':
            gene_cgc = pos_cgc + neg_cgc
            df_PANCAN_gene = pd.DataFrame(gene_cgc,columns=['gene'])
            df_PANCAN_gene.set_index('gene',inplace=True)
            df_PANCAN_gene['score'] = y_p_cgc
            df_PANCAN_gene['class'] = y_cgc
            df_PANCAN_gene.to_csv('../analysis/PANCAN_test_performance_%s.csv' % args.featureNums)

        auprc = '%.3f'%(average_precision_score(y_cgc, y_p_cgc))
        fpr, tpr, thresholds = roc_curve(y_cgc, y_p_cgc)
        auroc = '%.3f' % auc(fpr, tpr)
        print('performance',auprc, auroc)
        path_cutoff = '../cuttoff/%s/cutoff_%s_%s.csv' % (args.learn,args.type,args.featureNums)
        df = pd.read_csv(path_cutoff,sep=',',index_col=0,header=0)
        cutoff = df['cutoff'].values.tolist()[0]
        gene_sig = df_score.loc[df_score['score'] > cutoff].index.tolist()
        gene_sig = set(gene_sig)
        gene_nsig = df_score.loc[df_score['score'] <= cutoff].index.tolist()
        gene_nsig = set(gene_nsig)
        a = len(gene_sig)
        print('the number of candidate driver genes: %d,'% a)
        b = fisher(gene_sig, gene_nsig, cgc_key)
        print('fisher:', '%0.1f' % b)


if __name__ == "__main__":
    main()