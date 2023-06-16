import sys
import argparse
sys.path.append(r"/InDEP")
from sklearn.metrics import average_precision_scor
from codes.base import *
from codes.interpret import *


def main(argv=None):
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(description='InDEP')
    parser.add_argument('-c', dest='key', default='./key.csv', help='coding file')
    parser.add_argument('-s', dest='pos', default='./pos_2020.txt', help='coding file')
    parser.add_argument('-g', dest='neg', default='./neg_2020.txt', help='non_codong file')
    parser.add_argument('-m', dest='mode', default='eval', help='mode')  # train,score,eval,interpretation,
    parser.add_argument('-t', dest='type', default='PANCAN', help='cancer type')  # PANCAN
    parser.add_argument('-p', dest='threads_num', type=int, default=1, help='threads num')
    parser.add_argument('-imb', dest='imb', type=int, default=10, help='the ratio of positive and negative sample')
    parser.add_argument('-fn', dest='featureNums', default='t5',
                        help='use part of features')  # t5 b5 cna mut methy exp know m_k
    parser.add_argument('-r', dest='feature', default='origin',
                        help='different features')  # origin,noncoding,cnv,CADD,subtype,norm_exp
    args = parser.parse_args()

    fea_type = args.feature
    scaler = args.scaler
    nb_imb = 10

    # extract the features
    if args.mode == 'fea':
        path_2020plus = '../../2020plus/'
        p = 16
        sample_train = pd.read_csv('../input/sample_subtype_train.csv', header=0, sep=',')
        sample_test = pd.read_csv('../input/sample_subtype_test.csv', header=0, sep=',')
        c = args.type
        # 拿到cancer的数据
        if c == 'PANCAN':
            sample_train = sample_train.copy()
            sample_test = sample_test.copy()
        elif c == 'COADREAD':
            sample_train = sample_train[sample_train['type'].isin(set(['COAD', 'READ']))]
            sample_test = sample_test[sample_test['type'].isin(set(['COAD', 'READ']))]
        else:
            sample_train = sample_train[sample_train['type'] == c]
            sample_test = sample_test[sample_test['type'] == c]

        # 拿到亚型名称 及 数目
        subtype_name = set(sample_train['subtype'].tolist())
        subtype_num = len(subtype_name)

        # 对每一个亚型创建数据集
        for subtype in subtype_name:
            # 得到亚型的数据
            sample_train = sample_train[sample_train['subtype'] == subtype]
            sample_test = sample_test[sample_test['subtype'] == subtype]

            sample_train_set = set(sample_train['bcr_patient_barcode'].tolist())
            sample_test_set = set(sample_test['bcr_patient_barcode'].tolist())
            print(len(sample_train_set), len(sample_test_set))
            input = '../input/tcga_18.maf'
            df1 = pd.read_csv(input, header=0, sep='\t')
            df1['Tumor_Sample'] = df1['Tumor_Sample_Barcode'].apply(lambda x: '-'.join(str(x).split('-')[0:3]))
            df1['v1'] = df1['Tumor_Sample_Barcode'].apply(lambda x: str(x).split('-')[3][0])
            df1 = df1.loc[df1['v1'] == '0', ['Gene', 'Tumor_Sample', "Chromosome", "Start_Position", "End_Position",
                                             "Reference_Allele",
                                             "Tumor_Allele",
                                             "Variant_Classification", "Variant_Type"]]
            df1_train = df1[df1['Tumor_Sample'].isin(sample_train_set)]
            df1_test = df1[df1['Tumor_Sample'].isin(sample_test_set)]
            maf_train = '../input/%s_train.maf' % (args.type)
            maf_test = '../input/%s_test.maf' % (args.type)
            print(df1_train.shape, df1_test.shape)
            df1_train.to_csv(maf_train, header=True, index=False, sep='\t')
            df1_test.to_csv(maf_test, header=True, index=False, sep='\t')
            # oncogene
            cmd = "probabilistic2020 oncogene -i %s/data/snvboxGenes.fa -b %s/data/snvboxGenes.bed -s %s/data/scores -m %s -c 1.5 -p %d -o ./train/%s_oncogene.txt" % (
                path_2020plus, path_2020plus, path_2020plus, maf_train, p, args.type)
            print(cmd)
            check_output(cmd, shell=True)
            cmd = "probabilistic2020 oncogene -i %s/data/snvboxGenes.fa -b %s/data/snvboxGenes.bed -s %s/data/scores -m %s -c 1.5 -p %d -o ./test/%s_oncogene.txt" % (
                path_2020plus, path_2020plus, path_2020plus, maf_test, p, args.type)
            print(cmd)
            check_output(cmd, shell=True)
            # tsg
            cmd = "probabilistic2020 tsg -i %s/data/snvboxGenes.fa -b %s/data/snvboxGenes.bed -m %s -c 1.5 -p %d -o ./train/%s_tsg.txt" % (
                path_2020plus, path_2020plus, maf_train, p, args.type)
            print(cmd)
            check_output(cmd, shell=True)
            cmd = "probabilistic2020 tsg -i %s/data/snvboxGenes.fa -b %s/data/snvboxGenes.bed -m %s -c 1.5 -p %d -o ./test/%s_tsg.txt" % (
                path_2020plus, path_2020plus, maf_test, p, args.type)
            print(cmd)
            check_output(cmd, shell=True)
            # sim, oncogene, tsg
            # cmd = "mut_annotate --maf -n 1 -i %s/data/snvboxGenes.fa -b %s/data/snvboxGenes.bed -m %s -c 1.5 -p %d -o ./sim/%s.maf" % (
            #     path_2020plus, path_2020plus, maf_train, p, args.type)
            # print(cmd)
            # check_output(cmd, shell=True)
            cmd = "probabilistic2020 oncogene -i %s/data/snvboxGenes.fa -b %s/data/snvboxGenes.bed -s %s/data/scores -m ./sim/%s.maf -c 1.5 -p %d -o ./sim/%s_oncogene.txt" % (
                path_2020plus, path_2020plus, path_2020plus, args.type, p, args.type)
            print(cmd)
            check_output(cmd, shell=True)
            # cmd = "probabilistic2020 tsg -i %s/data/snvboxGenes.fa -b %s/data/snvboxGenes.bed  -m ./sim/%s.maf -c 1.5 -p %d -o ./sim/%s_tsg.txt" % (
            #     path_2020plus, path_2020plus, args.type, p, args.type)
            # print(cmd)
            # check_output(cmd, shell=True)

            # summary
            cmd = "mut_annotate --summary -i %s/data/snvboxGenes.fa -b %s/data/snvboxGenes.bed -m ./sim/%s.maf -c 1.5 -p %d -o ./sim/%s_summary.txt" % (
                path_2020plus, path_2020plus, args.type, p, args.type)
            print(cmd)
            check_output(cmd, shell=True)
            cmd = "mut_annotate --summary -i %s/data/snvboxGenes.fa -b %s/data/snvboxGenes.bed -m %s -c 1.5 -p %d -o ./train/%s_summary.txt" % (
                path_2020plus, path_2020plus, maf_train, p, args.type)
            print(cmd)
            check_output(cmd, shell=True)
            cmd = "mut_annotate --summary -i %s/data/snvboxGenes.fa -b %s/data/snvboxGenes.bed -m %s -c 1.5 -p %d -o ./test/%s_summary.txt" % (
                path_2020plus, path_2020plus, maf_test, p, args.type)
            print(cmd)
            check_output(cmd, shell=True)

            # features
            cmd = "%s/2020plus.py features -og-test ./train/%s_oncogene.txt -tsg-test ./train/%s_tsg.txt --summary ./train/%s_summary.txt -o ./train/%s.txt" % (
                path_2020plus, args.type, args.type, args.type, args.type)
            print(cmd)
            check_output(cmd, shell=True)
            cmd = "%s/2020plus.py features -og-test ./test/%s_oncogene.txt -tsg-test ./test/%s_tsg.txt --summary ./test/%s_summary.txt -o ./test/%s.txt" % (
                path_2020plus, args.type, args.type, args.type, args.type)
            print(cmd)
            check_output(cmd, shell=True)
            # cmd = "%s/2020plus.py features -og-test ./sim/%s_oncogene.txt -tsg-test ./sim/%s_tsg.txt --summary ./sim/%s_summary.txt -o ./sim/%s.txt" % (
            #     path_2020plus, args.type, args.type, args.type, args.type)
            # print(cmd)
            # check_output(cmd, shell=True)

    elif args.mode == 'train':
        # a positive and negative sample ratio of 1:10
        X_train, y_train, _, _ = feature_input(args.type, nb_imb, fea_type)
        print(X_train.shape)
        # fit(X_train, y_train, args.type, fea_type)
        fit(X_train, y_train, args.type, fea_type)


    # Scoring on the test set
    elif args.mode == 'score':
        # get all genes with mutations
        genes = set(pd.read_csv('../input/gene.list', header=0, index_col=0, sep='\t').index.tolist())

        # X: raw test data(contains all genes)
        # ids: gene name of raw test data
        X, ids, _, _ = file2test(args.type, fea_type)
        y_p = predict(X, args.type, fea_type)

        # predict: raw test data
        df_score = pd.DataFrame(np.nan, index=ids, columns=['score'])
        df_score.loc[ids, 'score'] = y_p
        # predict: genes with mutation in test data
        df_all = pd.DataFrame(np.nan, index=genes, columns=['score'])
        g_and = set(genes) & set(ids)
        df_all.loc[g_and, 'score'] = df_score.loc[g_and, 'score']
        # predict: genes with mutation not in test data marked 0
        df_all['score'].fillna(0, inplace=True)

        if (fea_type == 'origin'):
            out_path = '../score/%s.score' % (args.type)
        elif ('norm_exp' in fea_type):
            out_path = '../score/norm_exp/%s/%s.score' % (fea_type[9:], args.type)
        else:
            out_path = '../score/%s/%s.score' % (fea_type, args.type)
        df_all = df_all.sort_values(by=['score'], ascending=[False])
        df_all.to_csv(out_path, header=True)

    # Result analysis
    elif args.mode == 'eval':
        if (fea_type == 'origin'):
            input = "../feature/%s_test.csv" % (args.type)
            res_file = '../score/%s.score' % (args.type)
            pancan_gene_path = '../PANCAN_test_performance.csv'
            path_cutoff = '../cutoff/cutoff_%s.csv' % (args.type)
        elif ('norm_exp' in fea_type):
            input = "../feature/norm_exp/%s/%s_test.csv" % (fea_type[9:], args.type)
            res_file = '../score/norm_exp/%s/%s.score' % (fea_type[9:], args.type)
            pancan_gene_path = '../PANCAN_test_performance_%s.csv' % fea_type
            path_cutoff = '../cutoff/norm_exp/%s/cutoff_%s.csv' % (fea_type[9:], args.type)
        else:
            input = "../feature/%s/%s_test.csv" % (fea_type, args.type)
            res_file = '../score/%s/%s.score' % (fea_type, args.type)
            pancan_gene_path = '../PANCAN_test_performance_%s.csv' % fea_type
            path_cutoff = '../cutoff/%s/cutoff_%s.csv' % (fea_type, args.type)

        df_tmp = pd.read_csv(input, header=0, index_col=0, sep=',')
        row_list = df_tmp.index.tolist()
        pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
        # cgc
        cgc_key = pd_cgc['Gene'].values.tolist()

        # prediction of test data
        df_score = pd.read_csv(res_file, index_col=0, header=0, sep=',')
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
            df_PANCAN_gene = pd.DataFrame(gene_cgc, columns=['gene'])
            df_PANCAN_gene.set_index('gene', inplace=True)
            df_PANCAN_gene['score'] = y_p_cgc
            df_PANCAN_gene['class'] = y_cgc
            # use in drawing picture later
            df_PANCAN_gene.to_csv(pancan_gene_path)

        auprc = '%.3f' % (average_precision_score(y_cgc, y_p_cgc))
        fpr, tpr, thresholds = roc_curve(y_cgc, y_p_cgc)
        auroc = '%.3f' % auc(fpr, tpr)
        print('performance(pr,auc):', auprc, auroc)
        p_value = (mannwhitneyu_(y_p_cgc, y_cgc))
        print('p_value', p_value)

        # fisher

        df = pd.read_csv(path_cutoff, sep=',', index_col=0, header=0)
        cutoff = df['cutoff'].values.tolist()[0]

        gene_sig = df_score.loc[df_score['score'] > cutoff].index.tolist()
        gene_sig = set(gene_sig)
        gene_nsig = df_score.loc[df_score['score'] <= cutoff].index.tolist()
        gene_nsig = set(gene_nsig)

        a = len(gene_sig)
        print('the number of candidate driver genes: %d,' % a)
        b = fisher(gene_sig, gene_nsig, cgc_key)
        print('fisher:', '%0.1f' % b)

    # interpretation
    elif args.mode == 'interpretation':

        interpretion_shap(args.type,nb_imb)

    # if you want to get the data in Table 1 or fig.5/6,  you can use the codes below
    elif args.mode == 'train_part':
        X_train, y_train, featureName, _ = feature_input_part(args.type, nb_imb, args.featureNums)
        fit_part(X_train, y_train, args.type, fea_nums=args.featureNums)

    elif args.mode == 'score_part':
        genes = set(pd.read_csv('../input/gene.list', header=0, index_col=0, sep='\t').index.tolist())
        X, ids, _, _ = file2test_part(args.type, fea_nums=args.featureNums)
        y_p = predict_part(X, args.type, fea_nums=args.featureNums)
        df_score = pd.DataFrame(np.nan, index=ids, columns=['score'])
        df_score.loc[ids, 'score'] = y_p
        df_all = pd.DataFrame(np.nan, index=genes, columns=['score'])
        g_and = set(genes) & set(ids)
        df_all.loc[g_and, 'score'] = df_score.loc[g_and, 'score']
        df_all['score'].fillna(0, inplace=True)
        out_path = '../score/%s_%s.score' % (args.type, args.featureNums)
        if args.featureInput == 'less':
            out_path = '../score/%s_less.score' % (args.type)
        df_all = df_all.sort_values(by=['score'], ascending=[False])
        df_all.to_csv(out_path, header=True)

    elif args.mode == 'eval_part':
        input = "../feature/%s_test.csv" % (args.type)
        df_tmp = pd.read_csv(input, header=0, index_col=0, sep=',')
        all_list1 = df_tmp.index.tolist()
        pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
        cgc_key = pd_cgc['Gene'].values.tolist()  # cgc
        res_file = '../score/%s_%s.score' % (args.type, args.featureNums)
        df_score = pd.read_csv(res_file, index_col=0, header=0, sep=',')
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
            df_PANCAN_gene = pd.DataFrame(gene_cgc, columns=['gene'])
            df_PANCAN_gene.set_index('gene', inplace=True)
            df_PANCAN_gene['score'] = y_p_cgc
            df_PANCAN_gene['class'] = y_cgc
            df_PANCAN_gene.to_csv('../analysis/PANCAN_test_performance_%s.csv' % args.featureNums)

        auprc = '%.3f' % (average_precision_score(y_cgc, y_p_cgc))
        fpr, tpr, thresholds = roc_curve(y_cgc, y_p_cgc)
        auroc = '%.3f' % auc(fpr, tpr)
        print('performance', auprc, auroc)
        path_cutoff = '../cuttoff/cutoff_%s_%s.csv' % (args.type, args.featureNums)
        df = pd.read_csv(path_cutoff, sep=',', index_col=0, header=0)
        cutoff = df['cutoff'].values.tolist()[0]
        gene_sig = df_score.loc[df_score['score'] > cutoff].index.tolist()
        gene_sig = set(gene_sig)
        gene_nsig = df_score.loc[df_score['score'] <= cutoff].index.tolist()
        gene_nsig = set(gene_nsig)
        a = len(gene_sig)
        print('the number of candidate driver genes: %d,' % a)
        b = fisher(gene_sig, gene_nsig, cgc_key)
        print('fisher:', '%0.1f' % b)


if __name__ == "__main__":
    main()
