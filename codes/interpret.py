import pickle
from shap import KernelExplainer
from codes.buildData import *
from datetime import datetime
np.random.seed(1)


def test_genes(type,nb_imb):
    _, _, df_tmp, feature_name = file2test(type)
    pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
    raw_list = df_tmp.index.tolist()
    cgc_key = pd_cgc['Gene'].values.tolist()  # cgc
    res_file = '../score/%s.score' % (type)
    df_score = pd.read_csv(res_file, index_col=0, header=0, sep=',')
    score_gene = df_score.index.tolist()
    all_list = []
    for i in raw_list:
        if i in score_gene:
            all_list.append(i)
    pos_cgc, neg_cgc = build_set(cgc_key, [], all_list, nb_imb)
    y_test_all = np.concatenate([np.ones((len(pos_cgc))), np.zeros((len(neg_cgc)))])
    test_gene_name = pos_cgc + neg_cgc
    test = df_tmp.loc[test_gene_name, ::]
    x_test_all = []
    x_test_all.append(np.concatenate([test]))
    return test_gene_name, x_test_all, y_test_all, feature_name


def interpretion_shap(type, nb_imb):
    print('start', datetime.now().isoformat())
    test_gene_name, x_test_all, y_test_all, feature_name = test_genes(type,nb_imb)
    X_all_test = np.concatenate(x_test_all, axis=1)

    scaler_path = '../model/%s.scaler' % (type)
    scaler = pickle.load(open(scaler_path, 'rb'))
    x_test_all = scaler.transform(X_all_test)

    x_train, y_train, _, _ = feature_input(type, nb_imb )
    X_train = scaler.transform(x_train)

    model_path = '../model/%s.model' % (type)
    model = pickle.load(open(model_path, 'rb'))
    f = lambda x: model.predict_proba(x)[:, 1]
    print('start inter',datetime.now().isoformat())
    x_test_all = x_test_all[:5, :]
    explainer = KernelExplainer(f, X_train)

    print('every gene', datetime.now().isoformat())
    shap_values = explainer.shap_values(x_test_all)
    print('end inter',datetime.now().isoformat())

    # save the interpretation
    np.save('../interpretation/data/%s_shap_value' % (type), shap_values)
    expected_value = explainer.expected_value
    np.save('../interpretation/data/%s_ex_value' % (type), expected_value)
    csvFile = pd.DataFrame(shap_values, index=test_gene_name, columns=feature_name)
    csvFile.to_csv('../interpretation/data/%s_shapValues.csv' % (type))
