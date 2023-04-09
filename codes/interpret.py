import pickle
from shap import KernelExplainer
from codes.buildData import *

np.random.seed(1)

def test_genes(type):
    _, _, df_tmp, feature_name = file2test(type)
    pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
    raw_list = df_tmp.index.tolist()
    cgc_key = pd_cgc['Gene'].values.tolist()  # cgc
    res_file = '../score/InDEP/%s.score' % (type)
    df_score = pd.read_csv(res_file, index_col=0, header=0, sep=',')
    score_gene = df_score.index.tolist()
    all_list = []
    for i in raw_list:
        if i in score_gene:
            all_list.append(i)
    pos_cgc, neg_cgc = build_set(cgc_key, [], all_list, nb_imb=10)
    y_test_all = np.concatenate([np.ones((len(pos_cgc))), np.zeros((len(neg_cgc)))])
    test_gene_name = pos_cgc + neg_cgc
    test = df_tmp.loc[test_gene_name,::]
    x_test_all = []
    x_test_all.append(np.concatenate([test]))
    return test_gene_name,x_test_all,y_test_all, feature_name


def interpretion_shap(type, method):
    test_gene_name,x_test_all,y_test_all, feature_name = test_genes(type)
    x_all = np.concatenate(x_test_all, axis=1)
    X_all_test = []
    X_all_test.append(x_all)
    model_path = '../model/%s/%s.model' % (method, type)
    X = []
    for j in range(len(X_all_test)):
        scaler_path = '../model/%s/%s_%d.scaler' % (method, type, j)
        scaler = pickle.load(open(scaler_path, 'rb'))
        X.append(scaler.transform(X_all_test[j]))
    model = pickle.load(open(model_path, 'rb'))
    x_test_all = np.concatenate(X, axis=1)
    x_train, y_train, _, _ = feature_input(type, 10)
    X = []
    for j in range(len(x_train)):
        scaler_path = '../model/%s/%s_%d.scaler' % (method, type, j)
        scaler = pickle.load(open(scaler_path, 'rb'))
        X.append(scaler.transform(x_train[j]))
    X_train = np.concatenate(X, axis=1)
    f = lambda x: model.predict_proba(x)[:,1]
    explainer = KernelExplainer(f, X_train)
    shap_values = explainer.shap_values(x_test_all)
    # save the interpretation
    np.save('../interpretation/%s/data/%s_shap_value'%(method,type),shap_values)
    expected_value = explainer.expected_value
    np.save('../interpretation/%s/data/%s_ex_value'%(method,type),expected_value)
    csvFile = pd.DataFrame(shap_values, index=test_gene_name, columns=feature_name)
    csvFile.to_csv('../interpretation/%s/data/%s_shapValues.csv'%(method,type))


