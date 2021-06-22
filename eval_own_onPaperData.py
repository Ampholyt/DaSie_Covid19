import lightgbm as lgb
import numpy as np
# data: https://github.com/nshomron/covidpred/tree/master/data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import metrics
import shap  # package used to calculate shap values
from bayes_opt import BayesianOptimization

#building models
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

#feature importance
import seaborn as sns

'''
Evaluate the own model on the Data from the paper
'''

fileprefix = 'data/preprocessed_full_'

# load train and validation data
train_pd = pd.read_csv(fileprefix + 'train.csv', sep=',')
valid_pd = pd.read_csv(fileprefix + 'val.csv', sep=',')
test_pd = pd.read_csv(fileprefix + 'test.csv', sep=',')

train_X = train_pd.drop(labels="corona_result", axis=1)
valid_X = valid_pd.drop(labels="corona_result", axis=1)
test_X = test_pd.drop(labels="corona_result", axis=1)

# possible print
# print(train_X.dtypes)
# print(test_X.dtypes)
# print(valid_X.dtypes)

#hyper param training -> optimal parameter
def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=3, random_seed=6,n_estimators=10000, output_process=False):
    # prepare data

    def lgb_eval(learning_rate,num_leaves, feature_fraction, bagging_fraction, max_depth, max_bin, min_data_in_leaf,min_sum_hessian_in_leaf,subsample):
        train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)
        # parameters
        params = {'application':'binary', 'metric':'auc'}
        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['max_bin'] = int(round(max_depth))
        params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
        params['subsample'] = max(min(subsample, 1), 0)
        params['verbose'] = -1

        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
        return max(cv_result['auc-mean'])

    lgbBO = BayesianOptimization(lgb_eval, {'learning_rate': (0.01, 1.0),
                                            'num_leaves': (24, 80),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 30),
                                            'max_bin':(20,90),
                                            'min_data_in_leaf': (20, 80),
                                            'min_sum_hessian_in_leaf':(0,100),
                                           'subsample': (0.01, 1.0)},
                                           random_state=200)


    #n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
    #init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.

    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    model_auc=[]
    for model in range(len( lgbBO.res)):
        model_auc.append(lgbBO.res[model]['target'])

    # return best parameters
    return lgbBO.res[pd.Series(model_auc).idxmax()]['target'],lgbBO.res[pd.Series(model_auc).idxmax()]['params']

opt_params = bayes_parameter_opt_lgb(train_X, train_pd["corona_result"], init_round=5, opt_round=10, n_folds=3, random_seed=6,n_estimators=10000)
opt_params[1]["num_leaves"] = int(round(opt_params[1]["num_leaves"]))
opt_params[1]['max_depth'] = int(round(opt_params[1]['max_depth']))
opt_params[1]['min_data_in_leaf'] = int(round(opt_params[1]['min_data_in_leaf']))
opt_params[1]['max_bin'] = int(round(opt_params[1]['max_bin']))
opt_params[1]['objective']='binary'
opt_params[1]['metric']='auc'
opt_params[1]['is_unbalance']=True
opt_params[1]['boost_from_average']=False
opt_params[1]['verbose'] = -1
opt_params=opt_params[1]
print(opt_params)

target=train_pd['corona_result']
features= [c for c in train_pd.columns if c not in ['corona_result']]


folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

oof = np.zeros(len(train_pd))
predictions = np.zeros(len(train_pd))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_pd.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train_pd.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_pd.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 15000
    clf = lgb.train(opt_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 250)
    oof[val_idx] = clf.predict(train_pd.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    # I don't know what that means
    # predictions += clf.predict(test_pd[features], num_iteration=clf.best_iteration) / folds.n_splits


    # print("test_pd[features]: ",test_pd[features], "\nnum iterations: ", clf.best_iteration, "\nfolds.n_splits: ", folds.n_splits,
    # "\nclf.predict(test_pd[features], num_iteration=clf.best_iteration): ",clf.predict(test_pd[features], num_iteration=clf.best_iteration),
    # "\ntype of clf.predict(test_pd[features], num_iteration=clf.best_iteration)", type(clf.predict(test_pd[features], num_iteration=clf.best_iteration)))
    # predi = clf.predict(test_pd[features], num_iteration=clf.best_iteration)
    # divi = predi / folds.n_splits
    # predictions = np.add(predictions,divi)
print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:20].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(20,28))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()
plt.savefig('Feature_Importance.png')


# training:
# param_readin = {'feature_pre_filter': False} # why is this needed? (https://lightgbm.readthedocs.io/en/latest/Parameters.html search for feature_pre_filter)
# train_data = lgb.Dataset(train_X, label=train_pd['corona_result'])
# val_data = lgb.Dataset(valid_X, params=param_readin,  label=valid_pd['corona_result'], reference=train_data)
# # train model
# train_param = {'num_leaves': 20,
#                'objective': 'binary',
#                'min_data_in_leaf': 4,
#                'feature_fraction': 0.2,
#                'bagging_fraction': 0.8,
#                'bagging_freq': 5,
#                'learning_rate': 0.05,
#                'verbose': 1,
#     }
# #num_boost_round=603
# bst = lgb.train(train_param, train_data, num_boost_round=604, early_stopping_rounds=5, valid_sets=[val_data])
#
# #predict
# ypred = bst.predict(test_X)
# print(ypred)
# print(max(ypred))
# print(min(ypred))
#
#
# #validation/testset
# vY = test_pd["corona_result"]
# vX = test_pd.drop(labels="corona_result", axis=1)
# print(vX)
#
# #roc curve:
# fpr, tpr, thresholds = metrics.roc_curve(vY, ypred, pos_label=1)
#
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkblue',
#          lw=lw, label='ROC curve')
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
#
# #########AUC#############
# AUC = metrics.auc(fpr,tpr)
# print("The AUC is {}".format(AUC))
#
# ########auPRC############
# precision, recall, thresholds = metrics.precision_recall_curve(vY, ypred, pos_label=1)
# plt.figure()
# lw = 2
# plt.plot(recall, precision, color='darkblue',
#          lw=lw, label='Precision Recall Curve')
#
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall-Curve')
# plt.legend(loc="lower left")
# plt.show()
#
#
# #######SHAP##############
#
# shaping = input("Do you want to calculate the shap values?")
# if (shaping == 'yes' or shaping == 'y'):
#     # fix?
#     bst.params["objective"] = "binary"
#     # Create object that can calculate shap values
#     explainer = shap.TreeExplainer(bst)
#
#     # Calculate Shap values
#     shap_values = explainer.shap_values(vX)
#
#     # shap.initjs()
#     shap.summary_plot(shap_values, vX)

    # # How to use this?
    # shap.force_plot(explainer.expected_value[1], shap_values[1], vX)
    #
    # # use Kernel SHAP to explain test set predictions
    # k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
    # k_shap_values = k_explainer.shap_values(vX)
    # shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], vX)
