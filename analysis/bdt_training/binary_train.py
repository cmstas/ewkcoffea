import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import os
import shutil

from topcoffea.scripts.make_html import make_html

def main():

    ### Get input data ###

    out_dir_name = "outdir_train"
    if not os.path.exists(out_dir_name):
        os.mkdir(out_dir_name)

    #dd = pickle.load(gzip.open("bdt.pkl.gz"))
    #dd = pickle.load(gzip.open("bdt_v0.pkl.gz"))
    #dd = pickle.load(gzip.open("bdt_zzz_siphon02.pkl.gz"))
    #dd = pickle.load(gzip.open("bdt_zzz_siphon02_18vars.pkl.gz"))
    #dd = pickle.load(gzip.open("bdt_siphon04_18vars_j00.pkl.gz"))
    dd = pickle.load(gzip.open("bdt_zzz_siphon05_18vars.pkl.gz"))

    X_train_of = dd["X_train_of"]
    y_train_of = dd["y_train_of"]
    w_train_of = dd["w_train_of"]
    X_test_of = dd["X_test_of"]
    y_test_of = dd["y_test_of"]
    w_test_of = dd["w_test_of"]

    print("X_train_of",X_train_of,len(X_train_of))
    print("y_train_of",y_train_of,len(y_train_of))
    print("w_train_of",w_train_of,len(w_train_of))
    print("X_test_of",X_train_of,len(X_test_of))
    print("y_test_of",y_train_of,len(y_test_of))
    print("w_test_of",w_train_of,len(w_test_of))


    ### Define model and fit ###

    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        missing=1,
        booster="gbtree",
        grow_policy="depthwise",
        #learning_rate=2.0,
        learning_rate=0.5,
        n_estimators=5,
        #n_estimators=1000,
        eval_metric=['error','logloss'],
        device="cuda",
        seed=42,
        #n_jobs=32
        n_jobs=64
    )
    params = xgb_clf.get_params()
    print("XGBoost Classifier Parameters:\n")
    for param, value in params.items():
        print(f"  {param}: {value}")

    # Train
    xgb_clf.fit(
        X_train_of,
        y_train_of,
        sample_weight=np.maximum(w_train_of, 0),
        verbose=1, # set to 1 to see xgb training round intermediate results
        eval_set=[(X_train_of, y_train_of), (X_test_of, y_test_of)]
    )

    xgb_clf.save_model(f"{out_dir_name}/bdt.json")


    ############# Make resutls plots #############

    shutil.copyfile("/home/users/phchang/public_html/dump/forKelci/index.php.txt", os.path.join(out_dir_name,"index.php"))

    ###  Make hists ###

    p_test_0 = xgb_clf.predict_proba(X_test_of)[:,0]
    p_test_1 = xgb_clf.predict_proba(X_test_of)[:,1]
    p_train_0 = xgb_clf.predict_proba(X_train_of)[:,0]
    p_train_1 = xgb_clf.predict_proba(X_train_of)[:,1]

    is_sig_test = label_binarize(y_test_of, classes=np.unique(y_test_of))[:,0] == 0
    is_bkg_test = label_binarize(y_test_of, classes=np.unique(y_test_of))[:,0] == 1
    is_sig_train = label_binarize(y_train_of, classes=np.unique(y_train_of))[:,0] == 0
    is_bkg_train = label_binarize(y_train_of, classes=np.unique(y_train_of))[:,0] == 1

    print("p0",p_test_0)
    print("p1",p_test_1)
    print("is_sig_test",is_sig_test)
    print("is_bkg_test",is_bkg_test)
    print("s",p_test_0[is_sig_test],max(p_test_0[is_sig_test]))
    print("b",p_test_0[is_bkg_test],max(p_test_0[is_bkg_test]))

    fig, ax = plt.subplots(figsize=(9,5))
    plt.hist(p_train_1[is_sig_train],bins=100,histtype="step",label="train is_sig",density=True)
    plt.hist(p_test_1[is_sig_test],bins=100,histtype="step",label="test is_sig",density=True)
    plt.hist(p_train_0[is_bkg_train],bins=100,histtype="step",label="train is_bkg",density=True)
    plt.hist(p_test_0[is_bkg_test],bins=100,histtype="step",label="test is_bkg",density=True)

    plt.legend()
    ax.set_xlim(0,1)
    plt.xlabel('prob')
    plt.title('Hist of prob')
    plt.savefig(f"{out_dir_name}/hist.png")
    plt.savefig(f"{out_dir_name}/hist.pdf")
    plt.clf()


    ### Make metric plots ###

    # preparing evaluation metric plots
    results = xgb_clf.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    # xgboost 'logloss' plot
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('logloss')
    plt.title('GridSearchCV XGBoost logloss')
    plt.savefig(f"{out_dir_name}/logloss.png")
    plt.savefig(f"{out_dir_name}/logloss.pdf")
    plt.clf()

    # xgboost 'error' plot
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('error')
    plt.title('GridSearchCV XGBoost error')
    plt.savefig(f"{out_dir_name}/error.png")
    plt.savefig(f"{out_dir_name}/error.pdf")
    plt.clf()

    ### Make ROC plots ###

    y_prob_of = xgb_clf.predict_proba(X_test_of)
    n_classes = len(np.unique(y_test_of))
    fpr = {}
    tpr = {}
    roc_auc = {}
    y_test_of_bin = label_binarize(y_test_of, classes=np.unique(y_test_of))

    fpr, tpr, _ = roc_curve(label_binarize(y_train_of,classes=np.unique(y_train_of)), p_train_1, sample_weight=np.abs(w_train_of))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve train (AUC = {roc_auc})')
    fpr, tpr, _ = roc_curve(label_binarize(y_test_of,classes=np.unique(y_test_of)), p_test_1, sample_weight=np.abs(w_test_of))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve test (AUC = {roc_auc})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{out_dir_name}/roc.png")
    plt.savefig(f"{out_dir_name}/roc.pdf")


main()
