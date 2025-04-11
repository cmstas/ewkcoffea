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

# Take an array find a cap where 99.5% of vals are below
def get_cap_val(in_arr):

    cap_val = None
    min_val = min(in_arr)
    max_val = max(in_arr)
    nsteps = 10
    step_size = abs(max_val-min_val)/nsteps
    for iterator in range(nsteps):
        val = min_val + iterator*step_size
        tot = len(in_arr)
        sub_tot = len(in_arr[in_arr<val])
        if sub_tot/tot > 0.999:
            cap_val = val
            break

    if cap_val is None: cap_val = max(in_arr)

    return cap_val


def main():

    ############# Get input data #############

    out_dir_name = "outdir_train"
    if not os.path.exists(out_dir_name):
        os.mkdir(out_dir_name)

    #dd = pickle.load(gzip.open("bdt_zzz_siphon05_16vars_k_ordered.pkl.gz"))
    dd = pickle.load(gzip.open("bdt_zzz_siphon05_18vars_p_ordered.pkl.gz"))

    X_train_of = dd["X_train_of"]
    y_train_of = dd["y_train_of"]
    w_train_of = dd["w_train_of"]
    X_test_of = dd["X_test_of"]
    y_test_of = dd["y_test_of"]
    w_test_of = dd["w_test_of"]

    var_name_lst = dd["var_name_lst"]

    print("X_train_of",X_train_of.shape)
    print("y_train_of",y_train_of.shape)
    print("w_train_of",w_train_of.shape)
    print("X_test_of" ,X_test_of.shape)
    print("y_test_of" ,y_test_of.shape)
    print("w_test_of" ,w_test_of.shape)


    ############# Plot input vars #############

    shutil.copyfile("/home/users/phchang/public_html/dump/forKelci/index.php.txt", os.path.join(out_dir_name,"index.php"))

    nvars = len(var_name_lst)
    sig_key = 0
    bkg_key = 1

    print("var_name_lst",var_name_lst)
    for i,var_name in enumerate(var_name_lst):
        print(i,var_name)

        ### Get the capped arrays ###
        # Train
        var_train_i  = X_train_of[:,i]
        cap_val_train = get_cap_val(var_train_i)
        # Test
        var_test_i  = X_test_of[:,i]
        cap_val_test = get_cap_val(var_test_i)
        # Cap
        cap_val = max(cap_val_test,cap_val_train)
        var_train_i = np.where(var_train_i<cap_val,var_train_i,cap_val)
        var_test_i  = np.where(var_test_i<cap_val,var_test_i,cap_val)

        # Train
        wgt_train    = w_train_of
        var_train_i_sig = var_train_i[y_train_of == sig_key]
        var_train_i_bkg = var_train_i[y_train_of == bkg_key]
        wgt_train_sig   = w_train_of[y_train_of == sig_key]
        wgt_train_bkg   = w_train_of[y_train_of == bkg_key]

        # Test
        wgt_test    = w_test_of
        var_test_i_sig = var_test_i[y_test_of == sig_key]
        var_test_i_bkg = var_test_i[y_test_of == bkg_key]
        wgt_test_sig   = w_test_of[y_test_of == sig_key]
        wgt_test_bkg   = w_test_of[y_test_of == bkg_key]

        # Plot the hist
        fig, ax = plt.subplots(figsize=(5,5))
        hrange = (min(min(var_test_i),min(var_train_i)),cap_val)
        plt.hist(var_train_i_sig,weights=wgt_train_sig,bins=60,range=hrange,histtype="step",label="train sig",density=True)
        plt.hist(var_test_i_sig, weights=wgt_test_sig, bins=60,range=hrange,histtype="step",label="test sig",density=True)
        plt.hist(var_train_i_bkg,weights=wgt_train_bkg,bins=60,range=hrange,histtype="step",label="train bkg",density=True)
        plt.hist(var_test_i_bkg, weights=wgt_test_bkg, bins=60,range=hrange,histtype="step",label="test bkg",density=True)
        plt.legend()
        plt.xlabel(f'{var_name}')
        plt.title(f'Variable {i} {var_name}, density=T')
        plt.savefig(f"{out_dir_name}/var_{i}_{var_name}.png")
        plt.savefig(f"{out_dir_name}/var_{i}_{var_name}.pdf")
        plt.clf()

    #exit()

    ############# Define model and fit #############

    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        missing=1,
        booster="gbtree",
        grow_policy="depthwise",
        #learning_rate=2.0,
        learning_rate=0.5,
        #n_estimators=100,
        n_estimators=800,
        eval_metric=['error','logloss'],
        device="cuda",
        seed=42,
        n_jobs=32
        #n_jobs=64
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


    ###  Make prob hist ###

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

    fig, ax = plt.subplots(figsize=(5,5))
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
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('logloss')
    plt.title('GridSearchCV XGBoost logloss')
    plt.savefig(f"{out_dir_name}/logloss.png")
    plt.savefig(f"{out_dir_name}/logloss.pdf")
    ax.set_yscale('log')
    plt.savefig(f"{out_dir_name}/logloss_log.png")
    plt.savefig(f"{out_dir_name}/logloss_log.pdf")
    plt.clf()

    # xgboost 'error' plot
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('error')
    plt.title('GridSearchCV XGBoost error')
    plt.savefig(f"{out_dir_name}/error.png")
    plt.savefig(f"{out_dir_name}/error.pdf")
    plt.clf()

    ### Make ROC plot ###

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
