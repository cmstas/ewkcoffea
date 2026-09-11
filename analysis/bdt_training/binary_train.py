import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip
from sklearn.metrics import roc_curve, auc
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

    #dd = pickle.load(gzip.open("bdt.pkl.gz"))
    #dd = pickle.load(gzip.open("bdt_zzz_siphon05_18vars_p_ordered_b0s1_2.pkl.gz"))
    dd = pickle.load(gzip.open("bdt_zzz_siphon06_18vars_p_ordered.pkl.gz"))

    X_train_of = dd["X_train_of"]
    y_train_of = dd["y_train_of"]
    w_train_of = dd["w_train_of"]
    X_test_of = dd["X_test_of"]
    y_test_of = dd["y_test_of"]
    w_test_of = dd["w_test_of"]

    var_name_lst = dd["var_name_lst"]

    # Print out shape info and write to file
    print("Shapes:")
    print(f"\tX_train_of: {X_train_of.shape}")
    print(f"\ty_train_of: {y_train_of.shape}")
    print(f"\tw_train_of: {w_train_of.shape}")
    print(f"\tX_test_of: {X_test_of.shape}")
    print(f"\ty_test_of: {y_test_of.shape}")
    print(f"\tw_test_of: {w_test_of.shape}")
    print(f"\tvar_name_lst: {var_name_lst}")
    fout = open(f"{out_dir_name}/info.txt", "w")
    fout.write(f"\nShapes:\n")
    fout.write(f"\tX_train_of: {X_train_of.shape}\n")
    fout.write(f"\ty_train_of: {y_train_of.shape}\n")
    fout.write(f"\tw_train_of: {w_train_of.shape}\n")
    fout.write(f"\tX_test_of: {X_test_of.shape}\n")
    fout.write(f"\ty_test_of: {y_test_of.shape}\n")
    fout.write(f"\tw_test_of: {w_test_of.shape}\n")
    fout.write(f"var_name_lst: {var_name_lst}\n")
    fout.close()

    shutil.copyfile("/home/users/phchang/public_html/dump/forKelci/index.php.txt", os.path.join(out_dir_name,"index.php"))

    ############# Plot input vars #############

    make_input_var_plots = 0
    if make_input_var_plots:
        nvars = len(var_name_lst)
        sig_key = 1
        bkg_key = 0

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

    ################################################
    ############# Define model and fit #############

    xgb_clf = xgb.XGBClassifier(
        #objective='multi:softprob',
        #objective='multi:softmax',
        #num_class=2,
        #objective='reg:squarederror',
        #objective='reg:logistic',

        objective='binary:logistic',
        n_estimators=500,
        eval_metric=['error','logloss'],
        #eval_metric=['merror','mlogloss'],
        seed=42,
        n_jobs=64,
        scale_pos_weight=400,

        #device="cuda",
        #grow_policy="depthwise",
        #booster="gbtree",

        learning_rate=0.2,
        reg_lambda=1.5,
        reg_alpha=1.5,
        max_depth=4,
        min_child_weight=2,
        #min_split_loss=100, # Does not seem to help
        #max_delta_step=10, # Seems to not do anything

        subsample=0.5,
    )

    # Print the params and write out for ref
    fout = open(f"{out_dir_name}/info.txt", "a")
    params = xgb_clf.get_params()
    print("\nXGBoost Classifier Parameters:\n")
    fout.write("\nXGBoost Classifier Parameters:\n")
    for param, value in params.items():
        print(f"\t{param}: {value}")
        fout.write(f"  {param}: {value}\n")
    fout.close()


    #integral_train_of_sig = np.sum(w_train_of[y_train_of==1])
    #integral_train_of_bkg = np.sum(w_train_of[y_train_of==0])
    #integral_test_of_sig = np.sum(w_test_of[y_test_of==1])
    #integral_test_of_bkg = np.sum(w_test_of[y_test_of==0])
    #w_train_of[y_train_of==1] = w_train_of[y_train_of==1] / integral_train_of_sig
    #w_train_of[y_train_of==0] = w_train_of[y_train_of==0] / integral_train_of_bkg
    #w_test_of[y_test_of==1] = w_test_of[y_test_of==1] / integral_test_of_sig
    #w_test_of[y_test_of==0] = w_test_of[y_test_of==0] / integral_test_of_bkg

    # Train
    xgb_clf.fit(
        X_train_of,
        y_train_of,
        sample_weight=np.maximum(w_train_of, 0),
        verbose=1, # set to 1 to see xgb training round intermediate results
        eval_set=[(X_train_of, y_train_of), (X_test_of, y_test_of)]
    )

    # Save  the model
    xgb_clf.save_model(f"{out_dir_name}/bdt.json")



    ############# Make resutls plots #############


    ###  Make prob hist ###

    p_train = xgb_clf.predict_proba(X_train_of)[:,1]
    p_test  = xgb_clf.predict_proba(X_test_of)[:,1]

    p_train_sig = p_train[y_train_of==1]
    p_train_bkg = p_train[y_train_of==0]
    p_test_sig = p_test[y_test_of==1]
    p_test_bkg = p_test[y_test_of==0]

    w_train_sig = w_train_of[y_train_of==1]
    w_train_bkg = w_train_of[y_train_of==0]
    w_test_sig  = w_test_of[y_test_of==1]
    w_test_bkg  = w_test_of[y_test_of==0]

    # Print test and train sig and bkg info
    print(f"Test and train sig and bkg numbers:")
    print(f"Test sig len, avg -> {len(p_test_sig)}, {sum(p_test_sig)/len(p_test_sig)}")
    print(f"Test bkg: len, avg -> {len(p_test_bkg)}, {sum(p_test_bkg)/len(p_test_bkg)}")
    print(f"Train sig: len, avg -> {len(p_train_sig)}, {sum(p_train_sig)/len(p_train_sig)}")
    print(f"Train bkg: len, avg -> {len(p_train_bkg)} , {sum(p_train_bkg)/len(p_train_bkg)}")
    fout = open(f"{out_dir_name}/info.txt", "a")
    fout.write(f"\nTest and train sig and bkg numbers:\n")
    fout.write(f"\tTest sig len, avg -> {len(p_test_sig)}, {sum(p_test_sig)/len(p_test_sig)}\n")
    fout.write(f"\tTest bkg: len, avg -> {len(p_test_bkg)}, {sum(p_test_bkg)/len(p_test_bkg)}\n")
    fout.write(f"\tTrain sig: len, avg -> {len(p_train_sig)}, {sum(p_train_sig)/len(p_train_sig)}\n")
    fout.write(f"\tTrain bkg: len, avg -> {len(p_train_bkg)} , {sum(p_train_bkg)/len(p_train_bkg)}\n")
    fout.close()

    fig, ax = plt.subplots(figsize=(5,5))
    plt.hist(p_train_sig,weights=w_train_sig,bins=100,histtype="step",label="train is_sig",density=True)
    plt.hist(p_test_sig, weights=w_test_sig, bins=100,histtype="step",label="test is_sig",density=True)
    plt.hist(p_train_bkg,weights=w_train_bkg,bins=100,histtype="step",label="train is_bkg",density=True)
    plt.hist(p_test_bkg, weights=w_test_bkg, bins=100,histtype="step",label="test is_bkg",density=True)

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
    #epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(0, epochs)

    # xgboost 'logloss' plot
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    #ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    #ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
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
    #ax.plot(x_axis, results['validation_0']['merror'], label='Train')
    #ax.plot(x_axis, results['validation_1']['merror'], label='Test')
    ax.legend()
    plt.ylabel('error')
    plt.title('GridSearchCV XGBoost error')
    plt.savefig(f"{out_dir_name}/error.png")
    plt.savefig(f"{out_dir_name}/error.pdf")
    plt.clf()

    ### Make ROC plot ###

    # Roc for train
    fpr, tpr, _ = roc_curve(y_train_of, p_train, sample_weight=np.abs(w_train_of))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve train (AUC = {roc_auc})')

    # Roc for test
    fpr, tpr, _ = roc_curve(y_test_of, p_test, sample_weight=np.abs(w_test_of))
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
