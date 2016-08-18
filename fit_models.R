################################################################################
#
# Code for Interpretable Models for Recidivism Prediction
# by Jiaming Zeng/Berk Ustun/Cynthia Rudin
#
# Contact: jiaming@alum.mit.edu / ustunb@mit.edu
#
################################################################################
#
# This script fits models for a given dataset and set class-weights using:
#
# - C50 Rules
# - C50 Decision Trees
# - CART Decision Trees
# - Penalized Logistic Regression (Lasso Penalty)
# - Penalized Logistic Regression (Ridge Penalty)
# - Random Forests
# - Support Vector Machines (RBF Kernel)
# - Stochastic Gradient Boosting
#
# To work correctly, set the working directory to the directory that contains
# this file (i.e., to the directory that contains data/ and results/)
#
################################################################################
data_name = "arrest" #or 'drug', 'general_violence', 'fatal_violence'
class_weights = c(1.00, 1.00); #c(w_neg, w_pos);
print_flag = TRUE;

#set directories
home_dir = paste0(getwd(),"/")
data_dir = paste0(home_dir, "data/");
results_dir = paste0(home_dir, "results/");
if (!dir.exists(results_dir)){dir.create(results_dir)}; #created if it doesn't exist

#important run variables
run_name = sprintf("%s_wneg_%1.9f_wpos_%1.9f", data_name, class_weights[1], class_weights[2]);
data_file_name = paste0(data_dir, data_name, ".RData");
results_file_name = paste0(results_dir, run_name, "_results.RData");

#sanity checks
stopifnot(file.exists(data_file_name));
stopifnot(all(class_weights>0.0));

#load data and check data integrity
load(data_file_name);
X = as.matrix(X);
Y = as.matrix(Y);
X_test = as.matrix(X_test);
Y_test = as.matrix(Y_test);
stopifnot(nrow(X_test) == nrow(Y_test));
stopifnot(ncol(X_test) == ncol(X));
stopifnot(ncol(Y_test) == 1);
stopifnot(ncol(Y) == 1);

#load libraries
library("methods", warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE);
library("C50", warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE);
library("rpart", warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE);
library("e1071",warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE);
library("randomForest",warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE);
library("glmnet",warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE);
library("gbm", warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE);

#pick which methods to use
train_settings = list();
train_settings$run_plr_lasso = TRUE; #LR + Lasso Penalty
train_settings$run_plr_ridge = TRUE; #LR + Ridge Penalty
train_settings$run_C50_rule = TRUE; #C5.0 Rule Lists
train_settings$run_C50_tree = TRUE; #C5.0 Decision Trees
train_settings$run_cart = TRUE; #CART Decision Trees
train_settings$run_randomforest = TRUE; #Random Forests
train_settings$run_svm_rbf = TRUE; #SVM (RBF Kernel)
train_settings$run_sgb = TRUE; #SGB

#set to TRUE to store model objects (default: FALSE as this requires lots of storage space)
train_settings$store_model_objects = FALSE;

#pick free parameters for each method
train_settings$cart_minSplit = c(3, 5, 10, 15, 20);
train_settings$cart_cp = c(0.0001, 0.001, 0.01);
train_settings$cart_prune_to_L0_max = FALSE;
train_settings$cart_prune_to_min = FALSE;
train_settings$plr_nlambda = 100;
train_settings$randomforest_sampsize = c(0.632, 0.4, 0.2) * nrow(X);
train_settings$randomforest_nodesize = c(1, 5, 10, 20);
train_settings$svm_costs = c(0.01, 0.1, 1, 10);
train_settings$svm_gamma = c(1/10, 1/5, 1/2, 1.0, 2.0, 5.0, 10.0) *  1/ncol(X);
train_settings$sgb_shrinkage = c(0.001,0.01,0.1);
train_settings$sgb_ntrees = c(100, 500, 1500, 3000);
train_settings$sgb_depth = c(1, 2, 3, 4);
train_settings$randomforest_sampsize = c(0.632) * nrow(X);
train_settings$randomforest_nodesize = c(5);
train_settings$svm_costs = c(0.1);
train_settings$svm_gamma = c(1/10) *  1/ncol(X);
train_settings$sgb_shrinkage = c(0.1);
train_settings$sgb_ntrees = c(100);
train_settings$sgb_depth = c(2);

#### Script Helper Functions ####
print.to.console = function(print_string, flag = print_flag){
    if(flag){
        cat(sprintf('%s | %s\n',format(Sys.time(),"%X"), print_string))
    }
}

run.method = function(method_name) {
    run_flag_name = sprintf("run_%s", method_name);
    if (run_flag_name %in% names(train_settings)){
        run_flag = train_settings[[run_flag_name]];
    } else {
        run_flag = FALSE;
    }
    return(run_flag);
}

discard.model.objects = function(results){

    if (!train_settings$store_model_objects){
        results$models = NULL;
        results$cv_model = NULL;
        results$full_model = NULL;
    }

    return(results)

}

## C50
train.C50 = function(X, Y, X_test, Y_test, folds, train_rules = FALSE){


    print_message = sprintf("Running %s on %s with weights w- = %1.3f w+ = %1.3f",
                            method_name,
                            data_name,
                            2*class_weights[1],
                            2*class_weights[2]);

    print.to.console(print_message);
    #Initialize Cost Matrix
    if (!weighted_flag){
        cost_matrix = matrix(c(0,1.0,1.0,0), byrow=TRUE, nrow=2, dimnames=list(c("0","1"),c("0","1")));
    } else {
        cost_matrix = matrix(c(0,2*class_weights[1],2*class_weights[2],0), byrow=TRUE, nrow=2, dimnames=list(c("0","1"),c("0","1")));
    }

    #Initialize Function Output
    results = list();

    results$method_name = method_name;
    results$run_date = Sys.time();
    results$class_weights = class_weights;
    results$cost_matrix = cost_matrix;
    results$models = vector("list", K);

    results$train_errors = array(NA, K);
    results$train_pos_errors = array(NA, K);
    results$train_neg_errors = array(NA, K);
    results$train_weighted_errors = array(NA, K);

    results$valid_errors = array(NA, K);
    results$valid_pos_errors = array(NA, K);
    results$valid_neg_errors = array(NA, K);
    results$valid_weighted_errors = array(NA, K);

    results$full_train_errors = array(NA, 1);
    results$full_train_pos_errors = array(NA, 1);
    results$full_train_neg_errors = array(NA, 1);
    results$full_train_weighted_errors = array(NA, 1);

    results$model_sizes = array(NA, K);
    results$full_model_sizes = array(NA, 1);

    # Validation Errors
    results$test_errors = array(NA, K);
    results$pos_test_errors = array(NA, K);
    results$neg_test_errors = array(NA, K);
    results$weighted_test_errors = array(NA, K);

    results$full_test_errors = array(NA, 1);
    results$full_test_pos_errors = array(NA, 1);
    results$full_test_neg_errors = array(NA, 1);
    results$full_test_weighted_errors = array(NA, 1);

    #For Each Fold
    for (k in 1:K){

        #training set
        X_train = as.matrix(X[k!=folds,]);
        Y_train = as.factor(Y[k!=folds]);
        N_train = nrow(X_train);
        pos_train_ind= Y_train==1;
        neg_train_ind= !pos_train_ind;
        N_train_pos = sum(pos_train_ind);
        N_train_neg = sum(neg_train_ind);

        #Specify Validation Set
        X_valid = as.matrix(X[k==folds,]);
        Y_valid = as.factor(Y[k==folds]);
        N_valid = nrow(X_valid)
        pos_valid_ind = Y_valid==1;
        neg_valid_ind = !pos_valid_ind;
        N_valid_pos = sum(pos_valid_ind);
        N_valid_neg = sum(neg_valid_ind);

        #Set up the Model
        fold_model = C5.0(x=X_train, y=Y_train, rules=train_rules, costs=t(cost_matrix));

        #Process + Store the Training Error
        y_hat = predict(fold_model,newdata=X_train);
        results$train_errors[k] = sum(Y_train!=y_hat)/N_train;
        results$train_pos_errors[k] = sum(Y_train[pos_train_ind]!=y_hat[pos_train_ind])/N_train_pos;
        results$train_neg_errors[k] = sum(Y_train[neg_train_ind]!=y_hat[neg_train_ind])/N_train_neg;
        results$train_weighted_errors[k] = (2*class_weights[1]*results$train_neg_errors[k]*N_train_neg + 2*class_weights[2]*results$train_pos_errors[k]*N_train_pos)/N_train;

        # validation error metrics
        y_hat = predict(fold_model,newdata=X_valid);
        results$valid_errors[k] = sum(Y_valid!= y_hat)/N_valid;
        results$valid_pos_errors[k] = sum(Y_valid[pos_valid_ind] != y_hat[pos_valid_ind])/N_valid_pos;
        results$valid_neg_errors[k] = sum(Y_valid[neg_valid_ind] != y_hat[neg_valid_ind])/N_valid_neg;
        results$valid_weighted_errors[k] = (2*class_weights[1]*results$valid_neg_errors[k]*N_valid_neg + 2*class_weights[2]*results$valid_pos_errors[k]*N_valid_pos)/N_valid;

        # test error metrics
        y_hat = predict(fold_model, newdata=X_test)
        results$test_errors[k] = sum(y_hat!=Y_test)/N_test;
        results$pos_test_errors[k] = sum(y_hat[pos_test_ind] != Y_test[pos_test_ind])/N_test_pos;
        results$neg_test_errors[k] = sum(y_hat[neg_test_ind] != Y_test[neg_test_ind])/N_test_neg;
        results$weighted_test_errors[k] = (2*class_weights[1]*results$neg_test_errors[k]*N_test_neg + 2*class_weights[2]*results$pos_test_errors[k]*N_test_pos)/N_test;

        #Model Size
        results$model_sizes[k] = fold_model$size

        # store model object
        results$models[[k]] = fold_model;
    }

    # fit final (full) model using all data
    results$full_model = C5.0(x=X, y=as.factor(Y), rules = train_rules, costs=t(cost_matrix))
    y_hat = predict(results$full_model, newdata = X);
    results$full_train_errors[1] = sum(Y!=y_hat)/N;
    results$full_train_pos_errors[1] = sum(Y[pos_ind]!=y_hat[pos_ind])/N_pos;
    results$full_train_neg_errors[1] = sum(Y[neg_ind]!=y_hat[neg_ind])/N_neg;
    results$full_model_sizes[1] = results$full_model$size;
    results$full_train_weighted_errors[1] = (2*class_weights[1]*results$full_train_neg_errors[1]*N_neg + 2*class_weights[2]*results$full_train_pos_errors[1]*N_pos)/N;

    # # test errors for final model
    y_hat = predict(results$full_model, newdata = X_test);
    results$full_test_errors[1] = sum(Y_test!=y_hat)/N_test;
    results$full_test_pos_errors[1] = sum(Y_test[pos_test_ind]!=y_hat[pos_test_ind])/N_test_pos;
    results$full_test_neg_errors[1] = sum(Y_test[neg_test_ind]!=y_hat[neg_test_ind])/N_test_neg;
    results$full_test_weighted_errors[1] = (2*class_weights[1]*results$full_test_neg_errors[1]*N_test_neg + 2*class_weights[2]*results$full_test_pos_errors[1]*N_test_pos)/N_test;

    print_message = sprintf("Finished %s on %s with weights w- = %1.3f w+ = %1.3f \n",
                            method_name,
                            data_name,
                            2*class_weights[1],
                            2*class_weights[2]);

    print.to.console(print_message);
    return(results);
}

## CART
train.cart = function(X, Y, X_test, Y_test, folds){

    print_message = sprintf("Running %s on %s with weights w- = %1.3f w+ = %1.3f",
                            method_name,
                            data_name,
                            2*class_weights[1],
                            2*class_weights[2]);
    print.to.console(print_message);

    #Initialize Cost Matrix
    if (!weighted_flag){
        cost_matrix = matrix(c(0,1.0,1.0,0), byrow=TRUE, nrow=2, dimnames=list(c("0","1"),c("0","1")));
    } else {
        cost_matrix = matrix(c(0,2*class_weights[1],2*class_weights[2],0), byrow=TRUE, nrow=2, dimnames=list(c("0","1"),c("0","1")));
    }

    #Initialize Function Output
    results = list();
    results$method_name = "cart";
    results$run_date = Sys.time();
    results$class_weights = class_weights;
    results$cost_matrix = cost_matrix;
    results$minSplit = train_settings$cart_minSplit;
    results$cp = train_settings$cart_cp;

    nM = length(results$minSplit);
    nC = length(results$cp);

    results$models[[1]] = list();
    for(i in 1:nM) {
        results$models[[1]][[i]] = list();
        for(j in 1:nC) {
            results$models[[1]][[i]][[j]] = list();
            for (k in 1:K) {
                results$models[[1]][[i]][[j]][[k]] = NA;
            }
        }
    }

    results$train_errors = array(NA, c(1,nM,nC,K));
    results$valid_errors = array(NA, c(1,nM,nC,K));
    results$train_pos_errors = array(NA, c(1,nM,nC,K));
    results$train_neg_errors = array(NA, c(1,nM,nC,K));
    results$valid_pos_errors = array(NA, c(1,nM,nC,K));
    results$valid_neg_errors = array(NA, c(1,nM,nC,K));
    results$valid_weighted_errors = array(NA, c(1,nM,nC,K));
    results$train_weighted_errors = array(NA, c(1,nM,nC,K));
    results$model_sizes = array(NA, c(1,nM,nC,K));

    results$full_model[[1]] = list();
    for(i in 1:nM) {
        results$full_model[[1]][[i]] = list();
        for(j in 1:nC) {
            results$full_model[[1]][[i]][[j]] = list();
            results$full_model[[1]][[i]][[j]][[1]] = NA;
        }
    }

    results$full_train_errors = array(NA, c(1,nM,nC,1));
    results$full_train_pos_errors = array(NA, c(1,nM,nC,1));
    results$full_train_neg_errors = array(NA, c(1,nM,nC,1));
    results$full_train_weighted_errors = array(NA, c(1,nM,nC,1));
    results$full_model_sizes = array(NA, c(1,nM,nC,1));

    # Validation Errors
    results$test_errors = array(NA, c(1,nM,nC,K));
    results$pos_test_errors = array(NA, c(1,nM,nC,K));
    results$neg_test_errors = array(NA, c(1,nM,nC,K));
    results$weighted_test_errors= array(NA, c(1,nM,nC,K));

    results$full_test_errors = array(NA, c(1,nM,nC,1));
    results$full_test_pos_errors = array(NA, c(1,nM,nC,1));
    results$full_test_neg_errors = array(NA, c(1,nM,nC,1));
    results$full_test_weighted_errors = array(NA, c(1,nM,nC,1));

    for (m in 1:nM) {
        for (c in 1:nC) {
            #For Each Fold
            for (k in 1:K) {
                X_train = X[k!=folds,];
                Y_train = as.factor(Y[k!=folds]);
                N_train = nrow(X_train)
                train_data = as.data.frame(cbind(X_train,Y_train));
                names(train_data)[P+1] = "Y";
                pos_train_ind = Y_train==1;
                neg_train_ind = !pos_train_ind;
                N_train_pos = sum(pos_train_ind);
                N_train_neg = sum(neg_train_ind);

                #Specify Valiudation Set
                X_valid = X[k==folds,];
                Y_valid = as.factor(Y[k==folds]);
                N_valid = nrow(X_valid)
                valid_data = as.data.frame(cbind(X_valid,Y_valid));
                names(valid_data)[P+1] = "Y";
                pos_valid_ind = Y_valid==1;
                neg_valid_ind = !pos_valid_ind;
                N_valid_pos = sum(pos_valid_ind);
                N_valid_neg = sum(neg_valid_ind);

                #Run CART
                fold_model = trycatch.rpart(Y, train_data, cost_matrix, results, m, c);

                if (typeof(fold_model) != "character") {
                    #Get Model Sizes
                    fit = fold_model;
                    cptable = fit$cptable;
                    n_cp = nrow(fit$cptable);
                    psize = rep(NA,n_cp)

                    for(r in 1:n_cp){
                        pfit = trycatch.cart.prune(fit, cptable, r);
                        if (typeof(pfit) == "character") {
                            break;
                        }
                        psize[r] = length(which(pfit$frame[,"var"]=="<leaf>"))
                    }

                    # Break loop if psize contains empty entries
                    if (any(is.na(psize))) {
                        next;
                    }

                    if (train_settings$cart_prune_to_min && train_settings$cart_prune_to_L0_max) {
                        prune_ind = min(which.min(cptable[psize<=L0_max,"xerror"]));
                    } else if (train_settings$cart_prune_to_min && !train_settings$cart_prune_to_L0_max) {
                        prune_ind = min(which.min(cptable[,"xerror"]));
                    } else if (!train_settings$cart_prune_to_min && train_settings$cart_prune_to_L0_max) {
                        prune_ind = max(which.max(psize[psize<=L0_max]));
                    } else {
                        prune_ind = n_cp;
                    }

                    if(prune_ind < n_cp){
                        fold_model = trycatch.cart.prune(fold_model, cptable, prune_ind);
                    }

                    # move on if the fold_model returns error.
                    if (typeof(fold_model) == 'character') {
                        next;
                    }

                    #store model object
                    results$models[[1]][[m]][[c]][[k]] = fold_model;

                    #store model size
                    results$model_sizes[,m,c,k] = length(which(fold_model$frame[,"var"]=="<leaf>"));

                    # training error metrics
                    y_hat = predict(fold_model, newdata=train_data)
                    y_hat = round(y_hat[,2])
                    results$train_errors[,m,c,k] = sum(Y_train != y_hat) / N_train;
                    results$train_pos_errors[,m,c,k] = sum(Y_train[pos_train_ind]!=y_hat[pos_train_ind])/N_train_pos;
                    results$train_neg_errors[,m,c,k] = sum(Y_train[neg_train_ind]!=y_hat[neg_train_ind])/N_train_neg;
                    results$train_weighted_errors[,m,c,k] = (2*class_weights[1]*results$train_neg_errors[,m,c,k]*N_train_neg + 2*class_weights[2]*results$train_pos_errors[,m,c,k]*N_train_pos)/N_train;

                    # validation error metrics
                    y_hat = predict(fold_model, newdata=valid_data);
                    y_hat = round(y_hat[,2]);
                    results$valid_errors[,m,c,k] = sum(Y_valid != y_hat) / N_valid;
                    results$valid_pos_errors[,m,c,k] = sum(Y_valid[pos_valid_ind]!=y_hat[pos_valid_ind])/N_valid_pos;
                    results$valid_neg_errors[,m,c,k] = sum(Y_valid[neg_valid_ind]!=y_hat[neg_valid_ind])/N_valid_neg;
                    results$valid_weighted_errors[,m,c,k] = (2*class_weights[1]*results$valid_neg_errors[,m,c,k]*N_valid_neg + 2*class_weights[2]*results$valid_pos_errors[,m,c,k]*N_valid_pos)/N_valid;

                    # test error metrics
                    y_hat = predict(fold_model, newdata=test_data);
                    y_hat = round(y_hat[,2]);
                    results$test_errors[,m,c,k] = sum(y_hat!=Y_test)/N_test;
                    results$pos_test_errors[,m,c,k] = sum(y_hat[pos_test_ind]!=Y_test[pos_test_ind])/N_test_pos;
                    results$neg_test_errors[,m,c,k] = sum(y_hat[neg_test_ind]!=Y_test[neg_test_ind])/N_test_neg;
                    results$weighted_test_errors[,m,c,k] = (2*class_weights[1]*results$neg_test_errors[,m,c,k]*N_test_neg + 2*class_weights[2]*results$pos_test_errors[,m,c,k]*N_test_pos)/N_test;

                }
            }

            # fit final (full) model using all data
            full_data = as.data.frame(cbind(X,as.factor(Y)));
            names(full_data)[P+1]="Y"
            full_data$Y = as.factor(full_data$Y)

            full_model = trycatch.rpart(Y, full_data, cost_matrix, results, m, c);

            if (typeof(full_model) != "character") {
                fit = full_model;
                cptable = fit$cptable;
                n_cp = nrow(fit$cptable);
                psize = rep(NA,n_cp)

                for(r in 1:n_cp){
                    pfit = trycatch.cart.prune(fit, cptable, r);
                    if (typeof(pfit) == "character") {
                        break;
                    }
                    psize[r] = length(which(pfit$frame[,"var"]=="<leaf>"))
                }

                # Break loop if psize contains empty entries
                if (any(is.na(psize))) {
                    next;
                }

                if (train_settings$cart_prune_to_min && train_settings$cart_prune_to_L0_max) {
                    prune_ind = min(which.min(cptable[psize<=L0_max,"xerror"]));
                } else if (train_settings$cart_prune_to_min && !train_settings$cart_prune_to_L0_max) {
                    prune_ind = min(which.min(cptable[,"xerror"]));
                } else if (!train_settings$cart_prune_to_min && train_settings$cart_prune_to_L0_max) {
                    prune_ind = max(which.max(psize[psize<=L0_max]));
                } else {
                    prune_ind = n_cp;
                }

                if(prune_ind < n_cp){
                    full_model = trycatch.cart.prune(full_model, cptable, prune_ind);
                }

                if (typeof(full_model) == "character") {
                    next;
                }

                results$full_model[[1]][[m]][[c]][[1]] = full_model;

                y_hat = predict(full_model, newdata=full_data)
                y_hat = round(y_hat[,2]);
                results$full_train_errors[,m,c,1] = sum(Y!=y_hat)/N;
                results$full_train_pos_errors[,m,c,1] = sum(Y[pos_ind]!=y_hat[pos_ind])/N_pos;
                results$full_train_neg_errors[,m,c,1] = sum(Y[neg_ind]!=y_hat[neg_ind])/N_neg;
                results$full_model_sizes[,m,c,1] = length(which(full_model$frame[,"var"]=="<leaf>"));
                results$full_train_weighted_errors[,m,c,1] = (2*class_weights[1]*results$full_train_neg_errors[,m,c,1]*N_neg + 2*class_weights[2]*results$full_train_pos_errors[,m,c,1]*N_pos)/N;

                # # test errors for final model
                y_hat = predict(full_model, newdata=test_data)
                y_hat = round(y_hat[,2]);
                results$full_test_errors[,m,c,1] = sum(Y_test!=y_hat)/N_test;
                results$full_test_pos_errors[,m,c,1] = sum(Y_test[pos_test_ind]!=y_hat[pos_test_ind])/N_test_pos;
                results$full_test_neg_errors[,m,c,1] = sum(Y_test[neg_test_ind]!=y_hat[neg_test_ind])/N_test_neg;
                results$full_test_weighted_errors[,m,c,1] = (2*class_weights[1]*results$full_test_neg_errors[,m,c,1]*N_test_neg + 2*class_weights[2]*results$full_test_pos_errors[,m,c,1]*N_test_pos)/N_test;
            }
        }
    }

    #Print Completion Message
    print_message = sprintf("Finished %s on %s with weights w- = %1.3f w+ = %1.3f",
                            method_name,
                            data_name,
                            2*class_weights[1],
                            2*class_weights[2]);
    print.to.console(print_message);

    return(results)
}

trycatch.rpart = function(Y, train_data, cost_matrix, results, m, c) {
    out = tryCatch(
        {
            rpart(Y~., data=train_data, method="class", parms=list(loss=cost_matrix),
                  control = rpart.control(minsplit=results$minSplit[m], cp=results$cp[c]));
        },
        error = function(e) {
            print(paste0("Error in minSplit = ",results$minSplit[m],", cp = ",results$cp[c],", error = ",e));
            return ("error");
        }
    )
    return (out);
}

trycatch.cart.prune = function(fit, cptable, r) {
    out = tryCatch(
        {
            prune(fit,cp=cptable[r,"CP"])
        },
        error = function(e) {
            print(paste0("Error in Pruning: error = ",e));
            return ("error");
        }
    )
    return (out);
}

## Penalized Logistic Regression
train.plr = function(X, Y, X_test, Y_test, folds, alpha_value = 1.0){


    print_message = sprintf("Running %s on %s with weights w- = %1.3f w+ = %1.3f",
                            method_name,
                            data_name,
                            2*class_weights[1],
                            2*class_weights[2]);
    print.to.console(print_message);

    #Initialize Function Output
    results = list();
    results$run_date = Sys.time();
    results$method_name = method_name;
    results$class_weights = class_weights;

    weights = rep(1.0, N);
    if (weighted_flag) {
        weights = rep(NA,N);
        weights[neg_ind] = 2*class_weights[1];
        weights[pos_ind] = 2*class_weights[2];
    }

    # fit final (full) model using all data to determine lambda values
    results$full_model = cv.glmnet(X, Y, foldid=folds, family="binomial", weights=weights, alpha=alpha_value, nlambda=train_settings$plr_nlambda, type.measure="class", standardize=FALSE);

    C = results$full_model$lambda
    nC = length(C);
    results$C = C;
    results$C1 = alpha_value*C;
    results$C2 = 0.5*(1-alpha_value)*C;
    results$alpha = alpha_value;
    results$models = vector("list", K);
    results$train_errors = array(NA, c(1,1,nC,K));
    results$train_pos_errors = array(NA, c(1,1,nC,K));
    results$train_neg_errors = array(NA, c(1,1,nC,K));
    results$train_weighted_errors = array(NA, c(1,1,nC,K));
    results$valid_errors = array(NA, c(1,1,nC,K));
    results$valid_pos_errors = array(NA, c(1,1,nC,K));
    results$valid_neg_errors = array(NA, c(1,1,nC,K));
    results$valid_weighted_errors = array(NA, c(1,1,nC,K));
    results$model_sizes = array(NA, c(1,1,nC,K));
    results$full_model_sizes = array(NA, c(1,1,nC,1));
    results$full_train_errors = array(NA, c(1,1,nC,1));
    results$full_train_pos_errors = array(NA, c(1,1,nC,1));
    results$full_train_neg_errors = array(NA, c(1,1,nC,1));
    results$full_train_weighted_errors = array(NA, c(1,1,nC,1));
    results$coefficients = array(NA, c(1,P+1,nC,K),dimnames=list(NULL,rownames(coef(results$full_model)),NULL,NULL));
    results$full_coefficients = array(NA, c(1,P+1,nC,1),dimnames=list(NULL,rownames(coef(results$full_model)),NULL,NULL));
    results$se_index = which(C==results$full_model$lambda.1se);
    results$min_index = which(C==results$full_model$lambda.min);

    # Validation Errors
    results$test_errors = array(NA, c(1,1,nC,K));
    results$pos_test_errors = array(NA, c(1,1,nC,K));
    results$neg_test_errors = array(NA, c(1,1,nC,K));
    results$weighted_test_errors= array(NA, c(1,1,nC,K));

    results$full_test_errors = array(NA, c(1,1,nC,1));
    results$full_test_pos_errors = array(NA, c(1,1,nC,1));
    results$full_test_neg_errors = array(NA, c(1,1,nC,1));
    results$full_test_weighted_errors = array(NA, c(1,1,nC,1));

    #For Each Fold
    for (k in 1:K){
        #training set
        X_train = as.matrix(X[k!=folds,]);
        Y_train = as.matrix(Y[k!=folds]);
        N_train = nrow(X_train);
        pos_train_ind = Y_train==1;
        neg_train_ind = !pos_train_ind;
        N_train_pos = sum(pos_train_ind);
        N_train_neg = sum(neg_train_ind);

        #Specify Valiudation Set
        X_valid = as.matrix(X[k==folds,]);
        Y_valid = as.matrix(Y[k==folds]);
        N_valid = nrow(X_valid)
        pos_valid_ind = Y_valid==1;
        neg_valid_ind = !pos_valid_ind;
        N_valid_pos = sum(pos_valid_ind);
        N_valid_neg = sum(neg_valid_ind);

        #Set up the Model
        fold_model = glmnet(X_train, Y_train, family = "binomial", weights = weights[k!=folds], alpha=alpha_value, standardize=FALSE, lambda=C,maxit=1000000);

        #training error metrics
        y_hat = round(predict(fold_model,X_train,type="response"));
        results$train_errors[,,,k] = apply(y_hat,2, function(y) sum(Y_train != y)/N_train);
        results$train_pos_errors[,,,k] = apply(y_hat[pos_train_ind,],2, function(y) sum(Y_train[pos_train_ind,] != y)/N_train_pos);
        results$train_neg_errors[,,,k] = apply(y_hat[neg_train_ind,],2, function(y) sum(Y_train[neg_train_ind,] != y)/N_train_neg);
        results$train_weighted_errors[,,,k] = (2*class_weights[1]*results$train_neg_errors[,,,k]*N_train_neg + 2*class_weights[2]*results$train_pos_errors[,,,k]*N_train_pos)/N_train;

        #validation error metrics
        y_hat = round(predict(fold_model,newx=X_valid,type="response"))
        results$valid_errors[,,,k] = apply(y_hat,2, function(y) sum(Y_valid != y)/N_valid);
        results$valid_pos_errors[,,,k] = apply(y_hat[pos_valid_ind,],2, function(y) sum(Y_valid[pos_valid_ind,] != y)/N_valid_pos);
        results$valid_neg_errors[,,,k] = apply(y_hat[neg_valid_ind,],2, function(y) sum(Y_valid[neg_valid_ind,] != y)/N_valid_neg);
        results$valid_weighted_errors[,,,k] = (2*class_weights[1]*results$valid_neg_errors[,,,k]*N_valid_neg + 2*class_weights[2]*results$valid_pos_errors[,,,k]*N_valid_pos)/N_valid;

        #test error metrics
        y_hat = round(predict(fold_model,newx=X_test,type="response"))
        results$test_errors[,,,k] = apply(y_hat,2, function(y) sum(y!=Y_test)/N_test);
        results$pos_test_errors[,,,k] = apply(y_hat[pos_test_ind,],2, function(y) sum(y!=Y_test[pos_test_ind])/N_test_pos);
        results$neg_test_errors[,,,k] = apply(y_hat[neg_test_ind,],2, function(y) sum(y!=Y_test[neg_test_ind])/N_test_neg);
        results$weighted_test_errors[,,,k] = (2*class_weights[1]*results$neg_test_errors[,,,k]*N_test_neg + 2*class_weights[2]*results$pos_test_errors[,,,k]*N_test_pos)/N_test;

        #coefficients
        coefs = as.array(coef(fold_model));
        results$coefficients[,,,k] = coefs;
        results$model_sizes[,,,k] = apply(coefs, 2, function(b) sum(b!=0.0));

        #model object
        results$models[[k]] = fold_model;
    }

    #fit final model using all data
    results$cv_model = results$full_model;
    results$full_model = glmnet(X, Y, family = "binomial", alpha=alpha_value, weights=weights, standardize = FALSE, lambda=C,maxit=1000000);

    #training error metrics for final model
    y_hat = round(predict(results$full_model,X,type="response"));
    results$full_train_errors[1,,,] = apply(y_hat,2, function(y) sum(Y != y)/N);
    results$full_train_pos_errors[1,,,] = apply(y_hat[pos_ind,],2, function(y) sum(Y[pos_ind,] != y)/N_pos);
    results$full_train_neg_errors[1,,,] = apply(y_hat[neg_ind,],2, function(y) sum(Y[neg_ind,] != y)/N_neg);
    results$full_train_weighted_errors[1,,,] = (2*class_weights[1]*results$full_train_neg_errors[1,,,]*N_neg + 2*class_weights[2]*results$full_train_pos_errors[1,,,]*N_pos)/N;

    #test error metrics for final model
    y_hat = round(predict(results$full_model,X_test,type="response"));
    results$full_test_errors[1,,,] = apply(y_hat,2, function(y) sum(Y_test!=y)/N_test);
    results$full_test_pos_errors[1,,,] = apply(y_hat[pos_test_ind,],2, function(y) sum(Y_test[pos_test_ind]!=y)/N_test_pos);
    results$full_test_neg_errors[1,,,] = apply(y_hat[neg_test_ind,],2, function(y) sum(Y_test[neg_test_ind]!=y)/N_test_neg);
    results$full_test_weighted_errors[1,,,] = (2*class_weights[1]*results$full_test_neg_errors[1,,,]*N_test_neg + 2*class_weights[2]*results$full_test_pos_errors[1,,,]*N_test_pos)/N_test;

    #coefficients for final model
    coefs = as.array(coef(results$full_model));
    results$full_coefficients[,,,1] = coefs;
    results$full_model_sizes[,,,1] = apply(coefs, 2, function(b) sum(b!=0.0));
    results$full_model_sizes[,,,1] = array(results$full_model_sizes,dim = c(1, dim(results$full_model_sizes)));

    #Print Completion Message
    print_message = sprintf("Finished %s (alpha = %1.3f) on %s with weights w- = %1.3f w+ = %1.3f",
                            method_name,
                            alpha_value,
                            data_name,
                            2*class_weights[1],
                            2*class_weights[2]);
    print.to.console(print_message);

    return(results)
}

## Random Forests
train.randomforest = function(X, Y, X_test, Y_test, folds){
    #
    print_message = sprintf("Running %s on %s with weights w- = %1.3f w+ = %1.3f",
                            method_name, data_name, 2*class_weights[1], 2*class_weights[2]);
    print.to.console(print_message);

    #Initialize Function Output
    results = list();
    results$method_name = "randomforest";
    results$run_date = Sys.time();

    results$class_weights = class_weights;
    results$sampsize = train_settings$randomforest_sampsize;
    results$nodesize = train_settings$randomforest_nodesize;

    nS = length(results$sampsize);
    nN = length(results$nodesize);

    results$train_errors = array(NA, c(1,nS,nN,K));
    results$valid_errors = array(NA, c(1,nS,nN,K));
    results$train_pos_errors = array(NA, c(1,nS,nN,K));
    results$valid_pos_errors = array(NA, c(1,nS,nN,K));
    results$train_neg_errors = array(NA, c(1,nS,nN,K));
    results$valid_neg_errors = array(NA, c(1,nS,nN,K));
    results$valid_weighted_errors = array(NA, c(1,nS,nN,K));
    results$train_weighted_errors = array(NA, c(1,nS,nN,K));
    results$models = NULL
    results$model_sizes = array(NA, c(1,nS,nN,K));

    results$full_train_errors = array(NA, c(1,nS,nN,1));
    results$full_train_pos_errors = array(NA, c(1,nS,nN,1));
    results$full_train_neg_errors = array(NA, c(1,nS,nN,1));
    results$full_train_weighted_errors = array(NA, c(1,nS,nN,1));
    results$full_model_sizes = NULL

    # Validation Errors
    results$test_errors = array(NA, c(1,nS,nN,K));
    results$pos_test_errors = array(NA, c(1,nS,nN,K));
    results$neg_test_errors = array(NA, c(1,nS,nN,K));
    results$weighted_test_errors= array(NA, c(1,nS,nN,K));

    results$full_test_errors = array(NA, c(1,nS,nN,1));
    results$full_test_pos_errors = array(NA, c(1,nS,nN,1));
    results$full_test_neg_errors = array(NA, c(1,nS,nN,1));
    results$full_test_weighted_errors = array(NA, c(1,nS,nN,1));

    for (s in 1:nS) {
        for (n in 1:nN) {
            #For Each Fold
            for (k in 1:K){

                #training set
                X_train = as.matrix(X[k!=folds,]);
                Y_train = as.factor(Y[k!=folds]);
                N_train = nrow(X_train);
                pos_train_ind = Y_train==1;
                neg_train_ind = !pos_train_ind;
                N_train_pos = sum(pos_train_ind);
                N_train_neg = sum(neg_train_ind);

                #Specify Validation Set
                X_valid = as.matrix(X[k==folds,]);
                Y_valid = as.factor(Y[k==folds]);
                N_valid = nrow(X_valid);
                pos_valid_ind = Y_valid==1;
                neg_valid_ind = !pos_valid_ind;
                N_valid_pos = sum(pos_valid_ind);
                N_valid_neg = sum(neg_valid_ind);

                #Set up the Model
                tmp_weights = class_weights*2+0;
                fold_model = randomForest(x = X_train,
                                          y = Y_train,
                                          classwt = tmp_weights,
                                          sampsize = results$sampsize[s],
                                          nodesize = results$nodesize[n]);

                #note: RF does not report training error on actual data. for an explanation see:
                #http://stats.stackexchange.com/questions/162353/what-measure-of-training-error-to-report-for-random-forests/162924#162924

                # training error metrics
                y_hat = predict(fold_model) #OOB error
                #y_hat = predict(fold_model, newx = X_train) #real training error
                results$train_errors[,s,n,k] = sum(y_hat!=Y_train)/N_train;
                results$train_pos_errors[,s,n,k] = sum(y_hat[pos_train_ind]!=Y_train[pos_train_ind])/N_train_pos;
                results$train_neg_errors[,s,n,k] = sum(y_hat[neg_train_ind]!=Y_train[neg_train_ind])/N_train_neg;
                results$train_weighted_errors[,s,n,k] = (2*class_weights[1]*results$train_neg_errors[,s,n,k]*N_train_neg + 2*class_weights[2]*results$train_pos_errors[,s,n,k]*N_train_pos)/N_train;

                # validation error metrics
                y_hat = predict(fold_model, X_valid)
                results$valid_errors[,s,n,k] = sum(y_hat!=Y_valid)/N_valid;
                results$valid_pos_errors[,s,n,k] = sum(y_hat[pos_valid_ind]!=Y_valid[pos_valid_ind])/N_valid_pos;
                results$valid_neg_errors[,s,n,k] = sum(y_hat[neg_valid_ind]!=Y_valid[neg_valid_ind])/N_valid_neg;
                results$valid_weighted_errors[,s,n,k] = (2*class_weights[1]*results$valid_neg_errors[,s,n,k]*N_valid_neg + 2*class_weights[2]*results$valid_pos_errors[,s,n,k]*N_valid_pos)/N_valid;

                # test error metrics
                y_hat = predict(fold_model, X_test)
                results$test_errors[,s,n,k] = sum(y_hat!=Y_test)/N_test;
                results$pos_test_errors[,s,n,k] = sum(y_hat[pos_test_ind]!=Y_test[pos_test_ind])/N_test_pos;
                results$neg_test_errors[,s,n,k] = sum(y_hat[neg_test_ind]!=Y_test[neg_test_ind])/N_test_neg;
                results$weighted_test_errors[,s,n,k] = (2*class_weights[1]*results$neg_test_errors[,s,n,k]*N_test_neg + 2*class_weights[2]*results$pos_test_errors[,s,n,k]*N_test_pos)/N_test;

                #Store Model
                #results$models[[k]] = fold_model;
            }

            #fit final ("full") model using all data
            tmp_weights = class_weights*2+0;
            full_model = randomForest(x = X,
                                      y = as.factor(Y),
                                      classwt = tmp_weights,
                                      sampsize=results$sampsize[s],nodesize=results$nodesize[n])
            y_hat = predict(full_model);
            results$full_train_errors[,s,n,1] = sum(Y!=y_hat)/N;
            results$full_train_pos_errors[,s,n,1] = sum(Y[pos_ind]!=y_hat[pos_ind])/N_pos;
            results$full_train_neg_errors[,s,n,1] = sum(Y[neg_ind]!=y_hat[neg_ind])/N_neg;
            results$full_train_weighted_errors[,s,n,1] = (2*class_weights[1]*results$full_train_neg_errors[,s,n,1]*N_neg + 2*class_weights[2]*results$full_train_pos_errors[,s,n,1]*N_pos)/N;

            # # test errors for final model
            y_hat = predict(full_model, X_test);
            results$full_test_errors[,s,n,1] = sum(Y_test!=y_hat)/N_test;
            results$full_test_pos_errors[,s,n,1] = sum(Y_test[pos_test_ind]!=y_hat[pos_test_ind])/N_test_pos;
            results$full_test_neg_errors[,s,n,1] = sum(Y_test[neg_test_ind]!=y_hat[neg_test_ind])/N_test_neg;
            results$full_test_weighted_errors[,s,n,1] = (2*class_weights[1]*results$full_test_neg_errors[,s,n,1]*N_test_neg + 2*class_weights[2]*results$full_test_pos_errors[,s,n,1]*N_test_pos)/N_test;

        }
    }

    #Print Completion Message
    print_message = sprintf("Finished %s on %s with weights w- = %1.3f w+ = %1.3f",
                            method_name,
                            data_name,
                            2*class_weights[1],
                            2*class_weights[2]);

    print.to.console(print_message);
    return(results);
}

## SVM
train.svm_rbf = function(X, Y, X_test, Y_test, folds, svm_costs=10^seq(-3,3,0.5)){
    results = train.svm(X, Y, X_test, Y_test, folds, svm_kernel = "radial", svm_costs);
    results$method_name = method_name;
    results$run_date = Sys.time();
    return(results)
}

train.svm = function(X, Y,  X_test, Y_test, folds, svm_kernel="radial", svm_costs=10^seq(-3,3,0.5)){

    print_message = sprintf("Running %s on %s with weights w- = %1.3f w+ = %1.3f",
                            method_name,data_name, 2*class_weights[1], 2*class_weights[2]);
    print.to.console(print_message);

    if (!weighted_flag){
        class.weights = c(1.0,1.0);
        names(class.weights) = c("0","1");
    } else {
        class.weights = 2*class_weights;
        names(class.weights) = c("0","1");
    }

    nC = length(train_settings$svm_costs);
    nG = length(train_settings$svm_gamma);

    #Initialize Function Output
    results = list();
    results$C = train_settings$svm_costs;
    results$G = train_settings$svm_gamma;
    results$class_weights = class_weights;
    results$train_errors = array(NA, c(1,nG,nC,K));
    results$valid_errors = array(NA, c(1,nG,nC,K));
    results$valid_weighted_errors = array(NA, c(1,nG,nC,K));
    results$train_weighted_errors = array(NA, c(1,nG,nC,K));
    results$train_pos_errors = array(NA, c(1,nG,nC,K));
    results$valid_pos_errors = array(NA, c(1,nG,nC,K));
    results$train_neg_errors = array(NA, c(1,nG,nC,K));
    results$valid_neg_errors = array(NA, c(1,nG,nC,K));
    results$models = list();
    results$model_sizes = array(NA, c(1,nG,nC,K));

    results$full_train_errors = array(NA, c(1,nG,nC,1));
    results$full_train_weighted_errors = array(NA, c(1,nG,nC,1));
    results$full_train_pos_errors = array(NA, c(1,nG,nC,1));
    results$full_train_neg_errors = array(NA, c(1,nG,nC,1));
    results$full_model = NULL;
    results$full_model_sizes = array(NA, c(1,nG,nC,1));

    # validation errors
    results$test_errors = array(NA, c(1,nG,nC,K));
    results$pos_test_errors = array(NA, c(1,nG,nC,K));
    results$neg_test_errors = array(NA, c(1,nG,nC,K));
    results$weighted_test_errors= array(NA, c(1,nG,nC,K));

    results$full_test_errors = array(NA, c(1,nG,nC,1));
    results$full_test_pos_errors = array(NA, c(1,nG,nC,1));
    results$full_test_neg_errors = array(NA, c(1,nG,nC,1));
    results$full_test_weighted_errors = array(NA, c(1,nG,nC,1));

    if (svm_kernel=="linear") {
        results$coefficients = array(NA, c(1,P+1,nC,K),dimnames = list(NULL,variable_names,NULL,NULL));
        results$full_coefficients = array(NA, c(1,P+1,nC,1),dimnames = list(NULL,variable_names,NULL,NULL));
    }

    #train model
    for (g in 1:nG) {
        for(v in 1:nC){
            print.to.console(sprintf("Running %s for (G, C) = (%1.2e, %1.2e) ", method_name, results$G[g], svm_costs[v]));

            #For Each Fold
            for (k in 1:K){

                # training set
                X_train = as.matrix(X[k!=folds,]);
                Y_train = as.factor(Y[k!=folds]);
                N_train = nrow(X_train);
                pos_train_ind = Y_train==1;
                neg_train_ind = !pos_train_ind;
                N_train_pos = sum(pos_train_ind);
                N_train_neg = sum(neg_train_ind);

                # validation set
                X_valid = as.matrix(X[k==folds,]);
                Y_valid = as.factor(Y[k==folds]);
                N_valid = nrow(X_valid);
                pos_valid_ind = Y_valid==1;
                neg_valid_ind = !pos_valid_ind;
                N_valid_pos = sum(pos_valid_ind);
                N_valid_neg = sum(neg_valid_ind);

                fold_model = svm(x=X_train,y=Y_train,cost=results$C[v],gamma=results$G[g],kernel=svm_kernel,scale=TRUE,type="C-classification", class.weights=class.weights);

                # training error metrics
                y_hat = predict(fold_model,X_train);
                results$train_errors[,g,v,k] = sum(y_hat!=Y_train)/N_train;
                results$train_pos_errors[,g,v,k] = sum(y_hat[pos_train_ind]!=Y_train[pos_train_ind])/N_train_pos;
                results$train_neg_errors[,g,v,k] = sum(y_hat[neg_train_ind]!=Y_train[neg_train_ind])/N_train_neg;
                results$train_weighted_errors[,g,v,k] = (2*class_weights[1]*results$train_neg_errors[,g,v,k]*N_train_neg + 2*class_weights[2]*results$train_pos_errors[,g,v,k]*N_train_pos)/N_train;

                # validation error metrics
                y_hat = predict(fold_model,X_valid);
                results$valid_errors[,g,v,k] = sum(y_hat!=Y_valid)/N_valid;
                results$valid_pos_errors[,g,v,k] = sum(y_hat[pos_valid_ind]!=Y_valid[pos_valid_ind])/N_valid_pos;
                results$valid_neg_errors[,g,v,k] = sum(y_hat[neg_valid_ind]!=Y_valid[neg_valid_ind])/N_valid_neg;
                results$valid_weighted_errors[,g,v,k] = (2*class_weights[1]*results$valid_neg_errors[,g,v,k]*N_valid_neg + 2*class_weights[2]*results$valid_pos_errors[,g,v,k]*N_valid_pos)/N_valid;

                # test error metrics
                y_hat = predict(fold_model,X_test);
                results$test_errors[,g,v,k] = sum(y_hat!=Y_test)/N_test;
                results$pos_test_errors[,g,v,k] = sum(y_hat[pos_test_ind]!=Y_test[pos_test_ind])/N_test_pos;
                results$neg_test_errors[,g,v,k] = sum(y_hat[neg_test_ind]!=Y_test[neg_test_ind])/N_test_neg;
                results$weighted_test_errors[,g,v,k] = (2*class_weights[1]*results$neg_test_errors[,g,v,k]*N_test_neg + 2*class_weights[2]*results$pos_test_errors[,g,v,k]*N_test_pos)/N_test;

                # model size metrics
                if(svm_kernel=="linear"){
                    tryCatch({
                        coefs = rbind(-fold_model$rho,t(t(fold_model$coefs) %*% X_train[fold_model$index,]))
                    }, error = function(e) {
                        coefs = rep(-1,P+1)
                    }, finally = {
                    })
                    results$coefficients[,g,v,k] = as.array(coefs)
                    results$model_sizes[,g,v,k] = apply(coefs,2, function(b) sum(b!=0.0));
                }

            }

            # fit final (full) model using all data
            full_model = svm(X,y=as.factor(Y),cost=results$C[v],gamma=results$G[g],kernel=svm_kernel,scale=TRUE,type="C-classification",class.weights=class.weights)

            #Calculate weighted_flag Training Errors
            y_hat = predict(full_model,newdata=X);
            results$full_train_errors[,g,v,1] = sum(Y!=y_hat)/N;
            results$full_train_pos_errors[,g,v,1] = sum(Y[pos_ind]!=y_hat[pos_ind])/N_pos;
            results$full_train_neg_errors[,g,v,1] = sum(Y[neg_ind]!=y_hat[neg_ind])/N_neg;
            results$full_train_weighted_errors[,g,v,1] = (2*class_weights[1]*results$full_train_neg_errors[,g,v,1]*N_neg + 2*class_weights[2]*results$full_train_pos_errors[,g,v,1]*N_pos)/N;

            # # test errors for final model
            y_hat= predict(full_model, newdata=X_test);
            results$full_test_errors[,g,v,1] = sum(Y_test!=y_hat)/N_test;
            results$full_test_pos_errors[,g,v,1] = sum(Y_test[pos_test_ind]!=y_hat[pos_test_ind])/N_test_pos;
            results$full_test_neg_errors[,g,v,1] = sum(Y_test[neg_test_ind]!=y_hat[neg_test_ind])/N_test_neg;
            results$full_test_weighted_errors[,g,v,1] = (2*class_weights[1]*results$full_test_neg_errors[,g,v,1]*N_test_neg + 2*class_weights[2]*results$full_test_pos_errors[,g,v,1]*N_test_pos)/N_test;

            if ( svm_kernel == "linear" ){
                coefs = rbind(-full_model$rho, t(t(full_model$coefs) %*% X[full_model$index,]));
                results$full_coefficients[,g,v,1] = as.array(coefs);
                results$full_model_sizes[,g,v,1] = apply(coefs, 2, function(b) sum(b!=0.0));
            }
        }
    }

    #Completion Message
    print_message = sprintf("Finished %s on %s with weights w- = %1.3f w+ = %1.3f",
                            method_name,
                            data_name,
                            2*class_weights[1],
                            2*class_weights[2]);

    print.to.console(print_message);
    return(results);
}

### SGB
train.sgb = function(X, Y, X_test, Y_test, folds){

    print_message = sprintf("Running %s on %s with weights w- = %1.3f w+ = %1.3f",
                            method_name,
                            data_name,
                            2*class_weights[1],
                            2*class_weights[2]);
    print.to.console(print_message);

    #Initialize Weights
    weights = rep(1,N);
    if (weighted_flag) {
        weights[neg_ind] = 2*class_weights[1];
        weights[pos_ind] = 2*class_weights[2];
    }

    #Initialize Function Output
    results = list();
    results$method_name = "boosting";
    results$run_date = Sys.time();

    results$class_weights = class_weights;
    results$shrinkage = train_settings$sgb_shrinkage;
    results$depth = train_settings$sgb_depth;
    results$ntrees = train_settings$sgb_ntrees;

    nD = length(results$depth);
    nS = length(results$shrinkage);
    nT = length(results$ntrees);

    results$train_errors = array(NA, c(nD,nS,nT,K));
    results$valid_errors = array(NA, c(nD,nS,nT,K));
    results$train_pos_errors = array(NA, c(nD,nS,nT,K));
    results$valid_pos_errors = array(NA, c(nD,nS,nT,K));
    results$train_neg_errors = array(NA, c(nD,nS,nT,K));
    results$valid_neg_errors = array(NA, c(nD,nS,nT,K));
    results$valid_weighted_errors = array(NA, c(nD,nS,nT,K));
    results$train_weighted_errors = array(NA, c(nD,nS,nT,K));

    results$models = list();
    results$model_sizes = array(NA, c(nD,nS,nT,K));

    results$full_train_errors = array(NA, c(nD,nS,nT,1));
    results$full_train_pos_errors = array(NA, c(nD,nS,nT,1));
    results$full_train_neg_errors = array(NA, c(nD,nS,nT,1));
    results$full_train_weighted_errors = array(NA, c(nD,nS,nT,1));

    # Validation Errors
    results$test_errors = array(NA, c(nD,nS,nT,K));
    results$pos_test_errors = array(NA, c(nD,nS,nT,K));
    results$neg_test_errors = array(NA, c(nD,nS,nT,K));
    results$weighted_test_errors= array(NA, c(nD,nS,nT,K));

    results$full_test_errors = array(NA, c(nD,nS,nT,1));
    results$full_test_pos_errors = array(NA, c(nD,nS,nT,1));
    results$full_test_neg_errors = array(NA, c(nD,nS,nT,1));
    results$full_test_weighted_errors = array(NA, c(nD,nS,nT,1));

    results$full_model_sizes = array(NA, c(nD,nS,nT,1));
    results$full_model = NULL;

    for (d in 1:nD) {
        for (s in 1:nS) {
            for (t in 1:nT) {
                #for each fold
                for (k in 1:K){

                    #training set
                    X_train = X[k!=folds,];
                    Y_train = Y[k!=folds];
                    N_train = nrow(X_train)
                    train_data = as.data.frame(cbind(X_train,Y_train));
                    names(train_data)[P+1] = "Y";
                    pos_train_ind = Y_train==1;
                    neg_train_ind = !pos_train_ind;
                    N_train_pos = sum(pos_train_ind);
                    N_train_neg = sum(neg_train_ind);

                    #Specify Validation Set
                    X_valid = X[k==folds,];
                    Y_valid = Y[k==folds];
                    N_valid = nrow(X_valid)
                    valid_data = as.data.frame(cbind(X_valid,Y_valid));
                    names(valid_data)[P+1] = "Y";
                    pos_valid_ind = Y_valid==1;
                    neg_valid_ind = !pos_valid_ind;
                    N_valid_pos = sum(pos_valid_ind);
                    N_valid_neg = sum(neg_valid_ind);

                    #Run SGB
                    fold_model = gbm(Y~., distribution="adaboost",data=train_data, w=weights[k!=folds],
                                     n.trees = results$ntrees[t], interaction.depth=results$depth[d],
                                     n.minobsinnode=10, shrinkage=results$shrinkage[s]);

                    # training error metrics
                    y_hat = round(predict(fold_model, newdata=train_data, n.trees=results$ntrees[t],type="response"))
                    results$train_errors[d,s,t,k] = sum(y_hat!=Y_train)/N_train;
                    results$train_pos_errors[d,s,t,k] = sum(y_hat[pos_train_ind]!=Y_train[pos_train_ind])/N_train_pos;
                    results$train_neg_errors[d,s,t,k] = sum(y_hat[neg_train_ind]!=Y_train[neg_train_ind])/N_train_neg;
                    results$train_weighted_errors[d,s,t,k] = (2*class_weights[1]*results$train_neg_errors[d,s,t,k]*N_train_neg + 2*class_weights[2]*results$train_pos_errors[d,s,t,k]*N_train_pos)/N_train;

                    # validation error metrics
                    y_hat = round(predict(fold_model, newdata=valid_data, n.trees=results$ntrees[t],type="response"))
                    results$valid_errors[d,s,t,k] = sum(y_hat!=Y_valid)/N_valid;
                    results$valid_pos_errors[d,s,t,k] = sum(y_hat[pos_valid_ind]!=Y_valid[pos_valid_ind])/N_valid_pos;
                    results$valid_neg_errors[d,s,t,k] = sum(y_hat[neg_valid_ind]!=Y_valid[neg_valid_ind])/N_valid_neg;
                    results$valid_weighted_errors[d,s,t,k] = (2*class_weights[1]*results$valid_neg_errors[d,s,t,k]*N_valid_neg + 2*class_weights[2]*results$valid_pos_errors[d,s,t,k]*N_valid_pos)/N_valid;

                    # test error metrics
                    y_hat = round(predict(fold_model, newdata=test_data, n.trees=results$ntrees[t],type="response"))
                    results$test_errors[d,s,t,k] = sum(y_hat!=Y_test)/N_test;
                    results$pos_test_errors[d,s,t,k] = sum(y_hat[pos_test_ind]!=Y_test[pos_test_ind])/N_test_pos;
                    results$neg_test_errors[d,s,t,k] = sum(y_hat[neg_test_ind]!=Y_test[neg_test_ind])/N_test_neg;
                    results$weighted_test_errors[d,s,t,k] = (2*class_weights[1]*results$neg_test_errors[d,s,t,k]*N_test_neg + 2*class_weights[2]*results$pos_test_errors[d,s,t,k]*N_test_pos)/N_test;

                    #Store Model
                    results$model_sizes[d,s,t,k] = fold_model$n.trees;
                }

                #fit final ("full") model using all data
                full_data = as.data.frame(cbind(X,Y));
                names(full_data)[P+1]="Y"
                full_model = gbm(Y~., distribution="adaboost",data=full_data, w=weights,
                                 n.trees = results$ntrees[t], interaction.depth = results$depth[d],
                                 n.minobsinnode = 10, shrinkage = results$shrinkage[s]);

                results$full_model_sizes[d,s,t,1] = full_model$n.trees;

                # Full Training Errors
                y_hat = round(predict(full_model, newdata=full_data, n.trees=results$ntrees[t],type="response"));
                results$full_train_errors[d,s,t,1] = sum(Y!=y_hat)/N;
                results$full_train_pos_errors[d,s,t,1] = sum(Y[pos_ind]!=y_hat[pos_ind])/N_pos;
                results$full_train_neg_errors[d,s,t,1] = sum(Y[neg_ind]!=y_hat[neg_ind])/N_neg;
                results$full_train_weighted_errors[d,s,t,1] = (2*class_weights[1]*results$full_train_neg_errors[d,s,t,1]*N_neg + 2*class_weights[2]*results$full_train_pos_errors[d,s,t,1]*N_pos)/N;

                # # test errors for final model
                y_hat = round(predict(full_model, newdata=test_data, n.trees=results$ntrees[t],type="response"));
                results$full_test_errors[d,s,t,1] = sum(Y_test!=y_hat)/N_test;
                results$full_test_pos_errors[d,s,t,1] = sum(Y_test[pos_test_ind]!=y_hat[pos_test_ind])/N_test_pos;
                results$full_test_neg_errors[d,s,t,1] = sum(Y_test[neg_test_ind]!=y_hat[neg_test_ind])/N_test_neg;
                results$full_test_weighted_errors[d,s,t,1] = (2*class_weights[1]*results$full_test_neg_errors[d,s,t,1]*N_test_neg + 2*class_weights[2]*results$full_test_pos_errors[d,s,t,1]*N_test_pos)/N_test;

            }
        }
    }

    #Print Completion Message
    print_message = sprintf("Finished %s on %s with weights w- = %1.3f w+ = %1.3f",
                            method_name,
                            data_name,
                            2*class_weights[1],
                            2*class_weights[2]);
    print.to.console(print_message);
    return(results);
}

#### Train Models ####

#global variables for training / validation set
N = nrow(X);
P = ncol(X);
pos_ind = Y==1;
neg_ind = !pos_ind;
K = max(folds)
N_pos = sum(pos_ind);
N_neg = sum(neg_ind);
variable_names = c("(Intercept)", colnames(X));

#global variables for test set
N_test = nrow(X_test);
test_data = as.data.frame(cbind(X_test,Y_test));
names(test_data)[P+1] = "Y";
pos_test_ind = Y_test==1;
neg_test_ind = !pos_test_ind;
N_test_pos = sum(pos_test_ind);
N_test_neg = sum(neg_test_ind);

#class weights
class_weights = matrix(class_weights, nrow = 1, ncol = 2, byrow=TRUE)
class_weights = class_weights/sum(class_weights);
weighted_flag = !(is.null(class_weights) || class_weights[1]==class_weights[2]);
output = list();

#PLR + Lasso Penalty
method_name = "plr_lasso";
if (run.method(method_name)) {
    output$plr_lasso = train.plr(X, Y, X_test, Y_test, folds, alpha_value = 1.0);
}

#PLR + Ridge Penalty
method_name = "plr_ridge";
if (run.method(method_name)) {
    output$plr_ridge = train.plr(X, Y, X_test, Y_test, folds, alpha_value = 0.0);
}

#C5.0 (Rule-Based)
method_name = "C50_rule"
if (run.method(method_name)) {
    output$C50_rule = train.C50(X, Y, X_test, Y_test, folds, train_rules = TRUE);
}

#C5.0 (Tree-Based)
method_name = "C50_tree"
if (run.method(method_name)) {
    output$C50_rule = train.C50(X, Y, X_test, Y_test, folds, train_rules = FALSE);
}

#CART Decision Trees
method_name = "cart"
if (run.method(method_name)) {
    output$cart = train.cart(X, Y, X_test, Y_test, folds);
}

#Random Forests
method_name = "randomforest"
if (run.method(method_name)) {
    output$randomforest = train.randomforest(X, Y, X_test, Y_test, folds);
}

#SVM (RBF Kernel)
method_name = "svm_rbf"
if (run.method(method_name)) {
    output$svm_rbf = train.svm_rbf(X, Y, X_test, Y_test, folds, svm_costs = train_settings$svm_costs);
}

#SGB
method_name = "sgb"
if (run.method(method_name)) {
    output$sgb = train.sgb(X, Y, X_test, Y_test, folds);
}

#Save Data in R Format
save(output, file = results_file_name);


