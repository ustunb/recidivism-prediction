%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Code for Interpretable Models for Recidivism Prediction
% by Jiaming Zeng/Berk Ustun/Cynthia Rudin
%
% Contact: ustunb@mit.edu
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script will train SLIM scoring systems for fixed dataset, and set of
% class weights. 
%
% Requirements:
%
% 1. slim-matlab and prerequisites (https://github.com/ustunb/slim-matlab)
%
% For this script to work correctly, you must change your working directory
% to the root directory of the repository (i.e. the directory that contains 
% this script, /data and /results subdirectories).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;
dbstop if error;

%% User Parameters

data_name = 'arrest';           %options: 'arrest', 'drug', 'domestic_violence', 'general_violence', 'fatal_violence' ,' sexual_violence'
class_weights = [1.00, 1.00];   %[class weight on negative class, class weight on positive class]

cplex_solver_time = 300;        %seconds to solve SLIM IP using CPLEX (we used 12000/IP in the paper)
cplex_polishing_time = 0;       %seconds to use CPLEX polishing procedure on SLIM IP (optional --  feasible solution, we used 1200 per IP)
cplex_populate_time = 30;       %seconds to use CPLEX populate procedure on SLIM IP (optionalfinds feasible solution, we used 1200 per IP)
active_set_polishing_time = 5;  %seconds to run active set polishing (very fast)

%% Setup

%check that user has set script parameters correctly
assert(cplex_solver_time > 0.0)
assert(cplex_polishing_time >= 0.0);
assert(cplex_populate_time >= 0.0);
assert(all(active_set_polishing_time > 0.0));

%normalize class weights
class_weights = 2.00 * class_weights./sum(class_weights);

%set key files and directories
home_dir = [pwd,'/'];
data_dir = [home_dir, 'data/'];
results_dir = [home_dir, 'results/'];
mkdir(results_dir)
run_name = sprintf('%s_wneg_%1.9f_wpos_%1.9f', data_name, class_weights(1), class_weights(2));
data_file_name = [data_dir, data_name,'.mat'];
results_file_name = [results_dir, run_name, '_SLIM_results.mat'];

%load data set
data = load(data_file_name);
X = data.X;
Y = data.Y;
X_names = data.X_names;
Y_name = data.Y_name;
X_test = data.X_test;
Y_test = data.Y_test;
folds = data.folds;
K = max(folds);
clear data;

%create default input for slim IP
input = struct();
input.w_neg = class_weights(1);
input.w_pos = class_weights(2);
input.X_names = X_names;
input.Y_name = Y_name;

%coefficient constraints
coef_cons = SLIMCoefficientConstraints(X_names);  %by default each coefficient lambda_j is an integer from -10 to 10
coef_cons = coef_cons.setfield('(Intercept)', 'C_0j', 0); %the regularization penalty for the intercept should be set to 0 manually
coef_cons = coef_cons.setfield('(Intercept)', 'ub', 100);
coef_cons = coef_cons.setfield('(Intercept)', 'lb', -100);
input.coefConstraints = coef_cons;

%min/max # of input variables
input.L0_min = 0;
input.L0_max = 8;
%% Fit SLIM Models

all_slim_coefficients = cell(K+1, 1);

% solve K + 1 slim IPs
% 1 final model using the full dataset for actual use
% K models using subsets of dataset to assess accuracy via K-fold CV
for k = 1:(K + 1)
    
    train_ind = folds~=k;
    input.X = X(train_ind, :);
    input.Y = Y(train_ind, :);
    [N, P] = size(input.X);
    
    %set C_0 (feature selection parameter)
    %C_0 is a value between (0, 1)
    %C_0 is the minimum training accuracy required to use a feature to the model
    %we set C_0 to a value that is small enough to guarantee that SLIM will use
    %any feature so long as it improves training accuracy (up to the max of 8).
    n_regularized_coefficients = sum(P - sum(coef_cons.C_0j==0.0));
    max_model_size = min(input.L0_max, n_regularized_coefficients);
    input.C_0 = min([input.w_neg, input.w_pos])/(N * max_model_size);
    
    %createSLIM creates a Cplex object, slim_IP and provides useful info in slim_info
    [slim_IP, slim_info] = createSLIM(input);
    
    %set CPLEX solver parameters
    slim_IP.Param.timelimit.Cur                     = cplex_solver_time;
    slim_IP.Param.threads.Cur                       = 1; %# of threads; >1 starts a parallel solver
    slim_IP.Param.output.clonelog.Cur               = 0; %disable CPLEX's clone log
    slim_IP.Param.mip.tolerances.lowercutoff.Cur    = 0;
    slim_IP.Param.emphasis.mip.Cur                  = 1; %mip solver strategy
    slim_IP.Param.randomseed.Cur                    = 0;
    slim_IP.Param.mip.pool.capacity.Cur             = 500; %# of feasbile solutions to keep for polishing after
    slim_IP.Param.mip.pool.replace.Cur              = 1;%
    %slim_IP.DisplayFunc = [] %uncomment to prevent on screen CPLEX display
    
    %solve CPLEX IP
    slim_IP.solve()
    
    %use CPLEX populate algorithm to find additional solutions (optional)
    if cplex_populate_time > 0
        slim_IP.Param.timelimit.Cur = cplex_polishing_time;
        slim_IP.Param.mip.limits.populate.Cur = slim_IP.Param.mip.limits.populate.Max;
        slim_IP.populate();
        slim_IP.Param.mip.limits.populate.Cur = slim_IP.Param.mip.limits.populate.Def;
    end
    
    %use CPLEX polishing algorithm to improve best solution (optional)
    if cplex_polishing_time > 0
        slim_IP.Param.timelimit.Cur = cplex_populate_time;
        slim_IP.Param.mip.polishafter.time.Cur = 0;
        slim_IP.solve();
        slim_IP.Param.mip.polishafter.time.Cur = slim_IP.Param.mip.polishafter.time.Def;
    end
    
    %record best set of coefficients as backup
    slim_summary = getSLIMSummary(slim_IP, slim_info);
    slim_coefficients = slim_summary.coefficients;
    
    %run active set polishing on solutions in solution pool to improve solution
    if active_set_polishing_time > 0
        polished_pool = runActiveSetPolishing(slim_IP, slim_info, active_set_polishing_time);
        if ~isempty(polished_pool(1).coefficients)
            slim_coefficients = polished_pool(1).coefficients;
        end
    end
    
    %record coefficients in results
    all_slim_coefficients{k} = slim_coefficients;
    
end

%% Get Summary Statistics

%helper functions
get_error = @(x, y, l) mean(y~=sign((x*l)-0.5));
get_pos_error = @(x, y, l) mean((x(y>0,:)*l)<1);
get_neg_error = @(x, y, l) mean((x(y<0,:)*l)>0);
non_intercept_idx = strcmp('(Intercept)', X_names);
get_model_size = @(l) sum(l(non_intercept_idx)~=0.0);

%helper variables
w_neg = class_weights(1);
w_pos = class_weights(2);
full_model = all_slim_coefficients{K+1};

results = struct();
results.run_date = datestr(now);
results.method_name = 'slim';
results.class_weights = class_weights;

%coefficients and model sizes
results.coefficients = all_slim_coefficients(1:K)';
results.full_coefficients = {full_model};
results.model_sizes = cellfun(get_model_size, results.coefficients);
results.full_model_sizes = cellfun(get_model_size, results.full_coefficients);

%training error metrics for final model (trained with full dataset)
results.full_train_errors = get_error(X, Y, full_model);
results.full_train_pos_errors = get_pos_error(X, Y, full_model);
results.full_train_neg_errors = get_neg_error(X, Y, full_model);

%test error metrics for final model (trained with full dataset)
results.full_test_errors = get_error(X_test, Y_test, full_model);
results.full_test_pos_errors = get_pos_error(X_test, Y_test, full_model);
results.full_test_neg_errors = get_neg_error(X_test, Y_test, full_model);

%training error metrics for fold-based models
results.train_errors = NaN(1,K);
results.train_pos_errors = NaN(1,K);
results.train_neg_errors = NaN(1,K);
results.train_weighted_errors = NaN(1,K);

%validation error metrics for fold-based models
results.valid_errors = NaN(1,K);
results.valid_pos_errors = NaN(1,K);
results.valid_neg_errors = NaN(1,K);
results.valid_weighted_errors = NaN(1,K);

%test error metrics for fold-based models
results.test_errors = NaN(1,K);
results.test_pos_errors = NaN(1,K);
results.test_neg_errors = NaN(1,K);
results.test_weighted_errors = NaN(1,K);

for k = 1:K
    
    fold_model = all_slim_coefficients{k};
    train_ind = k ~= folds;
    valid_ind = k == folds;
    
    results.train_errors(1,k) = get_error(X(train_ind,:), Y(train_ind,:), fold_model);
    results.train_pos_errors(1,k) = get_pos_error(X(train_ind,:), Y(train_ind,:), fold_model);
    results.train_neg_errors(1,k) = get_neg_error(X(train_ind,:), Y(train_ind,:), fold_model);
    
    results.valid_errors(1,k) = get_error(X(valid_ind,:), Y(valid_ind,:), fold_model);
    results.valid_pos_errors(1,k) = get_pos_error(X(valid_ind,:), Y(valid_ind,:), fold_model);
    results.valid_neg_errors(1,k) = get_neg_error(X(valid_ind,:), Y(valid_ind,:), fold_model);
    
    results.test_errors(1,k) = get_error(X_test, Y_test, fold_model); 
    results.test_pos_errors(1,k) = get_pos_error(X_test, Y_test, fold_model);
    results.test_neg_errors(1,k) = get_neg_error(X_test, Y_test, fold_model);

end

%weighted error metrics (used for model selection for R based models; not needed for SLIM)
results.full_train_weighted_errors = (w_pos * results.full_train_pos_errors + w_neg * results.full_train_neg_errors);
results.full_test_weighted_errors = (w_neg .* results.full_test_neg_errors) + (w_pos .* results.full_test_pos_errors);
results.train_weighted_errors = (w_neg .* results.train_neg_errors) + (w_pos .* results.train_pos_errors);
results.valid_weighted_errors = (w_neg .* results.valid_neg_errors) + (w_pos .* results.valid_pos_errors);
results.test_weighted_errors = (w_neg .* results.test_neg_errors) + (w_pos .* results.test_pos_errors);

%% Save Results
save(results_file_name, 'results')

