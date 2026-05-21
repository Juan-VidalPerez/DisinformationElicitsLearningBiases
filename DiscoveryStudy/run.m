% cross_fit_baysim_pilot.m
% Cross-fits the Bayesian simulated datasets using standard RL/CA models
% and restructures the output for parameter recovery analysis.

% =========================================================================
% 1. Load Simulated Data
% =========================================================================


% Calculate dimensions
n_total_sims = size(sim_data_bay_ideal, 1);
n_reps_pilot = 5; % Matches n_sim_pp in simulate_bayesian_models
n_subj_pilot = n_total_sims / n_reps_pilot; 

% =========================================================================
% 2. Fit Models to 'Ideal' Bayesian Simulations
% =========================================================================
disp('Fitting Standard Models to Ideal Bayesian Simulations...');
[p_id_cred, ev_id_cred]       = fit_models(sim_data_bay_ideal, 'cred');
[p_id_credval, ev_id_credval] = fit_models(sim_data_bay_ideal, 'cred-val');
[p_id_truth, ev_id_truth]     = fit_models(sim_data_bay_ideal, 'truth');

% =========================================================================
% 3. Fit Models to 'Credibility' Bayesian Simulations
% =========================================================================
disp('Fitting Standard Models to Credibility Bayesian Simulations...');
[p_cr_cred, ev_cr_cred]       = fit_models(sim_data_bay_cred, 'cred');
[p_cr_credval, ev_cr_credval] = fit_models(sim_data_bay_cred, 'cred-val');
[p_cr_truth, ev_cr_truth]     = fit_models(sim_data_bay_cred, 'truth');

% =========================================================================
% 4. Restructure Parameters (3D) and Squeeze Evals (2D)
% =========================================================================
disp('Restructuring arrays...');
% Helper function: reshapes to (Subj x Reps x Params), then permutes to (Subj x Params x Reps)
make_3d = @(x) permute(reshape(x, n_subj_pilot, n_reps_pilot, size(x, 2)), [1, 3, 2]);

% Restructure Parameters to [N_subj x N_params x N_reps]
params_simIdeal_fitCred    = make_3d(p_id_cred);
params_simIdeal_fitCredval = make_3d(p_id_credval);
params_simIdeal_fitTruth   = make_3d(p_id_truth);

params_simCred_fitCred     = make_3d(p_cr_cred);
params_simCred_fitCredval  = make_3d(p_cr_credval);
params_simCred_fitTruth    = make_3d(p_cr_truth);

% Squeeze Evals to [N_subj x N_reps]
fval_simIdeal_fitCred      = squeeze(make_3d(ev_id_cred));
fval_simIdeal_fitCredval   = squeeze(make_3d(ev_id_credval));
fval_simIdeal_fitTruth     = squeeze(make_3d(ev_id_truth));

fval_simCred_fitCred       = squeeze(make_3d(ev_cr_cred));
fval_simCred_fitCredval    = squeeze(make_3d(ev_cr_credval));
fval_simCred_fitTruth      = squeeze(make_3d(ev_cr_truth));

% =========================================================================
% 5. Save Output
% =========================================================================
disp('Saving parameters_baysim_pilot.mat...');
save('parameters_baysim_pilot.mat', ...
    'params_simIdeal_fitCred', 'params_simIdeal_fitCredval', 'params_simIdeal_fitTruth', ...
    'params_simCred_fitCred', 'params_simCred_fitCredval', 'params_simCred_fitTruth', ...
    'fval_simIdeal_fitCred', 'fval_simIdeal_fitCredval', 'fval_simIdeal_fitTruth', ...
    'fval_simCred_fitCred', 'fval_simCred_fitCredval', 'fval_simCred_fitTruth');

disp('Cross-fitting complete!');