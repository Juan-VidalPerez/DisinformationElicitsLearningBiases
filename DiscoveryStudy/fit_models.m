function [parameters, fval] = fit_models(data, model)
% FIT_MODELS Unified function to fit Reinforcement Learning Credit Assignment 
% (CA) and Truth Inference models to behavioral data (Pilot Study - 4 Agents).
%
% INPUTS:
%   data  - N x 2 cell array. 
%           Column 1: Subject ID.
%           Column 2: Blocked data (not interleaved).
%   model - String specifying the model architecture to fit:
%           'null'          : Single CA parameter across all agents.
%           'cred'          : Dedicated free CA parameters for each of 4 agents.
%           'val'           : Free CA parameters per agent, single valence bias.
%           'cred-val'      : Dedicated CA parameters for each agent AND feedback valence.
%           'truth'         : 4 CA + 1 global Bayesian Truth Bias (TB).

% OUTPUTS:
%   parameters - [N_subjects x N_parameters] matrix of best-fitting parameters.
%   fval       - [N_subjects x 1] array of negative log-likelihoods for the best fit.

    n_subj = size(data, 1);
    n_attempts_per_subj = 10; % Multi-start fitting attempts per subject
    n_attempts = n_attempts_per_subj * n_subj;
    
    % Retrieve bounds and randomized starting points
    [lb, ub, SP] = my_lbubsp(model, n_attempts); 
    
    % Optimization settings (FunValCheck ensures no NaNs/Infs disrupt the fit)
    options = optimset('Display', 'off', 'FunValCheck', 'on');
    
    % Preallocate temporary outputs for efficiency
    parameters_tmp = nan(n_attempts, size(SP, 2));
    fval_tmp       = nan(1, n_attempts);
    exit_flag_tmp  = nan(1, n_attempts);
    output_tmp     = cell(1, n_attempts);
    lambda_tmp     = cell(1, n_attempts);
    gradient_tmp   = nan(n_attempts, size(SP, 2));
    hessian_tmp    = nan(n_attempts, size(SP, 2), size(SP, 2));
    
    % Preallocate final outputs
    parameters = nan(n_subj, size(SP, 2));
    fval       = nan(n_subj, 1);
    
    % Fitting procedure
    counter = 0;
    for ss = 1:n_subj
        curr_data = data{ss, 2}; % Extract blocked task data from Column 2
        
        % Parallel computation for multi-start optimization
        parfor aa = 1:n_attempts_per_subj 
            [parameters_tmp(counter + aa, :), fval_tmp(counter + aa), exit_flag_tmp(counter + aa), ...
             output_tmp{counter + aa}, lambda_tmp{counter + aa}, gradient_tmp(counter + aa, :), hessian_tmp(counter + aa, :, :)] = ...
                fmincon(@(params) my_objective(params, curr_data, model), ...
                SP(counter + aa, :), [], [], [], [], lb, ub, [], options);
        end
        
        % Extract the best fit (minimum negative log-likelihood)
        counter = counter + n_attempts_per_subj;
        start_idx = counter - n_attempts_per_subj + 1;
        
        [fval(ss), attempt_idx] = max(-fval_tmp(start_idx:counter));
        parameters(ss, :) = parameters_tmp(start_idx + attempt_idx - 1, :);
    end
end

% =========================================================================
% SUBFUNCTIONS
% =========================================================================

function [lb, ub, sp] = my_lbubsp(model, nfits)
% Defines lower bounds (lb), upper bounds (ub), and starting points (sp).
    ub_pers = 5;  % Perseveration bound
    ub_ca   = 10; % Credit assignment bound
    ub_vb   = 5;  % Valence bias bound
    ub_tb   = 10; % Truth bias bound

    switch model
        % --- Baseline CA Models ---
        case 'null' 
            lb = [-ub_ca, -ub_pers, zeros(1, 2)];
            ub = [ ub_ca,  ub_pers, ones(1, 2)];
            sp = [(rand(nfits, 1) - 0.5) * 2 * ub_ca, ...
                  (rand(nfits, 1) - 0.5) * 2 * ub_pers, rand(nfits, 2)];
            
        case 'cred' 
            lb = [-ub_ca * ones(1, 4), -ub_pers, zeros(1, 2)];
            ub = [ ub_ca * ones(1, 4),  ub_pers, ones(1, 2)];
            sp = [(rand(nfits, 4) - 0.5) * 2 * ub_ca, ...
                  (rand(nfits, 1) - 0.5) * 2 * ub_pers, rand(nfits, 2)];
            
        case 'val' 
            lb = [-ub_ca * ones(1, 4), -ub_vb, -ub_pers, zeros(1, 2)];
            ub = [ ub_ca * ones(1, 4),  ub_vb,  ub_pers, ones(1, 2)];
            sp = [(rand(nfits, 4) - 0.5) * 2 * ub_ca, ...
                  (rand(nfits, 1) - 0.5) * 2 * ub_vb, ...
                  (rand(nfits, 1) - 0.5) * 2 * ub_pers, rand(nfits, 2)];
                  
        case 'cred-val' 
            lb = [-ub_ca * ones(1, 8), -ub_pers, zeros(1, 2)];
            ub = [ ub_ca * ones(1, 8),  ub_pers, ones(1, 2)];
            sp = [(rand(nfits, 8) - 0.5) * 2 * ub_ca, ...
                  (rand(nfits, 1) - 0.5) * 2 * ub_pers, rand(nfits, 2)]; 
                  
        % --- Truth Inference Models ---
        case 'truth'
            lb = [-ub_ca * ones(1, 4), -ub_tb, -ub_pers, zeros(1, 2)];
            ub = [ ub_ca * ones(1, 4),  ub_tb,  ub_pers, ones(1, 2)];
            sp = [(rand(nfits, 4) - 0.5) * 2 * ub_ca, (rand(nfits, 1) - 0.5) * 2 * ub_tb, ...
                  (rand(nfits, 1) - 0.5) * 2 * ub_pers, rand(nfits, 2)]; 
                  
       
    end
end

function mll = my_objective(parameters, data, model)
% Calculates the negative log-likelihood of the data given the parameters.
    dbstop if error
    
    % Unpack parameters (TB is set to 0 for non-truth models)
    [CA, TB, PERS, fQ, fP] = my_param_unpacker(parameters, model);

    % Extract task variables
    n_block        = size(data.block, 1);
    n_trials_block = size(data.block, 2);
    n_trials       = n_block * n_trials_block;
    
    chosen      = data.pick';
    feedback    = 2 * (data.feedback' - 0.5); % Rescale to [-1, 1]
    agent       = data.agent'; 
    game        = ones(size(chosen)); % Blocked design = 1 active game
    
    % Credibility / Reliability extraction
    if isfield(data, 'reliability')
        credibility = data.reliability';
    elseif isfield(data, 'rel')
        credibility = data.rel';
    else
        credibility = ones(size(chosen)); 
    end
    
    if isfield(data, 'response_time')
        RT = data.response_time';
    else
        RT = inf * ones(size(chosen));
    end

    % Bandit prior reward probabilities (support for trial-by-trial shifts in control blocks)
    prew{1} = data.Prew(:, :, 1)';
    prew{2} = data.Prew(:, :, 2)';


    counter = 0;
    loglik_choice = nan(1, n_trials);

    % Loop through trials
    for nn = 1:n_trials
        if mod(nn, n_trials_block) == 1 % Reset Q and P at block start
            Q = zeros(1, 2); 
            P = zeros(1, 2); 
        end
        
        if ismember(chosen(nn), 1:2) && ismember(feedback(nn), [-1, 1])
            if RT(nn) > 150 % Filter out implausibly fast reaction times (<150ms)
                counter = counter + 1;
                
                % Softmax choice probability with log-sum-exp trick
                in_exp = Q + P;
                in_exp = in_exp - max(in_exp); 
                loglik_choice(counter) = in_exp(chosen(nn)) - log(sum(exp(in_exp)));
            end
            
            % 1. Apply Forgetting
            Q = (1 - fQ) * Q;
            
            % 2. Calculate Bayesian posterior probability of feedback being true (Ptruth)
            curr_prew = prew{chosen(nn)}(nn);
            curr_cred = credibility(nn);
            
            if feedback(nn) == 1
                Ptruth = (curr_cred * curr_prew) / ...
                         (curr_cred * curr_prew + (1 - curr_cred) * (1 - curr_prew));
            elseif feedback(nn) == -1
                Ptruth = (curr_cred * (1 - curr_prew)) / ...
                         (curr_cred * (1 - curr_prew) + (1 - curr_cred) * curr_prew);
            end

            % 3. Unified Update for Action Values (Q)
            update_val = (CA(1, agent(nn)) * (feedback(nn) == -1) + ...
                          CA(2, agent(nn)) * (feedback(nn) ==  1)) + ...
                          TB(agent(nn)) * (Ptruth - 0.5);
                          
            Q(chosen(nn)) = Q(chosen(nn)) + update_val * feedback(nn);
            
            % 4. Update Perseveration Traces (P)
            P = (1 - fP) * P; 
            P(chosen(nn)) = P(chosen(nn)) + PERS;
        end
    end

    % Return negative sum of log-likelihoods
    mll = -sum(loglik_choice(1:counter));
end

function [CA, TB, PERS, fQ, fP] = my_param_unpacker(parameters, model)
% Formats optimized parameter vector. For non-truth models, TB is explicitly 
% returned as [0, 0, 0, 0] to silence it in the objective function.
    parameters = squeeze(parameters);

    switch model
        % --- Baseline CA Models ---
        case 'null'
            CA   = [parameters(1) * ones(1, 4); parameters(1) * ones(1, 4)];
            TB   = [0, 0, 0, 0];
            PERS = parameters(2);
            fQ   = parameters(3);
            fP   = parameters(4); 
            
        case 'cred'
            CA   = [parameters(1:4); parameters(1:4)];
            TB   = [0, 0, 0, 0];
            PERS = parameters(5);
            fQ   = parameters(6); 
            fP   = parameters(7);
            
        case 'val'
            CA   = [parameters(1:4); parameters(1:4) + parameters(5)];
            TB   = [0, 0, 0, 0];
            PERS = parameters(6);
            fQ   = parameters(7); 
            fP   = parameters(8); 
            
        case 'cred-val'
            CA   = [parameters(1:4); parameters(5:8)];
            TB   = [0, 0, 0, 0];
            PERS = parameters(9);
            fQ   = parameters(10); 
            fP   = parameters(11); 
            
        % --- Truth Inference Models ---
        case 'truth'
            CA   = [parameters(1:4); parameters(1:4)];
            TB   = parameters(5) * ones(1, 4);
            PERS = parameters(6);
            fQ   = parameters(7); 
            fP   = parameters(8); 

    end
end