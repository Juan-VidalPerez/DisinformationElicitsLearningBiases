function [parameters, fval] = fit_models(data, model)
% FIT_MODELS Unified function to fit Reinforcement Learning Credit Assignment 
% (CA) and Truth Inference models to behavioral data.
%
% INPUTS:
%   data  - N x 3 cell array. 
%           Column 1: Subject ID.
%           Column 2: Data ordered by bandit pair.
%           Column 3: Data ordered as presented in the task (intermixed).
%   model - String specifying the model architecture to fit:
%           'null'     : Single CA parameter across all agents.
%           'cred'     : Dedicated free CA parameters for each agent.
%           'val'      : Dedicated free CA parameters per agent, single valence bias.
%           'cred-val' : Dedicated CA parameters for each agent AND feedback valence.
%           'truth'    : Dedicated CA parameters + Bayesian Truth Bias (TB) across agents.
%
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
        curr_data = data{ss, 3}; % Extract time-ordered task data
        
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
    ub_tb   = 5;  % Truth bias bound

    switch model
        case 'null' % 1 CA, 1 PERS, 2 Forgetting rates (fQ, fP)
            lb = [-ub_ca, -ub_pers, zeros(1, 2)];
            ub = [ ub_ca,  ub_pers, ones(1, 2)];
            sp = [(rand(nfits, 1) - 0.5) * 2 * ub_ca, ...
                  (rand(nfits, 1) - 0.5) * 2 * ub_pers, rand(nfits, 2)];
            
        case 'cred' % 3 CA, 1 PERS, 2 Forgetting rates
            lb = [-ub_ca * ones(1, 3), -ub_pers, zeros(1, 2)];
            ub = [ ub_ca * ones(1, 3),  ub_pers, ones(1, 2)];
            sp = [(rand(nfits, 3) - 0.5) * 2 * ub_ca, ...
                  (rand(nfits, 1) - 0.5) * 2 * ub_pers, rand(nfits, 2)];
            
        case 'val' % 3 CA, 1 VB, 1 PERS, 2 Forgetting rates
            lb = [-ub_ca * ones(1, 3), -ub_vb, -ub_pers, zeros(1, 2)];
            ub = [ ub_ca * ones(1, 3),  ub_vb,  ub_pers, ones(1, 2)];
            sp = [(rand(nfits, 3) - 0.5) * 2 * ub_ca, ...
                  (rand(nfits, 1) - 0.5) * 2 * ub_vb, ...
                  (rand(nfits, 1) - 0.5) * 2 * ub_pers, rand(nfits, 2)];
                  
        case 'cred-val' % 6 CA, 1 PERS, 2 Forgetting rates
            lb = [-ub_ca * ones(1, 6), -ub_pers, zeros(1, 2)];
            ub = [ ub_ca * ones(1, 6),  ub_pers, ones(1, 2)];
            sp = [(rand(nfits, 6) - 0.5) * 2 * ub_ca, ...
                  (rand(nfits, 1) - 0.5) * 2 * ub_pers, rand(nfits, 2)]; 
                  
        case 'truth' % 3 CA, 1 TB, 1 PERS, 2 Forgetting rates
            lb = [-ub_ca * ones(1, 3), -ub_tb, -ub_pers, zeros(1, 2)];
            ub = [ ub_ca * ones(1, 3),  ub_tb,  ub_pers, ones(1, 2)];
            sp = [(rand(nfits, 3) - 0.5) * 2 * ub_ca, ...
                  (rand(nfits, 1) - 0.5) * 2 * ub_tb, ...
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
    game        = mod(data.game' - 1, 3) + 1;
    
    % Credibility is extracted; used actively only if TB != 0
    if isfield(data, 'rel')
        credibility = data.rel';
    else
        credibility = ones(size(chosen)); % Fallback if missing
    end
    
    if isfield(data, 'response_time')
        RT = data.response_time';
    else
        RT = inf * ones(size(chosen));
    end

    % Prior reward probabilities for the bandit options
    prew = [0.25, 0.75];

    counter = 0;
    loglik_choice = nan(1, n_trials);

    % Loop through trials
    for nn = 1:n_trials
        if mod(nn, n_trials_block) == 1 % Reset Q and P at block start
            Q = zeros(3, 2); 
            P = zeros(3, 2); 
        end
        
        if ismember(chosen(nn), 1:2) && ~isnan(feedback(nn)) % Ignore timeouts
            if RT(nn) > 150 % Filter out implausibly fast reaction times (<150ms)
                counter = counter + 1;
                
                % Softmax choice probability with log-sum-exp trick
                in_exp = Q(game(nn), :) + P(game(nn), :);
                in_exp = in_exp - max(in_exp); 
                loglik_choice(counter) = in_exp(chosen(nn)) - log(sum(exp(in_exp)));
            end
            
            % 1. Apply Forgetting
            Q = (1 - fQ) * Q;
            
            % 2. Calculate Bayesian posterior probability of feedback being true (Ptruth)
            curr_prew = prew(chosen(nn));
            curr_cred = credibility(nn);
            
            if feedback(nn) == 1
                Ptruth = (curr_cred * curr_prew) / ...
                         (curr_cred * curr_prew + (1 - curr_cred) * (1 - curr_prew));
            elseif feedback(nn) == -1
                Ptruth = (curr_cred * (1 - curr_prew)) / ...
                         (curr_cred * (1 - curr_prew) + (1 - curr_cred) * curr_prew);
            end

            % 3. Unified Update for Action Values (Q)
            % If model is not 'truth', TB is [0 0 0], reducing this strictly 
            % to the standard Credit Assignment equations.
            update_val = (CA(1, agent(nn)) * (feedback(nn) == -1) + ...
                          CA(2, agent(nn)) * (feedback(nn) ==  1)) + ...
                          TB(agent(nn)) * (Ptruth - 0.5);
                          
            Q(game(nn), chosen(nn)) = Q(game(nn), chosen(nn)) + update_val * feedback(nn);
            
            % 4. Update Perseveration Traces (P)
            P = (1 - fP) * P; 
            P(game(nn), chosen(nn)) = P(game(nn), chosen(nn)) + PERS;
        end
    end

    % Return negative sum of log-likelihoods
    mll = -sum(loglik_choice(1:counter));
end

function [CA, TB, PERS, fQ, fP] = my_param_unpacker(parameters, model)
% Formats optimized parameter vector. For non-truth models, TB is explicitly 
% returned as [0, 0, 0] to silence it in the objective function.
% CA output is a 2x3 matrix: Row 1 = CA-, Row 2 = CA+, Columns = Agents.
    parameters = squeeze(parameters);

    switch model
        case 'null'
            CA   = [parameters(1) * ones(1, 3); parameters(1) * ones(1, 3)];
            TB   = [0, 0, 0];
            PERS = parameters(2);
            fQ   = parameters(3);
            fP   = parameters(4); 
            
        case 'cred'
            CA   = [parameters(1:3); parameters(1:3)];
            TB   = [0, 0, 0];
            PERS = parameters(4);
            fQ   = parameters(5); 
            fP   = parameters(6);
            
        case 'val'
            CA   = [parameters(1:3); parameters(1:3) + parameters(4)];
            TB   = [0, 0, 0];
            PERS = parameters(5);
            fQ   = parameters(6); 
            fP   = parameters(7); 
            
        case 'cred-val'
            CA   = [parameters(1:3); parameters(4:6)];
            TB   = [0, 0, 0];
            PERS = parameters(7);
            fQ   = parameters(8); 
            fP   = parameters(9); 
            
        case 'truth'
            CA   = [parameters(1:3); parameters(1:3)];
            TB   = parameters(4) * ones(1, 3);
            PERS = parameters(5);
            fQ   = parameters(6); 
            fP   = parameters(7); 
    end
end