function [parameters, fval] = fit_bayesian_models(data, model)
% FIT_BAYESIAN_MODELS Unified function to fit Bayesian learning models 
% (with and without perseveration) to behavioral data.
%
% INPUTS:
%   data  - N x 3 cell array. 
%           Column 1: Subject ID.
%           Column 2: Data ordered by bandit pair.
%           Column 3: Data ordered as presented in the task (intermixed).
%   model - String specifying the model architecture to fit:
%           'ideal'      : Credibility fixed to [0.5, 0.75, 1].
%           'cred'       : Credibilities of 1- and 2-star agents are free parameters.
%           'ideal_pers' : Ideal credibilities + Perseveration and Forgetting.
%           'cred_pers'  : Free credibilities + Perseveration and Forgetting.
%
% OUTPUTS:
%   parameters - [N_subjects x N_parameters] matrix of best-fitting parameters.
%   fval       - [N_subjects x 1] array of negative log-likelihoods for the best fit.

    n_subj = size(data, 1);
    n_attempts_per_subj = 10; % Number of multi-start fitting attempts per subject
    n_attempts = n_attempts_per_subj * n_subj;
    
    % Retrieve bounds and randomized starting points
    [lb, ub, SP] = my_lbubsp(model, n_attempts); 
    
    % Optimization settings
    options = optimset('Display', 'off', 'FunValCheck', 'on');
    
    % Preallocate temporary optimization arrays
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
        curr_data = data{ss, 3};
        
        % Fit all attempts per subject in parallel using direct indexing
        parfor aa = 1:n_attempts_per_subj 
            [parameters_tmp(counter + aa, :), fval_tmp(counter + aa), exit_flag_tmp(counter + aa), ...
             output_tmp{counter + aa}, lambda_tmp{counter + aa}, gradient_tmp(counter + aa, :), hessian_tmp(counter + aa, :, :)] = ...
                fmincon(@(params) my_objective_bayesian(params, curr_data, model), ...
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
    ub_beta = 30; % Inverse temperature bound
    ub_pers = 5;  % Perseveration bound

    switch model
        case 'ideal'
            lb = 0;
            ub = ub_beta;
            sp = rand(nfits, 1) * ub_beta;
            
        case 'cred' % Free credibilities for 1- and 2-star agents
            lb = zeros(1, 3);
            ub = [ub_beta, ones(1, 2)];
            sp = [ub_beta * rand(nfits, 1), rand(nfits, 2)]; 
            
        case 'ideal_pers'
            lb = [0, -ub_pers, 0];
            ub = [ub_beta, ub_pers, 1];
            sp = [ub_beta * rand(nfits, 1), (rand(nfits, 1) - 0.5) * 2 * ub_pers, rand(nfits, 1)]; 
            
        case 'cred_pers'
            lb = [zeros(1, 3), -ub_pers, 0];
            ub = [ub_beta, ones(1, 2), ub_pers, 1];
            sp = [ub_beta * rand(nfits, 1), rand(nfits, 2), ...
                 (rand(nfits, 1) - 0.5) * 2 * ub_pers, rand(nfits, 1)]; 
    end
end

function mll = my_objective_bayesian(parameters, data, model)
% Calculates the negative log-likelihood based on a Bayesian grid approximation.
    dbstop if error
    
    % Unpack parameters (PERS and fP are set to 0 for non-pers models)
    [beta, C, PERS, fP] = my_param_unpacker(parameters, model);

    % Extract task variables
    n_block        = size(data.block, 1);
    n_trials_block = size(data.block, 2);
    n_trials       = n_block * n_trials_block;
    
    chosen   = data.pick';
    agent    = data.agent';
    feedback = data.feedback';
    game     = mod(data.game' - 1, 3) + 1;
    
    if isfield(data, 'response_time')
        RT = data.response_time';
    else
        RT = inf * ones(size(chosen));
    end

    % Initialization for Bayesian integration
    counter = 0;
    loglik_choice = nan(1, n_trials);
    dx = 0.01; % Step size for the probability grid
    p = (0:dx:1)'; 
    uniform = ones(3, 2, length(p)) / (dx * length(p)); % Uniform priors
    value = zeros(1, 2);

    for nn = 1:n_trials
        if mod(nn, n_trials_block) == 1 % Reset on each block
            g_p = uniform; % Prior density grid
            P   = zeros(3, 2); % Perseveration trace
        end
        
        if ismember(chosen(nn), 1:2)
            if RT(nn) > 150 % Filter out implausibly fast reaction times
                counter = counter + 1;

                % 1. Calculate Expected Value from Posteriors (Integration)
                value(1) = p' * squeeze(g_p(game(nn), 1, :)) * dx;
                value(2) = p' * squeeze(g_p(game(nn), 2, :)) * dx;
        
                % 2. Choice Probability (Softmax with log-sum-exp trick)
                in_exp = beta * value + P(game(nn), :);
                in_exp = in_exp - max(in_exp);
                loglik_choice(counter) = in_exp(chosen(nn)) - log(sum(exp(in_exp)));
            end
            
            % 3. Bayesian Belief Updating
            % Update grid based on feedback (1 = reward, 0 = no reward)
            curr_prior = squeeze(g_p(game(nn), chosen(nn), :));
            curr_cred  = C(agent(nn));
            
            if feedback(nn) == 1
                g_p(game(nn), chosen(nn), :) = curr_prior .* (curr_cred .* p + (1 - curr_cred) .* (1 - p));
            elseif feedback(nn) == 0
                g_p(game(nn), chosen(nn), :) = curr_prior .* (curr_cred .* (1 - p) + (1 - curr_cred) .* p);
            end

            % Normalize the updated posterior distribution
            g_p(game(nn), chosen(nn), :) = squeeze(g_p(game(nn), chosen(nn), :)) / ...
                                           sum(dx * squeeze(g_p(game(nn), chosen(nn), :)));

            % 4. Update Perseveration Trace
            P = (1 - fP) * P;
            P(game(nn), chosen(nn)) = P(game(nn), chosen(nn)) + PERS;
        end
    end

    mll = -sum(loglik_choice(1:counter));
end

function [beta, C, PERS, fP] = my_param_unpacker(parameters, model)
% Formats optimized parameter vector. Non-pers models map PERS and fP to 0.
    parameters = squeeze(parameters);
    
    switch model
        case 'ideal'
            beta = parameters(1);
            C    = [0.5, 0.75, 1];
            PERS = 0;
            fP   = 0;
            
        case 'cred'
            beta = parameters(1);
            C    = [parameters(2:3), 1]; % 3rd agent credibility fixed to 1
            PERS = 0;
            fP   = 0;
            
        case 'ideal_pers'
            beta = parameters(1);
            C    = [0.5, 0.75, 1];
            PERS = parameters(2);
            fP   = parameters(3);
            
        case 'cred_pers'
            beta = parameters(1);
            C    = [parameters(2:3), 1]; % 3rd agent credibility fixed to 1
            PERS = parameters(4);
            fP   = parameters(5);
    end
end