function [simulated_data] = simulate_bayesian_models(parameters, model)
% SIMULATE_BAYESIAN_MODELS Generates synthetic trial-by-trial data for the
% pilot study using Bayesian learning models (4 Agents, Blocked Design).
%
% INPUTS:
%   parameters - Matrix with parameters used for simulations. Should match the 
%                structure of the output from the bayesian model fits.
%   model      - String specifying the model architecture:
%                'ideal'      : Fixed ideal credibilities [0.5, 0.7, 0.85, 1].
%                'cred'       : Free credibilities for agents 1, 2, and 3.
%                'ideal_pers' : Ideal credibilities + Perseveration & Forgetting.
%                'cred_pers'  : Free credibilities + Perseveration & Forgetting.
%
% OUTPUTS:
%   simulated_data - N_subjects x 2 cell array matching the pilot data format.

    % Define multiple simulations per participant
    n_sim_pp = 5; % Set to 5 to match the 1020 rows in parameters_baysim
    n_subj_original = size(parameters, 1);
    parameters = repmat(parameters, [n_sim_pp, 1]);

    % Task structure configuration (Blocked Pilot design)
    n_trials_game  = 16; % Number of trials with each bandit pair
    n_games_block  = 1;  % Number of bandit pairs per block (strictly blocked)
    n_blocks       = 15;
    n_trials_block = n_trials_game * n_games_block;

    % Block conditions: 1 = Informative (Bandit choice), 0 = Control (Uniform probs)
    condition = [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0];
    
    % Pilot objective agent credibilities
    cred = [0.5, 0.7, 0.85, 1]; 
    n_agents = length(cred);
    
    % Base array of agents to guarantee equal agent distribution per block
    base_agents = repmat(1:n_agents, [1, n_trials_game / n_agents]);

    % Initialize integration grid for Bayesian posteriors
    dx = 0.01;
    p = (0:dx:1)';
    % Uniform prior density grid: 2 rows (bandits) x 101 cols (probabilities)
    uniform = ones(2, length(p)) / (dx * length(p)); 

    % Preallocate output matrix
    n_subj = size(parameters, 1);
    simulated_data = cell(n_subj, 2);

    % Run Simulations
    for ss = 1:n_subj
        % Subject ID mapping (handles the n_sim_pp replication cleanly)
        simulated_data{ss, 1} = mod(ss - 1, n_subj_original) + 1; 
        simulated_data{ss, 2}.parameters = parameters(ss, :);
        
        % Unpack parameters for the current subject
        [beta, sub_cred, PERS, fP] = my_param_unpacker(parameters(ss, :), model);
        
        % Preallocate structured tables for the subject
        simulated_data{ss, 2}.block         = nan(n_blocks, n_trials_block);
        simulated_data{ss, 2}.trial         = nan(n_blocks, n_trials_block);
        simulated_data{ss, 2}.pick          = nan(n_blocks, n_trials_block);
        simulated_data{ss, 2}.reward        = nan(n_blocks, n_trials_block);
        simulated_data{ss, 2}.agent         = nan(n_blocks, n_trials_block);
        simulated_data{ss, 2}.feedback      = nan(n_blocks, n_trials_block);
        simulated_data{ss, 2}.accuracy      = nan(n_blocks, n_trials_block);
        simulated_data{ss, 2}.condition     = nan(n_blocks, n_trials_block);
        simulated_data{ss, 2}.reliability   = nan(n_blocks, n_trials_block);
        simulated_data{ss, 2}.Prew          = nan(n_blocks, n_trials_block, 2);
        simulated_data{ss, 2}.response_time = inf(n_blocks, n_trials_block); % Required by fit_models
        simulated_data{ss, 2}.timeouts      = zeros(n_blocks, n_trials_block); % Required by fit_models
        
        for bb = 1:n_blocks
            g_p = uniform; % Reset prior density grid for the new block
            P   = zeros(1, 2); % Reset perseveration trace
            
            % Randomize agent presentation order for this block
            rand_agent = base_agents(randperm(length(base_agents)));
            
            % Set bandit reward probabilities based on block condition
            if condition(bb) == 1
                p_rew = [0.25, 0.75]; % Informative block
            else
                p_rew = repmat(rand * 0.2 + 0.6, [1, 2]); % Control block
            end
            
            % Local block variables
            chosen   = nan(1, n_trials_block);
            reward   = nan(1, n_trials_block);
            feedback = nan(1, n_trials_block);
            
            for tt = 1:n_trials_block
                % 1. Calculate Expected Value from Posteriors via grid integration
                % Transpose g_p row to multiply with column vector p
                value(1) = p' * g_p(1, :)' * dx;
                value(2) = p' * g_p(2, :)' * dx;
                
                % 2. Action Selection (Softmax with Expected Value and Perseveration)
                % diff(P) naturally yields P(2) - P(1)
                prob_a = 1 / (1 + exp(beta * (value(2) - value(1)) + diff(P)));
                chosen(tt) = 1 + (rand > prob_a); % Yields 1 or 2
                
                % 3. Generate objective reward outcome (1 = win, 0 = loss)
                reward(tt) = double(rand < p_rew(chosen(tt)));
                
                % 4. Generate agent feedback (Deception dependent on agent credibility)
                if rand > cred(rand_agent(tt))
                     feedback(tt) = double(~reward(tt)); % Agent lies
                else
                     feedback(tt) = reward(tt); % Agent tells truth
                end
                
                % 5. Bayesian Posterior Belief Updating
                curr_prior = g_p(chosen(tt), :);
                curr_cred  = sub_cred(rand_agent(tt));
                
                if feedback(tt) == 1 
                     g_p(chosen(tt), :) = curr_prior .* (curr_cred .* p' + (1 - curr_cred) .* (1 - p'));
                elseif feedback(tt) == 0
                     g_p(chosen(tt), :) = curr_prior .* (curr_cred .* (1 - p') + (1 - curr_cred) .* p');
                end
                
                % Normalize posterior grid
                g_p(chosen(tt), :) = g_p(chosen(tt), :) / sum(dx * g_p(chosen(tt), :));
                
                % 6. Update Perseveration trace
                P = (1 - fP) * P;
                P(chosen(tt)) = P(chosen(tt)) + PERS;
            end

            % =============================================================
            % Store structured simulation data directly into Column 2
            % =============================================================
            simulated_data{ss, 2}.block(bb, :)       = bb * ones(1, n_trials_block);
            simulated_data{ss, 2}.trial(bb, :)       = 1:n_trials_block;
            simulated_data{ss, 2}.pick(bb, :)        = chosen;
            simulated_data{ss, 2}.reward(bb, :)      = reward;
            simulated_data{ss, 2}.agent(bb, :)       = rand_agent;
            simulated_data{ss, 2}.feedback(bb, :)    = feedback;
            simulated_data{ss, 2}.condition(bb, :)   = condition(bb) * ones(1, n_trials_block);
            simulated_data{ss, 2}.reliability(bb, :) = cred(rand_agent);
            
            % Accuracy coding (objective for informative blocks, subjective 0.5 for control)
            simulated_data{ss, 2}.accuracy(bb, :)    = (chosen == 2) * (condition(bb) == 1) + ...
                                                       0.5 * (condition(bb) == 0);
                                                   
            % Store dynamic reward priors
            simulated_data{ss, 2}.Prew(bb, :, 1)     = p_rew(1) * ones(1, n_trials_block);
            simulated_data{ss, 2}.Prew(bb, :, 2)     = p_rew(2) * ones(1, n_trials_block);
        end
    end
end

% =========================================================================
% SUBFUNCTIONS
% =========================================================================

function [beta, sub_cred, PERS, fP] = my_param_unpacker(parameters, model)
% Formats optimized parameter vector for 4 agents. Non-pers models map PERS and fP to 0.
% 4th agent's subjective credibility is always fixed to 1.0.
    
    parameters = squeeze(parameters);

    if strcmp(model, 'ideal')
        beta     = parameters(1);
        sub_cred = [0.5, 0.7, 0.85, 1]; % Pilot objective credibilities
        PERS     = 0;
        fP       = 0;
        
    elseif strcmp(model, 'cred')
        beta     = parameters(1);
        sub_cred = [parameters(2:4), 1]; % Free parameters for agents 1-3
        PERS     = 0;
        fP       = 0;
        
    elseif strcmp(model, 'ideal_pers')
        beta     = parameters(1);
        sub_cred = [0.5, 0.7, 0.85, 1];
        PERS     = parameters(2);
        fP       = parameters(3);
        
    elseif strcmp(model, 'cred_pers')
        beta     = parameters(1);
        sub_cred = [parameters(2:4), 1]; % Free parameters for agents 1-3
        PERS     = parameters(4);
        fP       = parameters(5);
    end
end