function [simulated_data] = simulate_models(parameters, model)
% SIMULATE_MODELS Generates synthetic trial-by-trial data for the
% pilot study using fitted computational parameters (4 Agents, Blocked Design).
%
% INPUTS:
%   model      - String specifying the architecture ('null', 'cred', 'cred-val').
%   parameters - [N_subjects x N_parameters] matrix of fitted subject parameters.
%
% OUTPUTS:
%   simulated_data - N_subjects x 2 cell array matching the pilot data format.
    % Define multiple simulations per participant if needed (currently 1x)
    n_sim_pp = 5; 
    n_subj_original = size(parameters, 1);
    parameters = repmat(parameters, [n_sim_pp, 1]);

    % Task structure configuration
    n_trials_game  = 16;
    n_games_block  = 1; % 1 game per block = strictly blocked, not interleaved
    n_blocks       = 15;
    n_trials_block = n_trials_game * n_games_block;
    
    % Block conditions: 1 = Informative (Bandit choice), 0 = Control (Uniform probabilities)
    condition = [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0];
    
    % 4 Agents with objective credibilities (0.5 to 1.0)
    ag_rel   = [0.5, 0.7, 0.85, 1];
    n_agents = length(ag_rel);

    % Define a base array of agents ensuring equal distribution per block
    base_agents = repmat(1:n_agents, [1, n_trials_game / n_agents]);

    % Unpack parameter matrix based on model type string
    [CA, PERS, fQ, fP] = my_param_generator(model, parameters);

    n_subj         = size(parameters, 1);
    simulated_data = cell(n_subj, 2);

    % Loop through each subject
    for ss = 1:n_subj

        simulated_data{ss, 1} = mod(ss,n_subj_original)+1;
        simulated_data{ss, 2}.parameters = parameters(ss, :);

        % Loop through each experimental block
        for block = 1:n_blocks
            % Reset value (Q) and perseveration (P) traces at the start of each block
            Q = zeros(1, 2);
            P = zeros(1, 2);
            
            % Randomize agent presentation order for the current block
            rand_agents = base_agents(:, randperm(length(base_agents)));
            
            % Set bandit reward probabilities based on block condition
            if condition(block) == 1
                p_rew = [0.25, 0.75]; % Informative block
            else
                p_rew = repmat(rand * 0.2 + 0.6, [1, 2]); % Control block (equal probs)
            end
            
            % Preallocate trial-specific arrays for the block
            a = nan(1, n_trials_block);
            r = nan(1, n_trials_block);
            f = nan(1, n_trials_block);
            
            % Loop through block trials
            for tt = 1:n_trials_block
        
                % 1. Softmax choice rule (using difference trick)
                prob_a = 1 / (1 + exp(diff(Q) + diff(P)));
                a(tt)  = 1 + (rand > prob_a); % Chosen option: 1 or 2
                
                % 2. Generate objective reward outcome rescaled to [-1, 1]
                r(tt) = 2 * (double(rand < p_rew(a(tt))) - 0.5);
                
                % 3. Generate agent feedback (dependent on agent credibility)
                lie = rand;
                if lie < ag_rel(rand_agents(tt))
                    f(tt) = r(tt);     % True feedback
                else
                    f(tt) = -r(tt);    % False feedback / Deception
                end
               
                % 4. Update Q-values with forgetting rate
                Q = (1 - fQ(ss)) * Q;
                
                % Separate Credit Assignment updates for valence (CA- vs CA+)
                if f(tt) < 0
                    % Update using negative feedback parameters (columns 1:4)
                    Q(a(tt)) = Q(a(tt)) + CA(ss, rand_agents(tt)) * f(tt);
                else
                    % Update using positive feedback parameters (columns 5:8)
                    Q(a(tt)) = Q(a(tt)) + CA(ss, 4 + rand_agents(tt)) * f(tt);
                end
                
                % 5. Update Perseveration traces with forgetting rate
                P = (1 - fP(ss)) * P;
                P(a(tt)) = P(a(tt)) + PERS(ss);
            end
            
            % =============================================================
            % Store simulated data structures (Blocked trial alignment)
            % =============================================================
            simulated_data{ss, 2}.block(block, :)         = block * ones(1, n_trials_block);
            simulated_data{ss, 2}.condition(block, :)     = condition(block) * ones(1, n_trials_block);
            simulated_data{ss, 2}.trial(block, :)         = 1:n_trials_block;
            simulated_data{ss, 2}.agent(block, :)         = rand_agents; 
            simulated_data{ss, 2}.rel(block, :)           = ag_rel(rand_agents);  
            simulated_data{ss, 2}.pick(block, :)          = a;
            simulated_data{ss, 2}.reward(block, :)        = (r + 1) / 2; % Rescale back to [0, 1]
            simulated_data{ss, 2}.feedback(block, :)      = (f + 1) / 2; % Rescale back to [0, 1]
            simulated_data{ss, 2}.response_time(block, :) = inf(1, n_trials_block);
            simulated_data{ss, 2}.timeouts(block, :)      = zeros(1, n_trials_block);
            simulated_data{ss,2}.Prew(block, :, 1)= p_rew(1) * ones(1, n_trials_block);
            simulated_data{ss,2}.Prew(block, :, 2)= p_rew(2) * ones(1, n_trials_block);
            
            % Accuracy coding: true choice accuracy if informative, 0.5 baseline if control
            simulated_data{ss, 2}.accuracy(block, :)      = (a == 2) * (condition(block) == 1) + ...
                                                            0.5 * (condition(block) == 0);
        end
    end
end

% =========================================================================
% SUBFUNCTIONS
% =========================================================================

function [CA, PERS, fQ, fP] = my_param_generator(model, parameters)
% Generates expanded parameter vectors for the 4-agent task configuration.
% Matrix dimensions mapping:
%   CA output is structured as: [Subjects x 8 Parameters] 
%   Cols 1-4 = CA- (Agents 1-4), Cols 5-8 = CA+ (Agents 1-4)

    switch model
        case 'null' % Null Model (1 CA, 1 PERS, 2 Forgetting rates)
            CA   = parameters(:, 1) * ones(1, 8);
            PERS = parameters(:, 2);
            fQ   = parameters(:, 3); 
            fP   = parameters(:, 4); 
            
        case 'cred' % Credibility Model (4 CAs pooled across valences, 1 PERS, 2 Forgetting)
            CA   = [parameters(:, 1:4), parameters(:, 1:4)];
            PERS = parameters(:, 5);
            fQ   = parameters(:, 6); 
            fP   = parameters(:, 7); 
            
        case 'cred-val' % Credibility-Valence Model (4 CA-, 4 CA+, 1 PERS, 2 Forgetting)
            CA   = [parameters(:, 1:4), parameters(:, 5:8)];
            PERS = parameters(:, 9);
            fQ   = parameters(:, 10); 
            fP   = parameters(:, 11); 
    end
end