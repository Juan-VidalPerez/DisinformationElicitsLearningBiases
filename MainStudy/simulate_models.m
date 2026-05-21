function [simulated_data] = simulate_models(parameters, model)
% SIMULATE_MODELS Unified function to simulate behavioral data for 
% Reinforcement Learning Credit Assignment (CA) and Truth Inference models.
%
% INPUTS:
%   parameters - [N_subjects x N_parameters] matrix with parameters used for 
%                simulations (same structure as output of the model fits).
%   model      - String specifying the model architecture:: 'null', 'cred', 'cred_noPERS', 'val', 'cred-val'
%
% OUTPUTS:
%   simulated_data - N x 3 cell array ordered exactly as the empirical data,
%                    allowing the same downstream analysis functions to be used.
    
    % Define multiple simulations per participant if needed (currently 1x)
    n_sim_pp = 5; 
    n_subj_original = size(parameters, 1);
    parameters = repmat(parameters, [n_sim_pp, 1]);
    

    % Initialize data dimensions for simulation
    n_trials_game  = 15; % number of trials with each bandit pair
    n_games_block  = 3;  % number of bandit pairs per block
    n_blocks       = 8;
    n_trials_block = n_trials_game * n_games_block;
    n_trials       = n_trials_block * n_blocks;
    
    p_rew = [0.25, 0.75]; % reward probabilities for bandits in a pair
    cred  = [0.5, 0.75, 1]; % agent credibilities
    n_agents = length(cred);
    
    % Pre-generate agents and games to be shuffled per block
    agents_base = repmat(repmat(1:n_agents, [1, n_trials_game/n_agents]), [1, n_games_block]);
    game_base   = [1*ones(1, n_trials_game), 2*ones(1, n_trials_game), 3*ones(1, n_trials_game)];
    
    % Preallocate output
    n_subj = size(parameters, 1);
    simulated_data = cell(n_subj, 3);
    
    for ss = 1:n_subj
        simulated_data{ss, 1} = mod(ss,n_subj_original)+1; % Subject ID
        
        % Unpack parameters (TB will be [0 0 0] for non-truth models)
        [CA, PERS, fQ, fP] = my_param_unpacker(parameters(ss, :), model);
        
        for bb = 1:n_blocks
            Q = zeros(n_games_block, 2); % Action values
            P = zeros(n_games_block, 2); % Perseveration traces
            
            % Shuffle agents and games for the current block
            [rand_agents, rand_game] = shuffle(agents_base, game_base); 
            
            % Initialize trial arrays for the block
            chosen   = nan(1, n_trials_block);
            reward   = nan(1, n_trials_block);
            feedback = nan(1, n_trials_block);
            
            for tt = 1:n_trials_block
                
                % 1. Action Selection (Softmax with Perseveration)
                prob_a = 1 / (1 + exp(diff(Q(rand_game(tt), :)) + diff(P(rand_game(tt), :))));
                chosen(tt) = 1 + (rand > prob_a);
                
                % 2. Generate Reward and Feedback
                % Determine true underlying reward (mapped to [-1, 1])
                reward(tt) = 2 * (double(rand < p_rew(chosen(tt))) - 0.5);
                
                % Determine if agent lies based on their credibility
                lie = rand;
                feedback(tt) = (lie < cred(rand_agents(tt))) * reward(tt) + ...
                               (lie > cred(rand_agents(tt))) * (-reward(tt));
                
                % 3. Calculate Bayesian Posterior (Ptruth)
                curr_cred = cred(rand_agents(tt));
                curr_prew = p_rew(chosen(tt));
                

                
                % 4. Update Q-values and Perseveration (Forgetting applied first)
                Q = (1 - fQ) * Q;
                
                % Unified update: For non-truth models, TB is zero, silencing the Ptruth term
                update_val = (CA(1, rand_agents(tt)) * (feedback(tt) == -1) + ...
                              CA(2, rand_agents(tt)) * (feedback(tt) ==  1));
                              
                Q(rand_game(tt), chosen(tt)) = Q(rand_game(tt), chosen(tt)) + update_val * feedback(tt);
                
                P = (1 - fP) * P;
                P(rand_game(tt), chosen(tt)) = P(rand_game(tt), chosen(tt)) + PERS;
            end
            
            %% Store block data in Column 3 (Intermixed task format)
            simulated_data{ss, 3}.block(bb, :)         = bb * ones(1, n_trials_block);
            simulated_data{ss, 3}.trial(bb, :)         = 1:n_trials_block;
            simulated_data{ss, 3}.agent(bb, :)         = rand_agents; 
            simulated_data{ss, 3}.rel(bb, :)           = cred(rand_agents);  
            simulated_data{ss, 3}.pick(bb, :)          = chosen;
            simulated_data{ss, 3}.accuracy(bb, :)      = (chosen == 2);
            simulated_data{ss, 3}.reward(bb, :)        = (reward + 1) / 2;     % Map back to [0, 1]
            simulated_data{ss, 3}.feedback(bb, :)      = (feedback + 1) / 2;   % Map back to [0, 1]
            simulated_data{ss, 3}.game(bb, :)          = rand_game + 3 * (bb - 1);
            simulated_data{ss, 3}.response_time(bb, :) = inf(1, n_trials_block);
            simulated_data{ss, 3}.timeouts(bb, :)      = zeros(1, n_trials_block);
            
            %% Store block data in Column 2 (Organized per bandit pair)
            for gg = 1:3
                idx = find(rand_game == gg);
                target_row = 3 * (bb - 1) + gg;
                
                simulated_data{ss, 2}.block(target_row, :)    = bb * ones(1, n_trials_game);
                simulated_data{ss, 2}.trial(target_row, :)    = 1:n_trials_game;
                simulated_data{ss, 2}.game(target_row, :)     = rand_game(idx) + 3 * (bb - 1);
                simulated_data{ss, 2}.agent(target_row, :)    = rand_agents(idx);
                simulated_data{ss, 2}.rel(target_row, :)      = cred(rand_agents(idx));
                simulated_data{ss, 2}.pick(target_row, :)     = chosen(idx);
                simulated_data{ss, 2}.accuracy(target_row, :) = (chosen(idx) == 2);
                simulated_data{ss, 2}.reward(target_row, :)   = (reward(idx) + 1) / 2;
                simulated_data{ss, 2}.feedback(target_row, :) = (feedback(idx) + 1) / 2;
                
                % Inter-Trial Interval (number of intervening trials)
                simulated_data{ss, 2}.ITI(target_row, :)      = [idx(2:end) - idx(1:end-1), nan]; % Padded length appropriately if needed, based on original
                simulated_data{ss, 2}.Q(target_row, :)        = Q(gg, :);
            end
        end
        % Get previous game and agent mapping
        [simulated_data{ss, 2}.prev_game, simulated_data{ss, 2}.prev_agent] = get_previous(simulated_data{ss, 3});
    end
end

% =========================================================================
% SUBFUNCTIONS
% =========================================================================

function [CA, PERS, fQ, fP] = my_param_unpacker(parameters, model)
% Unpacks parameters based on model type. For non-truth models, TB is [0, 0, 0].
    parameters = squeeze(parameters);
    
    switch model
        % --- Standard CA Models ---
        case 'null' 
            CA   = [parameters(1) * ones(1, 3); parameters(1) * ones(1, 3)];
            PERS = parameters(2);
            fQ   = parameters(3);
            fP   = parameters(4); 
        case 'cred'
            CA   = [parameters(1:3); parameters(1:3)];
            PERS = parameters(4);
            fQ   = parameters(5); 
            fP   = parameters(6);
        case 'cred_noPERS'
            CA   = [parameters(1:3); parameters(1:3)];
            PERS = 0;
            fQ   = parameters(4); 
            fP   = 0;
        case 'val'
            CA   = [parameters(1:3); parameters(1:3) + parameters(4)];
            PERS = parameters(5);
            fQ   = parameters(6); 
            fP   = parameters(7); 
        case 'cred-val'
            CA   = [parameters(1:3); parameters(4:6)];
            PERS = parameters(7);
            fQ   = parameters(8); 
            fP   = parameters(9); 
            
        
    end
end

function [agent_out, game_out] = shuffle(agent_in, game_in)
% Randomly permutes the agent and game arrays synchronously
    A = [agent_in; game_in];
    A = A(:, randperm(size(A, 2)));
    agent_out = A(1, :);
    game_out  = A(2, :);
end

function [games, agent] = get_previous(data)
% Extracts the previous game and agent matrices for sequential effects analysis
    max_game = max(data.game(:));
    games = nan(max_game, size(data.game, 2)); % Preallocate
    agent = nan(max_game, size(data.game, 2)); % Preallocate
    
    counter = ones(1, max_game);
    
    for bb = 1:size(data.game, 1)
        for tt = 1:size(data.game, 2)
            curr_game = data.game(bb, tt);
            curr_count = counter(curr_game);
            
            if curr_count == 1
                games(curr_game, curr_count) = nan;
                agent(curr_game, curr_count) = nan;
            else
                games(curr_game, curr_count) = data.game(bb, tt - 1);
                agent(curr_game, curr_count) = data.agent(bb, tt - 1);
            end
            counter(curr_game) = curr_count + 1;
        end
    end
end