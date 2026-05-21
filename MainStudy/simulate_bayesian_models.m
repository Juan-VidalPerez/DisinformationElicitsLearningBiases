function [simulated_data] = simulate_bayesian_models(parameters, model)
% SIMULATE_BAYESIAN_MODELS Unified function to simulate behavioral data for 
% Bayesian learning models (with and without perseveration).
%
% INPUTS:
%   parameters - Matrix with parameters used for simulations. Should match the 
%                structure of the output from the bayesian model fits.
%                Number of rows dictates the number of simulated subjects.
%   model      - String specifying the model architecture:
%                'ideal'      : Credibility fixed to [0.5, 0.75, 1].
%                'cred'       : Free credibilities for 1- and 2-star agents.
%                'ideal_pers' : Ideal credibilities + Perseveration & Forgetting.
%                'cred_pers'  : Free credibilities + Perseveration & Forgetting.
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

    p_rew = [0.25, 0.75]; % reward probabilities for bandits in a pair
    cred  = [0.5, 0.75, 1]; % objective agent credibilities
    n_agents = length(cred);
    
    % Pre-generate agents and games to be shuffled per block
    agents_base = repmat(repmat(1:n_agents, [1, n_trials_game/n_agents]), [1, n_games_block]);
    game_base   = [1*ones(1, n_trials_game), 2*ones(1, n_trials_game), 3*ones(1, n_trials_game)];

    % Initialize integration grid for Bayesian posteriors
    dx = 0.01;
    p = (0:dx:1)';
    uniform = ones(3, 2, length(p)) / (dx * length(p));

    % Preallocate output
    n_subj = size(parameters, 1);
    simulated_data = cell(n_subj, 3);
    

    % Run Simulations
    for ss = 1:n_subj
        simulated_data{ss, 1} = mod(ss,n_subj_original)+1; % Subject ID
        
        % Unpack parameters for the current subject
        [beta, sub_cred, PERS, fP] = my_param_unpacker(parameters(ss, :), model);
        
        for bb = 1:n_blocks
            g_p = uniform; % Prior of Prob(Prob_row) for each bandit
            P = zeros(n_games_block, 2); % Perseveration trace
            
            [rand_agent, rand_game] = shuffle(agents_base, game_base); % Shuffle arrays
            
            % Initialize block arrays
            chosen   = nan(1, n_trials_block);
            reward   = nan(1, n_trials_block);
            feedback = nan(1, n_trials_block);
            
            for tt = 1:n_trials_block
                % Calculate expected value of current bandits via grid integration
                value(1) = p' * squeeze(g_p(rand_game(tt), 1, :)) * dx;
                value(2) = p' * squeeze(g_p(rand_game(tt), 2, :)) * dx;
                
                % Action Selection (Softmax incorporating Expected Value and Perseveration)
                prob_a = 1 / (1 + exp(beta * (value(2) - value(1)) + diff(P(rand_game(tt), :))));
                chosen(tt) = 1 + (rand > prob_a);
                
                % Generate true reward and agent feedback
                reward(tt) = double(rand < p_rew(chosen(tt)));
                if rand > cred(rand_agent(tt))
                     feedback(tt) = double(~reward(tt)); % Agent lies
                else
                     feedback(tt) = reward(tt); % Agent tells truth
                end
                
                % Bayesian Posterior Updating
                curr_prior = squeeze(g_p(rand_game(tt), chosen(tt), :));
                curr_cred  = sub_cred(rand_agent(tt));
                
                if feedback(tt) == 1 
                     g_p(rand_game(tt), chosen(tt), :) = curr_prior .* (curr_cred .* p + (1 - curr_cred) .* (1 - p));
                elseif feedback(tt) == 0
                     g_p(rand_game(tt), chosen(tt), :) = curr_prior .* (curr_cred .* (1 - p) + (1 - curr_cred) .* p);
                end
                
                % Normalize posterior
                g_p(rand_game(tt), chosen(tt), :) = squeeze(g_p(rand_game(tt), chosen(tt), :)) / ...
                                                    sum(dx * squeeze(g_p(rand_game(tt), chosen(tt), :)));
                
                % Update Perseveration trace
                P = (1 - fP) * P;
                P(rand_game(tt), chosen(tt)) = P(rand_game(tt), chosen(tt)) + PERS;
            end

            %% Store data in Column 3 (Intermixed task format)
            simulated_data{ss, 3}.block(bb, :)        = bb * ones(1, n_trials_block);
            simulated_data{ss, 3}.trial(bb, :)        = 1:n_trials_block;
            simulated_data{ss, 3}.pick(bb, :)         = chosen;
            simulated_data{ss, 3}.game(bb, :)         = rand_game + 3 * (bb - 1);
            simulated_data{ss, 3}.reward(bb, :)       = reward;
            simulated_data{ss, 3}.agent(bb, :)        = rand_agent;
            simulated_data{ss, 3}.feedback(bb, :)     = feedback;
            simulated_data{ss, 3}.accuracy(bb, :)     = (chosen == 2);
            simulated_data{ss, 3}.rel(bb, :)          = cred(rand_agent); % true credibility

            %% Store data in Column 2 (Organized per bandit pair)
            for gg = 1:3
                idx = find(rand_game == gg);
                target_row = 3 * (bb - 1) + gg;
                
                simulated_data{ss, 2}.block(target_row, :)    = bb * ones(1, n_trials_game);
                simulated_data{ss, 2}.game(target_row, :)     = rand_game(idx) + 3 * (bb - 1);
                simulated_data{ss, 2}.trial(target_row, :)    = 1:n_trials_game;
                simulated_data{ss, 2}.agent(target_row, :)    = rand_agent(idx);
                simulated_data{ss, 2}.rel(target_row, :)      = cred(rand_agent(idx));
                simulated_data{ss, 2}.pick(target_row, :)     = chosen(idx);
                simulated_data{ss, 2}.accuracy(target_row, :) = (chosen(idx) == 2);
                simulated_data{ss, 2}.reward(target_row, :)   = reward(idx);
                simulated_data{ss, 2}.feedback(target_row, :) = feedback(idx);
                
                % Inter-Trial Interval handling
                simulated_data{ss, 2}.ITI(target_row, :) = [idx(2:end) - idx(1:end-1), nan];
            end
        end
        % Get sequential effect variables
        [simulated_data{ss, 2}.prev_game, simulated_data{ss, 2}.prev_agent] = get_previous(simulated_data{ss, 3});
    end
end

% =========================================================================
% SUBFUNCTIONS
% =========================================================================

function [beta, sub_cred, PERS, fP] = my_param_unpacker(params, model)
% Formats the optimized parameter vector. Non-pers models map PERS and fP to 0.
    switch model
        case 'ideal'
            beta     = params(1);
            sub_cred = [0.5, 0.75, 1];
            PERS     = 0;
            fP       = 0;
            
        case 'cred'
            beta     = params(1);
            sub_cred = [params(2:3), 1];
            PERS     = 0;
            fP       = 0;
            
        case 'ideal_pers'
            beta     = params(1);
            sub_cred = [0.5, 0.75, 1];
            PERS     = params(2);
            fP       = params(3);
            
        case 'cred_pers'
            beta     = params(1);
            sub_cred = [params(2:3), 1];
            PERS     = params(4);
            fP       = params(5);
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