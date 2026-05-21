function [mdl] = MEM_contextEffect(data, kind)
% MEM_CONTEXTEFFECT Computes the mixed-effects binomial regression to analyze 
% how context (same vs. different bandit pair) and previous feedback influence 
% choice repetition (Fig 4).
%
% INPUTS:
%   data - N x 3 cell array. 
%          Column 1: Subject ID.
%          Column 2: Data ordered by bandit pair.
%          Column 3: Data ordered as presented in the task.
%   kind - String specifying the context filter:
%          'same' : Previous game and current game are the SAME (Default).
%          'diff' : Previous game and current game are DIFFERENT.
%
% OUTPUTS:
%   mdl  - Fitted Generalized Linear Mixed-Effects Model (GLME).

    % Set default argument if not provided
    if nargin < 2 || isempty(kind)
        kind = 'same';
    end
    
    if ~ismember(kind, {'same', 'diff'})
        error('Invalid ''kind'' argument. Must be ''same'' or ''diff''.');
    end

    n_subj = size(data, 1);
    
    % The loop isolates current agent = 3 (j=3), so we have:
    % 1 (current agent) x 2 (feedback) x 2 (better) x 3 (prev_agents) = 12 conditions
    n_conditions = n_subj * 12; 
    
    % Preallocate arrays for speed and strict data validation
    REPEAT        = nan(n_conditions, 1);
    OPPORTUNITIES = nan(n_conditions, 1);
    REL_75        = nan(n_conditions, 1);
    REL_100       = nan(n_conditions, 1);
    PREV_REL_50   = nan(n_conditions, 1);
    PREV_REL_75   = nan(n_conditions, 1);
    PREV_REL_100  = nan(n_conditions, 1);
    PREV_F        = nan(n_conditions, 1);
    BETTER        = nan(n_conditions, 1);
    SS            = nan(n_conditions, 1);

    % Define dummy coding maps
    dummy_rel = [1, 0, 0; 
                 0, 1, 0; 
                 0, 0, 1];
             
    counter = 1;
    for i = 1:n_subj
        curr_data = data{i, 2}; % Extract data ordered by bandit pair
        rel_levels = sort(unique(curr_data.rel), 'ascend'); 
        
        for j = 3 % Currently hardcoded to evaluate trials where current agent is 3-star
            for f = 0:1
                for b = 0:1
                    for prev_agent = 1:3
                        
                        % 1. Create base logical masks
                        valid_picks   = (curr_data.pick(:, 1:end-1) ~= 0) & (curr_data.pick(:, 2:end) ~= 0);
                        is_curr_rel   = curr_data.rel(:, 1:end-1) == rel_levels(j);
                        is_prev_agent = curr_data.prev_agent(:, 1:end-1) == prev_agent;
                        is_feedback   = curr_data.feedback(:, 1:end-1) == f;
                        is_better     = ismember(curr_data.accuracy(:, 1:end-1), b);
                        is_repeat     = curr_data.pick(:, 1:end-1) == curr_data.pick(:, 2:end);
                        
                        % 2. Create context-specific mask based on 'kind'
                        if strcmp(kind, 'same')
                            context_mask = curr_data.prev_game(:, 1:end-1) == curr_data.game(:, 1:end-1);
                        else % 'diff'
                            context_mask = curr_data.prev_game(:, 1:end-1) ~= curr_data.game(:, 1:end-1);
                        end
                        
                        % 3. Combine masks and compute sums
                        condition_mask = valid_picks & is_curr_rel & is_prev_agent & ...
                                         is_feedback & is_better & context_mask;
                                     
                        curr_repeat = sum(condition_mask & is_repeat, 'all');
                        curr_opps   = sum(condition_mask, 'all');
                        
                        % 4. Populate GLME arrays
                        REPEAT(counter)        = curr_repeat;
                        OPPORTUNITIES(counter) = curr_opps;
                        
                        % Current agent dummy coding
                        REL_75(counter)        = dummy_rel(2, j);
                        REL_100(counter)       = dummy_rel(3, j);
                        
                        % Previous agent dummy coding
                        PREV_REL_50(counter)   = dummy_rel(1, prev_agent);
                        PREV_REL_75(counter)   = dummy_rel(2, prev_agent);
                        PREV_REL_100(counter)  = dummy_rel(3, prev_agent);
                        
                        % Effect coding for feedback and better (-0.5 to 0.5)
                        PREV_F(counter)        = f - 0.5;
                        BETTER(counter)        = b - 0.5;
                        SS(counter)            = mod(i - 1, 204) + 1;
                        
                        counter = counter + 1;
                    end
                end
            end
        end
    end

    % =====================================================================
    % Mixed Effects Model Fitting
    % =====================================================================
    
    % Construct the table
    my_table = table(REPEAT, PREV_F, BETTER, REL_75, REL_100, ...
                     PREV_REL_50, PREV_REL_75, PREV_REL_100, SS, ...
                     'VariableNames', {'REPEAT', 'PREV_F', 'BETTER', 'REL_75', ...
                                       'REL_100', 'PREV_REL_50', 'PREV_REL_75', ...
                                       'PREV_REL_100', 'SS'});
                                   
    % Filter out rows where OPPORTUNITIES == 0 to prevent division by zero in binomial fit
    valid_idx = OPPORTUNITIES ~= 0;
    valid_table = my_table(valid_idx, :);
    valid_opps  = OPPORTUNITIES(valid_idx);

    % GLME Formula: Evaluates interaction between previous feedback and previous agent credibility
    my_model = 'REPEAT ~ 1 + PREV_F*(PREV_REL_75 + PREV_REL_100) + BETTER + (1|SS)';
    
    mdl = fitglme(valid_table, my_model, ...
                  'Distribution', 'Binomial', ...
                  'BinomialSize', valid_opps, ...
                  'CheckHessian', true, ...
                  'FitMethod', 'Laplace');
end