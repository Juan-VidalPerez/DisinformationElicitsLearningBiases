function [mdl] = MEM_choiceRepetition(data)
% MEM_CHOICEREPETITION Computes the mixed-effects binomial regression for 
% choice repetition (Fig 3a-b) and plots the behavioral results.
%
% INPUTS:
%   data - N x 3 cell array. 
%          Column 1: Subject ID.
%          Column 2: Data ordered by bandit pair.
%          Column 3: Data ordered as presented in the task.
%
% OUTPUTS:
%   mdl  - Fitted Generalized Linear Mixed-Effects Model (GLME).

    close all;
    
    n_subj = size(data, 1);
    
    % Preallocate arrays for speed and data validation
    % 3 (agents) x 2 (feedback) x 2 (accuracy/better) = 12 conditions per subject
    n_conditions_total = n_subj * 12;
    
    REPEAT        = nan(n_conditions_total, 1);
    OPPORTUNITIES = nan(n_conditions_total, 1);
    AGENT_2STAR   = nan(n_conditions_total, 1);
    AGENT_3STAR   = nan(n_conditions_total, 1);
    PREV_F        = nan(n_conditions_total, 1);
    BETTER        = nan(n_conditions_total, 1);
    SS            = nan(n_conditions_total, 1);
    
    % Store probability of repeating for the plot: [Subj, Agent, Feedback, Better]
    p_repeat = nan(n_subj, 3, 2, 2); 
    
    counter = 1;
    for ss = 1:n_subj
        curr_data = data{ss, 2}; % Extract data ordered by bandit pair
        
        for agent = 1:3
            for feedback = 0:1
                for better = 0:1
                    
                    % Create logical masks for trial extraction
                    % Ensures valid consecutive trials (no timeouts marked as 0)
                    valid_prev_trial = curr_data.pick(:, 1:end-1) ~= 0;
                    valid_curr_trial = curr_data.pick(:, 2:end) ~= 0;
                    is_agent         = curr_data.agent(:, 1:end-1) == agent;
                    is_feedback      = curr_data.feedback(:, 1:end-1) == feedback;
                    is_better        = ismember(curr_data.accuracy(:, 1:end-1), better);
                    is_repeat        = curr_data.pick(:, 1:end-1) == curr_data.pick(:, 2:end);
                    
                    % Calculate occurrences
                    condition_mask = is_agent & is_feedback & valid_prev_trial & valid_curr_trial & is_better;
                    
                    curr_repeat = sum(condition_mask & is_repeat, 'all');
                    curr_opps   = sum(condition_mask, 'all');
                    
                    % Populate GLME vectors
                    REPEAT(counter)        = curr_repeat;
                    OPPORTUNITIES(counter) = curr_opps;
                    AGENT_2STAR(counter)   = (agent == 2);
                    AGENT_3STAR(counter)   = (agent == 3);
                    % Effect coding (-0.5, 0.5) to make main effects interpretable at the mean
                    PREV_F(counter)        = feedback - 0.5; 
                    BETTER(counter)        = better - 0.5;
                    SS(counter)            = mod(ss - 1, 204) + 1; % Wraps Subject IDs if > 204
                    
                    % Compute P(repeat) specifically for visualization
                    p_repeat(ss, agent, feedback + 1, better + 1) = curr_repeat / curr_opps;
                    
                    counter = counter + 1;
                end
            end
        end
    end

    % =====================================================================
    % Mixed Effects Model Fitting
    % =====================================================================
    my_table = table(REPEAT, PREV_F, BETTER, AGENT_2STAR, AGENT_3STAR, SS, ...
                     'VariableNames', {'REPEAT', 'PREV_F', 'BETTER', 'AGENT_2STAR', 'AGENT_3STAR', 'SS'});
                 
    % Formula models choice repetition based on accuracy, previous feedback, and agent credibilities
    my_model = 'REPEAT ~ 1 + BETTER*PREV_F*(AGENT_2STAR + AGENT_3STAR) + (1|SS)';
    
    mdl = fitglme(my_table(OPPORTUNITIES>0,:), my_model, ...
                  'Distribution', 'Binomial', ...
                  'BinomialSize', OPPORTUNITIES(OPPORTUNITIES>0), ...
                  'CheckHessian', true, ...
                  'FitMethod', 'Laplace');

    % =====================================================================
    % Visualization
    % =====================================================================
    % Average across the 'better' dimension for the plot
    plot_data = [mean(p_repeat(:, :, 1, :), 4, 'omitnan'), ... % Negative feedback (Agents 1,2,3)
                 mean(p_repeat(:, :, 2, :), 4, 'omitnan')];    % Positive feedback (Agents 1,2,3)
             
    x_ticks = [1.5, 2, 2.5, 5.5, 6, 6.5];
    x_labels = {'', 'Negative', '', '', 'Positive', ''};
    
    % Colors map to 1-star (Purple), 2-star (Orange), 3-star (Yellow)
    agent_colors = [109, 25,  80; 
                    212, 136, 73; 
                    240, 209, 113] / 255;
    plot_colors  = [agent_colors; agent_colors];
    
    figure;
    my_swarmplot_lines(plot_data, x_ticks, x_labels, plot_colors, ...
                       'Feedback from previous trial', 'P(repeat)', []);
end

% =========================================================================
% SUBFUNCTIONS
% =========================================================================

function my_swarmplot_lines(data, my_x_ticks, x_tick_labels, colors, x_label, y_label, my_title)
% Generates a jittered scatter plot (swarm) connecting means across conditions
    
    % select matrix maps negative feedback columns (1,2,3) to positive feedback columns (4,5,6)
    select = [1, 4; 
              2, 5; 
              3, 6];
          
    if isempty(colors)
        colors = repmat([0, 0, 0], [size(data, 2), 1]);
    end
    if isempty(my_x_ticks)
        my_x_ticks = 1:size(data, 2);
    end

    % 1. Plot individual jittered points (swarm)
    for p = 1:size(data, 2)
        if ismember(p, select(:))
            swarmchart(ones(size(data, 1), 1) * my_x_ticks(p), data(:, p), 5, ...
                       colors(p, :), 'filled', 'XJitterWidth', 0.3, ...
                       'MarkerFaceAlpha', 0.5, 'MarkerEdgeAlpha', 0.5);
            hold on;
        end
    end

    % 2. Connect the condition means with lines (Negative -> Positive)
    plot([my_x_ticks(select(:, 1)); my_x_ticks(select(:, 2))], ...
         [mean(data(:, select(:, 1)), 1, 'omitnan'); mean(data(:, select(:, 2)), 1, 'omitnan')], ...
         'LineWidth', 1, 'Color', [0.5, 0.5, 0.5]);

    % 3. Overlay the Mean markers and Standard Error (SEM) bars
    for p = 1:size(data, 2)
        if ismember(p, select(:))
            curr_mean = mean(data(:, p), 'omitnan');
            curr_sem  = std(data(:, p), 'omitnan') / sqrt(sum(~isnan(data(:, p))));
            
            errorbar(my_x_ticks(p), curr_mean, curr_sem, 'Color', 'k', 'LineWidth', 1);
            plot(my_x_ticks(p), curr_mean, 'o', 'Color', colors(p, :), ...
                 'LineWidth', 1, 'MarkerSize', 8, 'MarkerFaceColor', colors(p, :), ...
                 'MarkerEdgeColor', 'black');
        end
    end
    
    hold off;
    
    % 4. Format Axes
    [my_x_ticks, idx] = sort(my_x_ticks);
    xticks(my_x_ticks);
    xticklabels(x_tick_labels(idx));

    if ~isempty(x_label)
        xlabel(x_label);
    end
    if ~isempty(y_label)
        ylabel(y_label);
    end
    if ~isempty(my_title)
        title(my_title);
    end

    xlim([min(my_x_ticks) - 0.5, max(my_x_ticks) + 0.5]);
    ylim([0, 1]);
end