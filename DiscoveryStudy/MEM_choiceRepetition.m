function [mdl] = MEM_choiceRepetition(data)
% MEM_CHOICEREPETITION Computes the mixed-effects binomial regression for 
% choice repetition (Pilot Study - 4 Agents) and plots the behavioral results.
%
% INPUTS:
%   data - N x 2 cell array. 
%          Column 1: Subject ID.
%          Column 2: Blocked pilot data.
%
% OUTPUTS:
%   mdl  - Fitted Generalized Linear Mixed-Effects Model (GLME).

    close all;
    
    n_subj = size(data, 1);
    
    % Preallocate arrays for speed and data validation
    % 4 (agents) x 2 (feedback) x 2 (accuracy/better) = 16 conditions per subject
    n_conditions_total = n_subj * 16;
    
    REPEAT        = nan(n_conditions_total, 1);
    OPPORTUNITIES = nan(n_conditions_total, 1);
    AGENT_2STAR   = nan(n_conditions_total, 1);
    AGENT_3STAR   = nan(n_conditions_total, 1);
    AGENT_4STAR   = nan(n_conditions_total, 1);
    PREV_F        = nan(n_conditions_total, 1);
    BETTER        = nan(n_conditions_total, 1);
    SS            = nan(n_conditions_total, 1);
    
    % Store probability of repeating for the plot: [Subj, Agent, Feedback, Better]
    p_repeat = nan(n_subj, 4, 2, 3); 
    
    counter = 1;
    for ss = 1:n_subj
        curr_data = data{ss, 2}; % Extract blocked data
        
        for agent = 1:4
            for feedback = 0:1
                for better = 0:0.5:1
                    
                    % Create logical masks for trial extraction
                    % Operating column-by-column across rows safely prevents crossing block boundaries
                    valid_prev_trial = curr_data.pick(:, 1:end-1) ~= 0 & ~isnan(curr_data.pick(:, 1:end-1));
                    valid_curr_trial = curr_data.pick(:, 2:end) ~= 0 & ~isnan(curr_data.pick(:, 2:end));
                    is_agent         = curr_data.agent(:, 1:end-1) == agent;
                    is_feedback      = curr_data.feedback(:, 1:end-1) == feedback;
                    
                    % Find trials where the previous choice accuracy matched the current 'better' loop index
                    % (Control blocks yielding accuracy=0.5 are naturally ignored here)
                    is_better        = curr_data.accuracy(:, 1:end-1) == better;
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
                    AGENT_4STAR(counter)   = (agent == 4);
                    
                    % Effect coding (-0.5, 0.5) to make main effects interpretable at the mean
                    PREV_F(counter)        = feedback - 0.5; 
                    BETTER(counter)        = better - 0.5;
                    SS(counter)            = ss; 
                    
                    % Compute P(repeat) specifically for visualization
                    p_repeat(ss, agent, feedback + 1, better*2 + 1) = curr_repeat / curr_opps;
                    
                    counter = counter + 1;
                end
            end
        end
    end

    % =====================================================================
    % Mixed Effects Model Fitting
    % =====================================================================
    my_table = table(REPEAT, PREV_F, BETTER, AGENT_2STAR, AGENT_3STAR, AGENT_4STAR, SS, ...
                     'VariableNames', {'REPEAT', 'PREV_F', 'BETTER', 'AGENT_2STAR', 'AGENT_3STAR', 'AGENT_4STAR', 'SS'});
                 
    % Formula models choice repetition based on accuracy, previous feedback, and agent credibilities
    my_model = 'REPEAT ~ 1 + BETTER*PREV_F*(AGENT_2STAR + AGENT_3STAR + AGENT_4STAR) + (1|SS)';
    
    mdl = fitglme(my_table(OPPORTUNITIES>0,:), my_model, ...
                  'Distribution', 'Binomial', ...
                  'BinomialSize', OPPORTUNITIES(OPPORTUNITIES>0), ...
                  'CheckHessian', true, ...
                  'FitMethod', 'Laplace');

    % =====================================================================
    % Visualization
    % =====================================================================
    % Average across the 'better' dimension for the plot
    plot_data = [mean(p_repeat(:, :, 1, :), 4, 'omitnan'), ... % Negative feedback (Agents 1,2,3,4)
                 mean(p_repeat(:, :, 2, :), 4, 'omitnan')];    % Positive feedback (Agents 1,2,3,4)
             
    % Expanded tick marks and labels for 4 agents per cluster
    x_ticks = [1, 1.5, 2, 2.5,   5, 5.5, 6, 6.5];
    x_labels = {'', 'Negative', '', '', '', 'Positive', '', ''};
    
    % Updated 4-Agent Color Palette
    agent_colors = [109, 25,  80; 
                    161, 81,  77;
                    226, 173, 93;
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
    
    % select matrix maps negative feedback columns (1-4) to positive feedback columns (5-8)
    select = [1, 5; 
              2, 6; 
              3, 7;
              4, 8];
          
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
                       colors(p, :), 'filled', 'XJitterWidth', 0.25, ...
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
            
            errorbar(my_x_ticks(p), curr_mean, curr_sem, 'Color', 'k', 'LineWidth', 1, 'CapSize', 0);
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