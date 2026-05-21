function plot_Parameters(parameters, model)
% PLOT_PARAMETERS Visualizes the fitted parameters for the computational models.
%
% INPUTS:
%   parameters - [N_subjects x N_parameters] matrix of fitted parameters.
%   model      - String specifying the model architecture to plot.
%
% HOW TO REPRODUCE PAPER FIGURES:
%   To generate the figures exactly as they appear in the manuscript, load your 
%   workspace with the fitted parameters and run the following commands:
%
%   - Figure 3c (Credibility-CA debias model):
%       plot_Parameters(parameters_cred, 'cred')
%
%   - Figure 3d (Bayesian-credibility model):
%       plot_Parameters(parameters_bay_cred, 'bay_cred')
%
%   - Figure 5 (Credibility-valence CA model, plus absolute and relative VBI):
%       plot_Parameters(parameters_3lr_phase2, 'cred-val')
%
%   - Figure 6c-d (Truth-CA model):
%       plot_Parameters(parameters_truth, 'truth')

    % Standardized color palette for 1-star, 2-star, and 3-star agents
    col_1star = [109, 25,  80]  / 255; % Purple
    col_2star = [212, 136, 73]  / 255; % Orange
    col_3star = [240, 209, 113] / 255; % Yellow
    gray_line = [0.6, 0.6, 0.6];

    % Average parameters for each subject from simulations
    if size(parameters,3)>1
        parameters=mean(parameters,3);
    end

    switch model
        case 'cred'
            % Plot 1: Standard Credibility CA
            figure;
            colors = [col_1star; col_2star; col_3star];
            custom_swarmplot(parameters(:, 1:3), [1, 2, 3], {'0.5', '0.75', '1'}, colors, ...
                'Credibility', 'Fitted CA', [], 'median');
            
            hold on; 
            plot([0, 4], [0, 0], '--', 'Color', gray_line);
            hold off;

        case 'cred-val'
            % Plot 1: Credibility-Valence CA (Split by Positive/Negative Feedback)
            figure;
            colors = [col_1star; col_1star; col_2star; col_2star; col_3star; col_3star];
            custom_swarmplot(parameters(:, [1, 4, 2, 5, 3, 6]), [1, 2, 4, 5, 7, 8], ...
                {'CA^-_{0.5}', 'CA^+_{0.5}', 'CA^-_{0.75}', 'CA^+_{0.75}', 'CA^-_{1}', 'CA^+_{1}'}, ...
                colors, 'Parameter', 'Fitted learning rate', [], 'median');
            
            hold on;
            % Add lines connecting the medians of negative and positive CA pairs
            plot([1, 2], median(parameters(:, [1, 4]), 1), '-', 'LineWidth', 2, 'Color', col_1star);
            plot([4, 5], median(parameters(:, [2, 5]), 1), '-', 'LineWidth', 2, 'Color', col_2star);
            plot([7, 8], median(parameters(:, [3, 6]), 1), '-', 'LineWidth', 2, 'Color', col_3star);
            plot([0, 9], [0, 0], '--', 'Color', gray_line);
            hold off;

            % Plot 2: Valence Bias Indices (Absolute and Relative)
            figure;
            colors = [col_1star; col_2star; col_3star];
            
            % Subplot A: Absolute VBI (CA+ minus CA-)
            subplot(1, 2, 1);
            aVBI = parameters(:, 4:6) - parameters(:, 1:3);
            custom_swarmplot(aVBI, [1, 2, 3], {'0.5', '0.75', '1'}, colors, ...
                'Credibility', 'aVBI', [], 'median');
            hold on; plot([0, 4], [0, 0], '--', 'Color', gray_line); hold off;
            
            % Subplot B: Relative VBI ((CA+ minus CA-) / (|CA+| + |CA-|))
            subplot(1, 2, 2);
            rVBI = aVBI ./ (abs(parameters(:, 4:6)) + abs(parameters(:, 1:3)));
            custom_swarmplot(rVBI, [1, 2, 3], {'0.5', '0.75', '1'}, colors, ...
                'Parameter', 'rVBI', [], 'mean');
            hold on; plot([0, 4], [0, 0], '--', 'Color', gray_line); hold off;

        case 'truth'
             % Plot 1: Standard Credibility CA
            figure;
            subplot(1,4,1:3)
            colors = [col_1star; col_2star; col_3star];
            custom_swarmplot(parameters(:, 1:3), [1, 2, 3], {'0.5', '0.75', '1'}, colors, ...
                'Credibility', 'Fitted CA', [], 'median');
            
            hold on; 
            plot([0, 4], [0, 0], '--', 'Color', gray_line);
            hold off;
            
            %Plot TB
            subplot(1,4,4)
            custom_swarmplot(parameters(:, 4), 1, {''}, [0.3 0.3 0.3], ...
                '', 'Truth Bias Parameter (TB)', [], 'median');
            hold on; 
            plot([0, 2], [0, 0], '--', 'Color', gray_line);
            hold off;

        case 'bay_cred'
            % Plot: Bayesian Credibility Fit (1-star and 2-star only)
            figure;
            colors = [col_1star; col_2star];
            custom_swarmplot(parameters(:, 2:3), [1, 2], {'0.5', '0.75'}, colors, ...
                'Credibility', 'Fitted credibility', [], 'median');
            
            hold on; 
            % Plot ground truth reference lines
            plot([0.5, 1.5], [0.5, 0.5], '--', 'Color', gray_line);
            plot([1.5, 2.5], [0.75, 0.75], '--', 'Color', gray_line);
            hold off;
            
        otherwise
            error('Unknown model type provided. Please use ''cred'', ''cred-val'', ''truth'', or ''bay_cred''.');
    end
end

% =========================================================================
% SUBFUNCTIONS
% =========================================================================

function custom_swarmplot(data, my_x_ticks, x_tick_labels, colors, x_label, y_label, my_title, stat_type)
% CUSTOM_SWARMPLOT Unified function to generate a scatter swarmplot overlayed 
% with central tendencies (mean or median) and error bars (SEM).
    
    n_subj = size(data, 1);
    n_conditions = size(data, 2);

    % Default handling for optional inputs
    if isempty(colors)
        colors = repmat([0, 0, 0], n_conditions, 1);
    end
    if isempty(my_x_ticks)
        my_x_ticks = 1:n_conditions;
    end

    hold on; 
    
    % 1. Plot the raw jittered data points (Swarm)
    for p = 1:n_conditions
        swarmchart(ones(n_subj, 1) * my_x_ticks(p), data(:, p), 5, ...
            colors(p, :), 'filled', 'XJitterWidth', 0.3, ...
            'MarkerFaceAlpha', 0.5, 'MarkerEdgeAlpha', 0.5);
    end

    % 2. Calculate and plot the summary statistics and Error Bars (SEM)
    for p = 1:n_conditions
        if strcmp(stat_type, 'mean')
            central_val = mean(data(:, p), 'omitnan');
        else
            central_val = median(data(:, p), 'omitnan');
        end
        
        sem_val = std(data(:, p), 'omitnan') / sqrt(sum(~isnan(data(:, p))));
        
        errorbar(my_x_ticks(p), central_val, sem_val, 'Color', 'k', ...
                 'LineWidth', 1, 'CapSize', 12);
             
        plot(my_x_ticks(p), central_val, 'o-', 'Color', colors(p, :), ...
            'LineWidth', 1, 'MarkerSize', 8, 'MarkerFaceColor', colors(p, :), ...
            'MarkerEdgeColor', 'black');
    end
    
    hold off;
    
    % 3. Format the Axes
    [my_x_ticks, idx] = sort(my_x_ticks);
    xticks(my_x_ticks);
    xticklabels(x_tick_labels(idx));

    if ~isempty(x_label), xlabel(x_label); end
    if ~isempty(y_label), ylabel(y_label); end
    if ~isempty(my_title), title(my_title); end

    xlim([min(my_x_ticks) - 0.5, max(my_x_ticks) + 0.5]);
end