function [mdl] = MEM_parameters(parameters, model_type)
% MEM_PARAMETERS Computes Generalized Linear Mixed-Effects Models (GLME) 
% on the fitted computational parameters (Pilot Study - 4 Agents).
%
% INPUTS:
%   parameters - [N_subjects x N_parameters] matrix of fitted parameters.
%   model_type - Integer specifying the regression model to run:
%                1 : Regress CA parameters from credibility-CA model on agent (Fig 3c).
%                2 : Regress CA parameters from credibility-valence CA model on agent and valence (Fig 5a-b).
%                3 : Regress relative Valence Bias Index (rVBI) on agent.
%
% OUTPUTS:
%   mdl        - Fitted GLME object.

    n_subj = size(parameters, 1);

    switch model_type
        case 1
            % =============================================================
            % Model 1: Credibility-CA (4 parameters: CA_0.5, CA_0.7, CA_0.85, CA_1)
            % =============================================================
            ca = parameters(:, 1:4)'; 
            n_obs = size(ca, 2);
            
            % Dummy coding for agents (Baseline is 1-star / 0.5 credibility)
            agent2 = repmat([0; 1; 0; 0], 1, n_obs);
            agent3 = repmat([0; 0; 1; 0], 1, n_obs);
            agent4 = repmat([0; 0; 0; 1], 1, n_obs);
            ss     = repelem(1:n_obs, 4);
            
            my_table = table(ca(:), agent2(:), agent3(:), agent4(:), ss(:), ...
                'VariableNames', {'CA', 'AGENT_2STAR', 'AGENT_3STAR', 'AGENT_4STAR', 'SS'});
            
            formula = 'CA ~ AGENT_2STAR + AGENT_3STAR + AGENT_4STAR + (1|SS)';
            mdl = fitglme(my_table, formula, 'Distribution', 'normal', ...
                'CheckHessian', true, 'FitMethod', 'Laplace');

        case 2
            % =============================================================
            % Model 2: Credibility-Valence CA (8 parameters)
            % Order: CA-_0.5, CA-_0.7, CA-_0.85, CA-_1, CA+_0.5, CA+_0.7, CA+_0.85, CA+_1
            % =============================================================
            ca = parameters(:, 1:8)';
            n_obs = size(ca, 2);
            
            % Dummy coding for 4 agents across both valences
            agent2 = repmat([0; 1; 0; 0; 0; 1; 0; 0], 1, n_obs);
            agent3 = repmat([0; 0; 1; 0; 0; 0; 1; 0], 1, n_obs);
            agent4 = repmat([0; 0; 0; 1; 0; 0; 0; 1], 1, n_obs);
            
            % Effect coding for valence (-0.5 = Negative, 0.5 = Positive)
            valence = repmat([-0.5; -0.5; -0.5; -0.5; 0.5; 0.5; 0.5; 0.5], 1, n_obs);
            ss      = repelem(1:n_obs, 8);
            
            my_table = table(ca(:), agent2(:), agent3(:), agent4(:), valence(:), ss(:), ...
                'VariableNames', {'CA', 'AGENT_2STAR', 'AGENT_3STAR', 'AGENT_4STAR', 'VALENCE', 'SS'});
            
            formula = 'CA ~ VALENCE * (AGENT_2STAR + AGENT_3STAR + AGENT_4STAR) + (1|SS)';
            mdl = fitglme(my_table, formula, 'Distribution', 'normal', ...
                'CheckHessian', true, 'FitMethod', 'Laplace');

        case 3
            % =============================================================
            % Model 3: Relative Valence Bias Index (rVBI)
            % Calculated from Credibility-Valence parameters
            % rVBI = (CA+ - CA-) / (|CA+| + |CA-|)
            % =============================================================
            % Calculate rVBI across all 4 agents
            ca_neg = parameters(:, 1:4);
            ca_pos = parameters(:, 5:8);
            
            rvbi = (ca_pos - ca_neg) ./ (abs(ca_pos) + abs(ca_neg));
            rvbi = rvbi';
            n_obs = size(rvbi, 2);
            
            % Dummy coding for 4 agents
            agent2 = repmat([0; 1; 0; 0], 1, n_obs);
            agent3 = repmat([0; 0; 1; 0], 1, n_obs);
            agent4 = repmat([0; 0; 0; 1], 1, n_obs);
            ss     = repelem(1:n_obs, 4);
            
            my_table = table(rvbi(:), agent2(:), agent3(:), agent4(:), ss(:), ...
                'VariableNames', {'rVBI', 'AGENT_2STAR', 'AGENT_3STAR', 'AGENT_4STAR', 'SS'});
            
            formula = 'rVBI ~ AGENT_2STAR + AGENT_3STAR + AGENT_4STAR + (1|SS)';
            mdl = fitglme(my_table, formula, 'Distribution', 'normal', ...
                'CheckHessian', true, 'FitMethod', 'Laplace');

            
        otherwise
            error('Invalid model_type specified. Please choose 1, 2, 3, or 4.');
    end 
end