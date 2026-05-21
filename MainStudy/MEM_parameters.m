function [mdl] = MEM_parameters(parameters, model_type)
% MEM_PARAMETERS Computes Generalized Linear Mixed-Effects Models (GLME) 
% on the fitted computational parameters.
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
%
% HOW TO REPRODUCE PAPER FIGURES/STATS:
%   - Model 1 (Credibility-CA):
%       mdl = MEM_parameters(parameters_cred, 1);
%   - Model 2 (Credibility-Valence CA):
%       mdl = MEM_parameters(parameters_credval, 2);
%   - Model 3 (Relative Valence Bias):
%       mdl = MEM_parameters(parameters_credval, 3);


    n_subj = size(parameters, 1);

    switch model_type
        case 1
            % =============================================================
            % Model 1: Credibility-CA (3 parameters: CA_0.5, CA_0.75, CA_1)
            % =============================================================
            ca = parameters(:, 1:3)'; 
            n_obs = size(ca, 2);
            
            % Dummy coding for agents (Baseline is 1-star / 0.5 credibility)
            cred100 = repmat([0; 0; 1], 1, n_obs);
            cred75  = repmat([0; 1; 0], 1, n_obs);
            ss      = repelem(1:n_obs, 3);
            
            my_table = table(ca(:), cred75(:), cred100(:), ss(:), ...
                'VariableNames', {'CA', 'AGENT_2STAR', 'AGENT_3STAR', 'SS'});
            
            formula = 'CA ~ AGENT_2STAR + AGENT_3STAR + (1|SS)';
            mdl = fitglme(my_table, formula, 'Distribution', 'normal', ...
                'CheckHessian', true, 'FitMethod', 'Laplace');

        case 2
            % =============================================================
            % Model 2: Credibility-Valence CA (6 parameters)
            % Order: CA-_0.5, CA-_0.75, CA-_1, CA+_0.5, CA+_0.75, CA+_1
            % =============================================================
            ca = parameters(:, 1:6)';
            n_obs = size(ca, 2);
            
            % Dummy coding for agents
            cred100 = repmat([0; 0; 1; 0; 0; 1], 1, n_obs);
            cred75  = repmat([0; 1; 0; 0; 1; 0], 1, n_obs);
            
            % Effect coding for valence (-0.5 = Negative, 0.5 = Positive)
            valence = repmat(repelem([-0.5; 0.5], 3), 1, n_obs);
            ss      = repelem(1:n_obs, 6);
            
            my_table = table(ca(:), cred75(:), cred100(:), valence(:), ss(:), ...
                'VariableNames', {'CA', 'AGENT_2STAR', 'AGENT_3STAR', 'VALENCE', 'SS'});
            
            formula = 'CA ~ VALENCE * (AGENT_2STAR + AGENT_3STAR) + (1|SS)';
            mdl = fitglme(my_table, formula, 'Distribution', 'normal', ...
                'CheckHessian', true, 'FitMethod', 'Laplace');

        case 3
            % =============================================================
            % Model 3: Relative Valence Bias Index (rVBI)
            % Calculated from Credibility-Valence parameters
            % rVBI = (CA+ - CA-) / (|CA+| + |CA-|)
            % =============================================================
            % Calculate rVBI across the 3 agents
            ca_neg = parameters(:, 1:3);
            ca_pos = parameters(:, 4:6);
            
            rvbi = (ca_pos - ca_neg) ./ (abs(ca_pos) + abs(ca_neg));
            rvbi = rvbi';
            n_obs = size(rvbi, 2);
            
            % Dummy coding for agents
            cred100 = repmat([0; 0; 1], 1, n_obs);
            cred75  = repmat([0; 1; 0], 1, n_obs);
            ss      = repelem(1:n_obs, 3);
            
            my_table = table(rvbi(:), cred75(:), cred100(:), ss(:), ...
                'VariableNames', {'rVBI', 'AGENT_2STAR', 'AGENT_3STAR', 'SS'});
            
            formula = 'rVBI ~ AGENT_2STAR + AGENT_3STAR + (1|SS)';
            mdl = fitglme(my_table, formula, 'Distribution', 'normal', ...
                'CheckHessian', true, 'FitMethod', 'Laplace');

            
        otherwise
            error('Invalid model_type specified. Please choose 1, 2, 3, or 4.');
    end 
end