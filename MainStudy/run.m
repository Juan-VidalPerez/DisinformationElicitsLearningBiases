% Squeeze and rename fvals (reduces 204x1x5 to 204x5)
fval_simCred_fitCred       = squeeze(fval3D_simCred_fitCred);
fval_simCred_fitCredval    = squeeze(fval3D_simCred_fitCredval);
fval_simCred_fitTruth      = squeeze(fval3D_simCred_fitTruth);

fval_simIdeal_fitCred      = squeeze(fval3D_simIdeal_fitCred);
fval_simIdeal_fitCredval   = squeeze(fval3D_simIdeal_fitCredval);
fval_simIdeal_fitTruth     = squeeze(fval3D_simIdeal_fitTruth);

% Rename params (maintaining their 3D structure: 204 x par x 5)
params_simCred_fitCred     = params3D_simCred_fitCred;
params_simCred_fitCredval  = params3D_simCred_fitCredval;
params_simCred_fitTruth    = params3D_simCred_fitTruth;

params_simIdeal_fitCred    = params3D_simIdeal_fitCred;
params_simIdeal_fitCredval = params3D_simIdeal_fitCredval;
params_simIdeal_fitTruth   = params3D_simIdeal_fitTruth;

% Clear the old variables from the workspace to keep things tidy
clear fval3D_simCred_fitCred fval3D_simCred_fitCredval fval3D_simCred_fitTruth ...
      fval3D_simIdeal_fitCred fval3D_simIdeal_fitCredval fval3D_simIdeal_fitTruth ...
      params3D_simCred_fitCred params3D_simCred_fitCredval params3D_simCred_fitTruth ...
      params3D_simIdeal_fitCred params3D_simIdeal_fitCredval params3D_simIdeal_fitTruth