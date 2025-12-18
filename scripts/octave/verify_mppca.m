% verify_mppca.m
% Wrapper to run denoising_mppca from verify_mppca.py

if ~exist('qMRLab_path', 'var')
    error('qMRLab_path must be defined');
end
addpath(genpath(qMRLab_path));
try
    startup;
catch
end

% Load Packages if needed (MPPCA might not need external packages if it just uses SVD?)
% But let's load just in case
try
    pkg load statistics;
catch
end

% Load Input
load(input_mat); % variable 'Data4D'
disp(['Data4D Size: ', num2str(size(Data4D))]);


% Create Model
Model = denoising_mppca;
data = struct();
data.Data4D = Data4D;
% Create mask (full ones)
data.Mask = ones(size(Data4D,1), size(Data4D,2), size(Data4D,3));

% Run Fit
% Options: kernel=[5 5 5], sampling='full' (default implied by nargout>1 in MPdenoising.m but here calling Model.fit)
% Model.fit calls MPdenoising.
% By default Model.options.sampling might be 'fast'?
% Let's check denoising_mppca defaults. 
% properties: buttons = {'sampling',{'fast','full'},...} -> Default is 'fast'?
% Wait, default in button2opts takes first? 
% Let's force 'full' for parity with our Sliding Window implementation.

Model.options.sampling = 'full';
Model.options.kernel = [5 5 5];

results = Model.fit(data);

% Save Output
denoised = results.Data4D_denoised;
sigma = results.sigma_g;

save(output_mat, 'denoised', 'sigma', '-v7');
