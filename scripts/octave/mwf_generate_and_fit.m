% mwf_generate_and_fit.m
% Generate a synthetic MET2 signal using qMRLab's mwf.equation, then fit it back with qMRLab's mwf.fit.
%
% Required input variables (set by --eval):
%   qMRLab_path (string)
%   out_mat (string)
%   MWF_percent (scalar)  e.g., 15 for 15%
%   T2MW_ms (scalar)
%   T2IEW_ms (scalar)
%   Cutoff_ms (scalar)   e.g., 40
%
% Optional:
%   EchoTimes_ms (vector)  if not provided, defaults to (10:10:320)'
%   NoiseModel (string)    'none'|'gaussian'|'rician' (default: 'none')
%   NoiseSigma (scalar)    (default: 0)
%   Seed (scalar)          RNG seed (default: 0)
%   QmrlabSigma (scalar)   overrides Model.options.Sigma used in fitting (default: NoiseSigma)
%
% Output saved to out_mat:
%   EchoTimes_ms, Signal_clean, Signal, FitResults, Spectrum_fit, T2vals

try
  pkg load statistics;
  pkg load optim;
catch
  % ok: qMRLab may still work without these packages depending on install
end

addpath(genpath(qMRLab_path));

if exist('EchoTimes_ms', 'var') == 0
  EchoTimes_ms = (10:10:320)';
end

if exist('NoiseModel', 'var') == 0
  NoiseModel = 'none';
end
if exist('NoiseSigma', 'var') == 0
  NoiseSigma = 0;
end
if exist('Seed', 'var') == 0
  Seed = 0;
end
if exist('QmrlabSigma', 'var') == 0
  QmrlabSigma = NoiseSigma;
end

% Deterministic RNG
rand('state', Seed);
randn('state', Seed);

Model = mwf;
Model.Prot.MET2data.Mat = EchoTimes_ms;

% Set cutoff option (qMRLab uses options.Cutoffms)
if exist('Cutoff_ms', 'var') ~= 0
  Model.options.Cutoffms = Cutoff_ms;
  Model = Model.UpdateFields();
end

% Set Sigma option (used in multi_comp_fit_v2 for regNNLS)
Model.options.Sigma = QmrlabSigma;

% Equation options (variances) — match defaults used in mwf.m when not provided
Opt = struct();
Opt.T2Spectrumvariance_Myelin = 5;
Opt.T2Spectrumvariance_IEIntraExtracellularWater = 20;

% Generate signal using qMRLab's own equation (MWF in %)
[Signal_clean, Spectrum_true] = Model.equation([MWF_percent, T2MW_ms, T2IEW_ms], Opt);

Signal = Signal_clean;
nm = lower(strtrim(NoiseModel));
if strcmp(nm, 'none') || strcmp(nm, '') || strcmp(nm, 'no')
  % no-op
elseif strcmp(nm, 'gaussian')
  Signal = Signal_clean + NoiseSigma * randn(size(Signal_clean));
elseif strcmp(nm, 'rician')
  n1 = NoiseSigma * randn(size(Signal_clean));
  n2 = NoiseSigma * randn(size(Signal_clean));
  Signal = sqrt((Signal_clean + n1).^2 + (n2).^2);
else
  error(['Unknown NoiseModel: ' NoiseModel]);
end

data = struct();
data.MET2data = Signal;
data.Mask = 1;

[FitResults, Spectrum_fit] = Model.fit(data);

T2 = getT2(Model, EchoTimes_ms);
T2vals = T2.vals;

% Use MAT v7 binary (scipy.io.loadmat can read this; v7.3/HDF5 cannot).
save(out_mat, 'EchoTimes_ms', 'Signal_clean', 'Signal', 'FitResults', 'Spectrum_fit', 'T2vals', '-mat7-binary');
