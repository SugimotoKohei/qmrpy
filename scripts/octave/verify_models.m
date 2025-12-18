% verify_models.m
% Input variables (set by --eval):
%   qMRLab_path
%   model_name
%   output_csv

try
    pkg load io;
    pkg load statistics;
    pkg load optim;
catch
    warning('Some packages could not be loaded.');
end
addpath(genpath(qMRLab_path));

% Load Data
data_table = csvread(input_csv, 1, 0); % Skip header
% Headers are assumed: id, t2_true, m0_true, S_0, S_1, ...
% Count columns. S starts from col 4 (1-based index in Octave? csvread loads matrix).
% csvread: 0-based offset? second arg R is row offset (0-based), C is col offset (0-based).
% We start R=1 to skip header.

% Re-read with pkg load io if needed, or simple dlmread.
% Let's use dlmread which is standard.
data_matrix = dlmread(input_csv, ',', 1, 0);

ids = data_matrix(:, 1);
% S columns start at index 3 (0-based is col 3 -> 4th col).
% Matrix: col 1=id, col 2=t2, col 3=m0, col 4...=S
signals = data_matrix(:, 4:end);

if strcmp(model_name, 'mono_t2')
    Model = mono_t2;
    % Load protocol
    % Assuming sidecar json exists
    [dir, name, ext] = fileparts(input_csv);
    proto_file = fullfile(dir, [name(1:end-6) '_protocol.json']); % remove _input
    % Parsing JSON in pure Octave is annoying without packages.
    % Let's assume fixed protocol for now or parse simple string.
    % Or use python to pass TE as a string in --eval.
    % For now: hardcoded TE matching python script for verification.
    % Python: [10, 20, ..., 100]
    TE = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]';
    
    Model.Prot.SEdata.Mat = TE;
    
    % Fitting loop
    n_samples = size(signals, 1);
    t2_oct = zeros(n_samples, 1);
    m0_oct = zeros(n_samples, 1);
    
    % Silence output
    warning('off', 'all');
    
    for i = 1:n_samples
        data = struct;
        data.SEdata = signals(i, :)';
        FitResults = Model.fit(data);
        t2_oct(i) = FitResults.T2;
        m0_oct(i) = FitResults.M0;
    end
    
    % Save Output
    % columns: id, t2_oct, m0_oct
    out_data = [ids, t2_oct, m0_oct];
    
    % Write CSV
    fid = fopen(output_csv, 'w');
    fprintf(fid, 'id,t2_oct,m0_oct\n');
    fclose(fid);
    dlmwrite(output_csv, out_data, '-append', 'precision', 16);

elseif strcmp(model_name, 'vfa_t1')
    Model = vfa_t1;
    % Protocol: FlipAngle (deg), TR (s)
    % Python: [3, 10, 20, 30], TR=0.015
    % We should parse from json ideally but hardcoding for speed/stability first.
    FA = [3, 10, 20, 30]';
    TR = 0.015;
    
    Model.Prot.VFAData.Mat = [FA, repmat(TR, length(FA), 1)]; 
    % qMRLab vfa_t1 protocol format: [FlipAngle TR]
    
    % Input CSV: id, t1, m0, b1, S0...S3
    % col 1=id, 2=t1, 3=m0, 4=b1, 5..=Signals
    b1_map = data_matrix(:, 4);
    signals = data_matrix(:, 5:end);
    
    n_samples = size(signals, 1);
    t1_oct = zeros(n_samples, 1);
    m0_oct = zeros(n_samples, 1);
    
    warning('off', 'all');
    for i = 1:n_samples
        data = struct;
        data.VFAData = signals(i, :)';
        data.B1map = b1_map(i);
        FitResults = Model.fit(data);
        t1_oct(i) = FitResults.T1;
        m0_oct(i) = FitResults.M0;
    end
    
    out_data = [ids, t1_oct, m0_oct];
    fid = fopen(output_csv, 'w');
    fprintf(fid, 'id,t1_oct,m0_oct\n');
    fclose(fid);
    dlmwrite(output_csv, out_data, '-append', 'precision', 16);

elseif strcmp(model_name, 'b1_dam')
    Model = b1_dam;
    % Protocol: Alpha (deg)
    % Python: 60
    alpha = 60;
    
    % qMRLab b1_dam protocol:
    % Looks like it expects two angles?
    % Model.Prot.SEQdata.Mat = [alpha; 2*alpha]?
    % Checking b1_dam.m source is best, but usually it knows 1 and 2.
    % Actually: Prot.MPdata.Mat = [alpha 2*alpha]?
    % Let's try setting [60 120]'.
    Model.Prot.SEQdata.Mat = [alpha; alpha*2];
    
    % Input CSV: id, b1_true, m0_true, S1, S2
    % col 1...3, 4=S1, 5=S2
    signals = data_matrix(:, 4:5);
    
    n_samples = size(signals, 1);
    b1_oct = zeros(n_samples, 1);
    
    warning('off', 'all');
    for i = 1:n_samples
        data = struct;
        % b1_dam expects SFalpha and SF2alpha?
        % Error said 'structure has no member SF2alpha', implying it looked for it.
        
        data.SFalpha = signals(i, 1);
        data.SF2alpha = signals(i, 2);
        
        FitResults = Model.fit(data);
        if i==1, disp(FitResults); end
        % Try to find field
        if isfield(FitResults, 'B1map_raw'), b1_oct(i)=FitResults.B1map_raw;
        elseif isfield(FitResults, 'B1map'), b1_oct(i)=FitResults.B1map;
        else, error('Cannot find B1 field'); end
    end
    
    out_data = [ids, b1_oct];
    fid = fopen(output_csv, 'w');
    fprintf(fid, 'id,b1_oct\n');
    fclose(fid);
    dlmwrite(output_csv, out_data, '-append', 'precision', 16);

elseif strcmp(model_name, 'inversion_recovery')
    Model = inversion_recovery;
    % Protocol: IRData.Mat = [TI] (ms)
    % Python: [50, 100, 200, 400, 800, 1600, 3000]
    TI = [50, 100, 200, 400, 800, 1600, 3000]';
    Model.Prot.IRData.Mat = TI;
    % Fix TR to infinite (100s) because Python data uses ra=M0, rb=-2M0 (Infinite TR assumption)
    Model.Prot.TimingTable.Mat = 100000;
    
    signals = data_matrix(:, 4:end);
    n_samples = size(signals, 1);
    t1_oct = zeros(n_samples, 1);
    
    warning('off', 'all');
    for i = 1:n_samples
        data = struct;
        data.IRData = signals(i, :)';
        FitResults = Model.fit(data);
        t1_oct(i) = FitResults.T1;
        % qMRLab T1 is ms if TI is ms.
    end
    
    % We just save T1 for now
    out_data = [ids, t1_oct];
    fid = fopen(output_csv, 'w');
    fprintf(fid, 'id,t1_oct\n');
    fclose(fid);
    dlmwrite(output_csv, out_data, '-append', 'precision', 16);

elseif strcmp(model_name, 'mwf')
    Model = mwf;
    % Protocol: MET2data.Mat = [TE] (ms)
    % Python: linear 10..320, 32 points
    TE = linspace(10, 320, 32)';
    Model.Prot.MET2data.Mat = TE;
    
    signals = data_matrix(:, 3:end); % col 3 is where S_0 starts?
    % Input CSV: id, mwf_true, S0...S31
    % col 1=id, 2=mwf_true, 3=S0...
    signals = data_matrix(:, 3:end);
    
    n_samples = size(signals, 1);
    mwf_oct = zeros(n_samples, 1);
    
    warning('off', 'all');
    for i = 1:n_samples
        data = struct;
        data.MET2data = signals(i, :)';
        data.Mask = 1;
        FitResults = Model.fit(data);
        mwf_oct(i) = FitResults.MWF;
    end
    
    out_data = [ids, mwf_oct];
    fid = fopen(output_csv, 'w');
    fprintf(fid, 'id,mwf_oct\n');
    fclose(fid);
    dlmwrite(output_csv, out_data, '-append', 'precision', 16);
    
else
    error(['Unknown model: ' model_name]);
end
