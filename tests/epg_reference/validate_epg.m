% EPG validation script for qmrpy
% Compares Python implementation with Weigel's reference MATLAB/Octave code
% Run with: octave --no-gui validate_epg.m

pkg load io;

fprintf('=== EPG Validation: Weigel Reference vs qmrpy ===\n\n');

% Test parameters
T2 = 80;    % ms
T1 = 1000;  % ms
ESP = 10;   % ms (echo spacing)
N = 32;     % number of echoes

% Output file for Python comparison
output_file = 'weigel_reference_results.csv';
fid = fopen(output_file, 'w');
fprintf(fid, 'test_name,echo_idx,value\n');

%% Test 1: Ideal CPMG (B1=1.0, alpha=180)
fprintf('Test 1: Ideal CPMG (alpha=180°)\n');
alpha = 180;
[F0, ~, ~, ~, ~] = cp_cpmg_epg_domain_fplus_fminus(N, alpha, ESP, T1, T2);
F0_abs = abs(F0);
fprintf('  First 5 echoes: ');
fprintf('%.6f ', F0_abs(1:5));
fprintf('\n');

for i = 1:N
    fprintf(fid, 'cpmg_b1_1.0,%d,%.12f\n', i, F0_abs(i));
end

%% Test 2: CPMG with B1=0.9 (alpha=162)
fprintf('Test 2: CPMG with B1=0.9 (alpha=162°)\n');
alpha = 162;  % 180 * 0.9
[F0, ~, ~, ~, ~] = cp_cpmg_epg_domain_fplus_fminus(N, alpha, ESP, T1, T2);
F0_abs = abs(F0);
fprintf('  First 5 echoes: ');
fprintf('%.6f ', F0_abs(1:5));
fprintf('\n');

for i = 1:N
    fprintf(fid, 'cpmg_b1_0.9,%d,%.12f\n', i, F0_abs(i));
end

%% Test 3: CPMG with B1=0.8 (alpha=144)
fprintf('Test 3: CPMG with B1=0.8 (alpha=144°)\n');
alpha = 144;  % 180 * 0.8
[F0, ~, ~, ~, ~] = cp_cpmg_epg_domain_fplus_fminus(N, alpha, ESP, T1, T2);
F0_abs = abs(F0);
fprintf('  First 5 echoes: ');
fprintf('%.6f ', F0_abs(1:5));
fprintf('\n');

for i = 1:N
    fprintf(fid, 'cpmg_b1_0.8,%d,%.12f\n', i, F0_abs(i));
end

%% Test 4: Different T2 values
fprintf('Test 4: Different T2 values (T2=50ms)\n');
alpha = 180;
T2_test = 50;
[F0, ~, ~, ~, ~] = cp_cpmg_epg_domain_fplus_fminus(N, alpha, ESP, T1, T2_test);
F0_abs = abs(F0);
fprintf('  First 5 echoes: ');
fprintf('%.6f ', F0_abs(1:5));
fprintf('\n');

for i = 1:N
    fprintf(fid, 'cpmg_t2_50,%d,%.12f\n', i, F0_abs(i));
end

%% Test 5: Short T1 effect
fprintf('Test 5: Short T1 (T1=200ms)\n');
alpha = 180;
T1_test = 200;
[F0, ~, ~, ~, ~] = cp_cpmg_epg_domain_fplus_fminus(N, alpha, ESP, T1_test, T2);
F0_abs = abs(F0);
fprintf('  First 5 echoes: ');
fprintf('%.6f ', F0_abs(1:5));
fprintf('\n');

for i = 1:N
    fprintf(fid, 'cpmg_t1_200,%d,%.12f\n', i, F0_abs(i));
end

fclose(fid);

fprintf('\nResults saved to: %s\n', output_file);
fprintf('\n=== Validation Complete ===\n');
