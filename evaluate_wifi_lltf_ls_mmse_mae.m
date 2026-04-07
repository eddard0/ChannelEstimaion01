function [lsNrmse, lmmseNrmse] = evaluate_wifi_lltf_ls_mmse_nrmse(evalCsv)
% Evaluate LS and same-file LMMSE(MMSE-style baseline) using one *_eval.csv file.
%
% Dataset row format:
% [ X(1)_I X(1)_Q ... X(160)_I X(160)_Q  Y(1)_I Y(1)_Q ... Y(52)_I Y(52)_Q ]
%
% X : received L-LTF after channel + CFO + AWGN
% Y : ground-truth channel label H(52)
%
% Output:
%   lsNrmse    : normalized RMSE of LS estimate
%   lmmseNrmse : normalized RMSE of same-file LMMSE estimate

if nargin < 1
    evalCsv = 'wifi_lltf_dataset_6db_eval.csv';
end

cfg = wlanNonHTConfig('ChannelBandwidth', 'CBW20');

[rxEval, hTrue] = readDataset(evalCsv);
numSamples = size(rxEval, 2);

fprintf('EVAL CSV: %s\n', evalCsv);

hLs = zeros(52, numSamples);

for n = 1:numSamples
    rx = rxEval(:, n);
    hLs(:, n) = estimateLS(rx, cfg);
end

lsNrmse = computeNRMSE(hLs, hTrue);

muH = mean(hTrue, 2);
muZ = mean(hLs, 2);

Hc = hTrue - muH;
Zc = hLs   - muZ;

Chz = (Hc * Zc') / max(numSamples - 1, 1);
Czz = (Zc * Zc') / max(numSamples - 1, 1);

reg = 1e-8 * real(trace(Czz)) / size(Czz, 1);
W = Chz / (Czz + reg * eye(size(Czz)));

hLmmse = muH + W * (hLs - muZ);
lmmseNrmse = computeNRMSE(hLmmse, hTrue);

fprintf('Results\n');
fprintf('  LS NRMSE    : %.6f\n', lsNrmse);
fprintf('  LMMSE NRMSE : %.6f\n', lmmseNrmse);
fprintf('\n');
fprintf('Note: LMMSE here is fitted and evaluated on the same *_eval file.\n');
fprintf('      So it is useful for error inspection, but optimistic as a baseline.\n');

end


function Hls = estimateLS(rxLLTF, cfg)
demodLLTF = wlanLLTFDemodulate(rxLLTF, cfg);
chEst = wlanLLTFChannelEstimate(demodLLTF, cfg);
Hls = chEst(:, 1, 1);
end


function nrmse = computeNRMSE(Hhat, Htrue)
num = sum(abs(Hhat(:) - Htrue(:)).^2);
den = sum(abs(Htrue(:)).^2);

assert(den > 0, 'Ground-truth energy is zero, cannot compute NRMSE.');

nrmse = sqrt(num / den);
end


function [rxEval, hLabel] = readDataset(csvFile)
data = readmatrix(csvFile);

numColsExpected = 160 * 2 + 52 * 2;
assert(size(data, 2) == numColsExpected, ...
    'Unexpected number of columns in %s. Expected %d, got %d.', ...
    csvFile, numColsExpected, size(data, 2));

xPart = data(:, 1:320);
yPart = data(:, 321:424);

rxEval = complex(xPart(:, 1:2:end), xPart(:, 2:2:end)).';
hLabel = complex(yPart(:, 1:2:end), yPart(:, 2:2:end)).';
end
