%% How many images of "1" do you want?
K = 50;                       % set to 20, 50, etc.

%% Locate or download the DigitDataset
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
if ~isfolder(digitDatasetPath)
    % Auto-download from MathWorks if not installed locally
    url = 'https://ssd.mathworks.com/supportfiles/nnet/data/digitDataset.zip';
    zipFile = websave('digitDataset.zip', url);
    unzip(zipFile, tempdir);
    digitDatasetPath = fullfile(tempdir,'digitDataset');
end

%% Build an imageDatastore and filter to label '1'
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

idxOnes = find(imds.Labels == categorical("1"));
idxOnes = idxOnes(randperm(numel(idxOnes)));          % random order
idxKeep = idxOnes(1:min(K, numel(idxOnes)));
imds1   = subset(imds, idxKeep);

%% Save the K images to a folder and also stack into an array
outDir = sprintf('digit1_%d', numel(idxKeep));
if ~exist(outDir,'dir'), mkdir(outDir); end

stack28 = zeros(28,28,1,numel(idxKeep),'uint8');      % 28×28×1×K
for i = 1:numel(idxKeep)
    I = readimage(imds1,i);
    if ~isa(I,'uint8'), I = im2uint8(I); end
    stack28(:,:,:,i) = I;
    imwrite(I, fullfile(outDir, sprintf('one_%03d.png', i)));
end
save('digit1_stack28.mat','stack28');                 % for later use

%% Quick preview
figure('Name',sprintf("Handwritten '1' (K=%d)", numel(idxKeep)));
montage(imds1); title(sprintf("K = %d samples of digit '1'", numel(idxKeep)));
%%
fh = figure('Name',sprintf("Handwritten '1' (K=%d)", numel(idxKeep)));
montage(imds1); title(sprintf("K = %d samples of digit '1'", numel(idxKeep)));
exportgraphics(fh, fullfile(outDir, 'montage_digit1.png'), 'Resolution', 200);  % saves PNG
savefig(fh, fullfile(outDir, 'montage_digit1.fig'));                              % optional .fig
%%
clear; clc;
S = load('digit1_stack28.mat');   % loads variable 'stack28'
montage(permute(S.stack28, [1 2 4 3])); colormap gray; axis image off;



%% ===============================================================
%  TQR on MNIST "1": images -> (28×1×28) tensors, LMI -> K, rollout
% ===============================================================
clear; clc; close all; rng(7);

%% ---------- Load K=10 images of digit '1' and build X tensors ----------
K = 12;   % number of frames
try
    S = load('digit1_stack28.mat');   % variable: stack28 (28×28×1×N)
    stack28 = S.stack28;
catch
    % fallback: fetch from DigitDataset
    digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
    if ~isfolder(digitDatasetPath)
        url = 'https://ssd.mathworks.com/supportfiles/nnet/data/digitDataset.zip';
        zipFile = websave('digitDataset.zip', url);
        unzip(zipFile, tempdir);
        digitDatasetPath = fullfile(tempdir,'digitDataset');
    end
    imds = imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
    idx = find(imds.Labels == categorical("1"));
    idx = idx(1:min(K, numel(idx)));
    stack28 = zeros(28,28,1,numel(idx),'uint8');
    for i=1:numel(idx)
        I = readimage(imds,idx(i));
        if ~isa(I,'uint8'), I = im2uint8(I); end
        stack28(:,:,:,i) = I;
    end
end

K = min(K, size(stack28,4));
stack28 = stack28(:,:,:,1:K);

% Normalize to [0,1], and convert each 28×28 frame to 28×1×28 tensor
n = 28; m = 1; r = 28;           % <-- your requested sizes
T = K;                           % #frames
X = zeros(n,m,r,T);
for t = 1:T
    img = double(stack28(:,:,1,t))/255;   % 28×28 in [0,1]
    % Put columns of the image as tubes: X(:,1,:,t) is n×1×r
    X(:,1,:,t) = img;                      % (n × r) mapped to (:,1,:)
end

%% ---------- Build stable Ahat, Bhat, and Qhat per slice ----------
p = n;           % choose p=n
epsilon = 1e-9;  % LMI PD margin
lambda_ridge = 1e-6;    % ridge for U-fit
USE_U_EQ_ZERO = false;   % allow nonzero K

Ahat = zeros(n,n,r);
Bhat = zeros(n,p,r);
Qhat = zeros(n,n,r);

for q = 1:r
    % Stable A_q (spectral radius in (0.3,0.8))
    [Uq,~] = qr(randn(n)); 
    evals  = 0.3 + 0.5*rand(n,1);
    Aq     = Uq * diag(evals) * Uq';
    Ahat(:,:,q) = Aq;

    % B_q inside col(A_q): B_q = A_q * G
    Gq = randn(n,p);
    Bq = Aq * Gq;
    Bhat(:,:,q) = Bq;

    % Q_q = projector onto orthogonal complement of col([A_q B_q])
    Mq = [Aq, Bq];
    [U,S,~] = svd(Mq,'econ');
    rk = rank(S, 1e-10*norm(S,2));
    if rk < n
        Uperp = U(:,rk+1:end);
        Qq = Uperp*Uperp';
    else
        Qq = zeros(n);
    end
    Qhat(:,:,q) = 0.5*(Qq+Qq');    % PSD
end

% Make all slices identical (keeps tensors real after ifft; also robust)
for q = 2:r
    Ahat(:,:,q) = Ahat(:,:,1);
    Bhat(:,:,q) = Bhat(:,:,1);
    Qhat(:,:,q) = Qhat(:,:,1);
end

% Back to time domain tensors
A = real(ifft(Ahat,[],3));   % n×n×r
B = real(ifft(Bhat,[],3));   % n×p×r

%% ---------- Fit U_t so model matches the observed frames ----------
% X_{t+1} ≈ tprod(A, X_t) + tprod(B, U_t)
U = zeros(p,m,r,T-1);

for t = 1:T-1
    U(:,:,:,t) = fit_U_ridge_fft(A, B, X(:,:,:,t), X(:,:,:,t+1), lambda_ridge);
end

%% ---------- Build snapshots X1, X2, V (mode-2 stacking) ----------
nT = T-1; nTeff = m*nT;   % = nT since m=1
X1 = zeros(n,nTeff,r);
X2 = zeros(n,nTeff,r);
V  = zeros(p,nTeff,r);
for t = 1:nT
    idx = (t-1)*m + (1:m);          % here m=1, so idx=t
    X1(:,idx,:) = X(:,:,:,t);
    X2(:,idx,:) = X(:,:,:,t+1);
    V (:,idx,:) = U(:,:,:,t);
end

Yhat1 = fft(X1,[],3);  % n×nTeff×r
Yhat2 = fft(X2,[],3);  % n×nTeff×r
Vhat  = fft(V ,[],3);  % p×nTeff×r

%% ---------- Θ–LMI per slice (with tiny slack on Q*X2Θ) ----------
cvx_quiet true
Theta_hat = complex(zeros(nTeff,n,r));
Khat      = complex(zeros(p    ,n,r));
status    = strings(1,r);

fprintf('--- CVX solve per slice ---\n');
for q = 1:r
    X11 = Yhat1(:,:,q);    % n×nTeff
    X21 = Yhat2(:,:,q);    % n×nTeff
    U11 = Vhat(:,:,q);     % p×nTeff
    Q11 = Qhat(:,:,q);     % n×n

    cvx_begin sdp
        variable Theta(nTeff, n) complex
        variable delta nonnegative
        X1T = X11*Theta;              % n×n
        X2T = X21*Theta;              % n×n

        minimize( delta )             % keep Q*X2T small
        subject to
            X1T == X1T';              % Hermitian
            M = [X1T, X2T; X2T', X1T];
            M == M';
            M >= epsilon*eye(2*n);
            norm(Q11*X2T, 'fro') <= 1e-7 + delta;

            if USE_U_EQ_ZERO
                U11*Theta == 0;
            end
    cvx_end

    status(q) = string(cvx_status);
    fprintf('slice %2d: %s  (delta≈%g)\n', q, cvx_status, max(0,evalin('caller','delta')));

    if contains(cvx_status,'Solved')
        YS = X11*Theta;
        Khat(:,:,q) = U11 * ( Theta / (YS + epsilon*eye(n)) );
    else
        warning('slice %d infeasible -> K_q = 0', q);
        Khat(:,:,q) = zeros(p,n);
    end

    Theta_hat(:,:,q) = Theta;
end

K = real(ifft(Khat,[],3));      % p×n×r  (= 28×28×28)

%% ---------- Stability check (per slice, FFT domain) ----------
rho_open = zeros(1,r);
rho_cl   = zeros(1,r);
for q = 1:r
    rho_open(q) = max(abs(eig(Ahat(:,:,q))));
    rho_cl(q)   = max(abs(eig(Ahat(:,:,q) + Bhat(:,:,q)*Khat(:,:,q))));
end
fprintf('\nOpen-loop  |A_q|           : %s\n', mat2str(rho_open,3));
fprintf('Closed-loop|A_q + B_q K_q| : %s\n', mat2str(rho_cl,3));
if all(rho_cl < 1-1e-12), disp('All slices Schur-stable.'); end

%% ---------- Closed-loop rollout (autonomous) ----------
Acl = A + tprod(B,K);
Tsim = 12;
Xcl  = zeros(n,m,r,Tsim+1);
Xcl(:,:,:,1) = X(:,:,:,1);         % start from last data frame

for t = 1:Tsim
    Xcl(:,:,:,t+1) = tprod(Acl, Xcl(:,:,:,t));
end
%%
% --- Select frames to show ---
frames = [2 4 8 11];
Torig = size(X,   4);
Tcl   = size(Xcl, 4);
frames = frames(frames <= min(Torig, Tcl));   % keep only available frames
Kshow  = numel(frames);

% --- Figure: top row = original (panel a), bottom row = closed loop (panel b) ---
fh = figure('Name','Observed vs Closed-loop (selected frames)');
T = tiledlayout(2, Kshow, 'TileSpacing','none', 'Padding','compact'); % tighter spacing
colormap(gray(256));
set(gcf, 'Color', 'w');

axTop    = gobjects(1, Kshow);
axBottom = gobjects(1, Kshow);

% Top row: original frames
for k = 1:Kshow
    t = frames(k);
    axTop(k) = nexttile(k);
    imagesc(toImage(X(:,:,:,t)), [0 1]); axis image off;
    title(sprintf('observed  t=%d', t));
end

% Bottom row: closed-loop frames
for k = 1:Kshow
    t = frames(k);
    axBottom(k) = nexttile(Kshow + k);
    imagesc(toImage(Xcl(:,:,:,t)), [0 1]); axis image off;
    title(sprintf('closed-loop  t=%d', t));
end

% Panel labels
text(axTop(1),    0.00, 1.02, '(a)', 'Units','normalized', ...
     'FontWeight','bold','FontSize',12, 'HorizontalAlignment','left');
text(axBottom(1), 0.00, 1.02, '(b)', 'Units','normalized', ...
     'FontWeight','bold','FontSize',12, 'HorizontalAlignment','left');

sgtitle('Observed (top) vs Closed-loop (bottom)');
%%
% --- Plot ALL reconstructed frames (closed-loop only) ---
Tcl = size(Xcl, 4);

maxCols = 10;                                   % change if you want more/less per row
nCols   = min(Tcl, maxCols);
nRows   = ceil(Tcl / nCols);

fh = figure('Name','Closed-loop reconstruction (all frames)');
T = tiledlayout(nRows, nCols, 'TileSpacing','none', 'Padding','compact');
colormap(gray(256));
set(gcf, 'Color', 'w');

for k = 1:Tcl
    nexttile(k);
    imagesc(toImage(Xcl(:,:,:,k)), [0 1]); 
    axis image off;
    title(sprintf('t=%d', k));
end

sgtitle(sprintf('Closed-loop reconstruction (T = %d frames)', Tcl));
%%
% --- Build 3-D stack of reconstructed frames and save as .mat ---
Tcl  = size(Xcl, 4);
img0 = toImage(Xcl(:,:,:,1));               % first frame to get H×W
[H, W] = size(img0);

reconFrames = zeros(H, W, Tcl, 'like', img0);
for k = 1:Tcl
    reconFrames(:,:,k) = toImage(Xcl(:,:,:,k));
end

outFile = 'closed_loop_reconstructed_frames.mat';
save(outFile, 'reconFrames', 'Xcl', 'Tcl', '-v7.3');  % -v7.3 is safer for big arrays

fprintf('Saved %d frames to %s\n', Tcl, outFile);
% (optional) Inspect contents:
% whos('-file', outFile)

%%
% --- Select frames to show ---
frames = [2 4 8 11];
Torig = size(X,4);  Tcl = size(Xcl,4);
frames = frames(frames <= min(Torig, Tcl));
Kshow  = numel(frames);

% --- Figure / tiles ---
figure('Name','Observed vs Closed-loop (selected frames)');
tiledlayout(2, Kshow, 'TileSpacing','none', 'Padding','compact');
colormap(gray(256)); set(gcf,'Color','w');

axTop    = gobjects(1,Kshow);
axBottom = gobjects(1,Kshow);

% Top row (observed) — no in-image labels
for k = 1:Kshow
    t = frames(k);
    axTop(k) = nexttile(k);
    imagesc(toImage(X(:,:,:,t)), [0 1]); axis image off;
end

% Bottom row (closed-loop) — no in-image labels
for k = 1:Kshow
    t = frames(k);
    axBottom(k) = nexttile(Kshow+k);
    imagesc(toImage(Xcl(:,:,:,t)), [0 1]); axis image off;
end

% ----- Bring rows closer -----
set([axTop(:); axBottom(:)], 'Units','normalized');
posTop1    = axTop(1).Position;
posBottom1 = axBottom(1).Position;
gap = posBottom1(2) - (posTop1(2) + posTop1(4));
desired_gap = gap/5;                      % smaller gap
shrink = max(0, gap - desired_gap);
if shrink > 0
    for k = 1:Kshow
        p = axBottom(k).Position;
        p(2) = p(2) - shrink;             % move bottom row up
        set(axBottom(k), 'Position', p);
    end
end

% Refresh positions
posTop    = axTop(1).Position;
posBottom = axBottom(1).Position;

% ----- Row labels 'a' and 'b' to the LEFT -----
xoff = 0.05;  % how far left to place the label
annotation('textbox', [max(posTop(1)-xoff,0.001),  posTop(2)+posTop(4)/2-0.02, 0.03, 0.04], ...
    'String','a', 'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','right','VerticalAlignment','middle','EdgeColor','none');
annotation('textbox', [max(posBottom(1)-xoff,0.001), posBottom(2)+posBottom(4)/2-0.02, 0.03, 0.04], ...
    'String','b', 'FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','right','VerticalAlignment','middle','EdgeColor','none');

% ----- Single "t=..." label ABOVE each column (top row only) -----
for k = 1:Kshow
    p = axTop(k).Position;                         % [x y w h]
    x_center = p(1) + p(3)/2;
    y_above  = p(2) + p(4) + 0.008;                % small offset above
    annotation('textbox', [x_center-0.04, y_above, 0.08, 0.03], ...
        'String', sprintf('t=%d', frames(k)), ...
        'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
        'FontWeight','bold', 'Color','k', 'EdgeColor','none');
end

% ----- Thin white separators between columns -----
sepw = 0.003;  % thickness (normalized)
y_low  = min(posBottom(2), posTop(2));
y_high = max(posBottom(2)+posBottom(4), posTop(2)+posTop(4));
for k = 1:Kshow-1
    x_sep = axTop(k).Position(1) + axTop(k).Position(3);
    annotation('rectangle', [x_sep - sepw/2, y_low, sepw, y_high - y_low], ...
               'FaceColor','w','EdgeColor','w');
end

% No sgtitle
%%
% Inputs: X, Xcl (n×1×r×T), toImage = @(T3) squeeze(T3(:,1,:))
frames = [2 4 8 11];
Tmax = min(size(X,4), size(Xcl,4));
frames = frames(frames <= Tmax);
Kshow  = numel(frames);

figure('Color','w','Toolbar','none','MenuBar','none');
colormap(gray(256));

% --- Layout (normalized) ---
LM=0.08; RM=0.04; TM=0.05; BM=0.06;     % outer margins
HG=0.02; VG=0.000;                       % column gap, vertical gap ≈ 0
W  = (1 - LM - RM - (Kshow-1)*HG) / Kshow;  % tile width
Hmax = (1 - TM - BM - VG) / 2;

% Square axes to eliminate internal padding
Hsq = min(W, Hmax);

rowBand   = 2*Hmax + VG;
usedH     = 2*Hsq + VG;
rowOffset = (rowBand - usedH)/2;
yBottom   = BM + rowOffset;
yTop      = yBottom + Hsq + VG;

axTop    = gobjects(1,Kshow);
axBottom = gobjects(1,Kshow);

% ----- Top row: observed -----
for k = 1:Kshow
    left = LM + (k-1)*(W+HG) + 0.5*(W - Hsq);
    axTop(k) = axes('Units','normalized', ...
        'Position',[left, yTop, Hsq, Hsq], ...
        'PositionConstraint','innerposition', ...
        'ActivePositionProperty','position');
    imagesc(toImage(X(:,:,:,frames(k))), [0 1]); axis off tight;
    set(axTop(k),'LooseInset',[0 0 0 0]);
end

% ----- Bottom row: closed-loop -----
for k = 1:Kshow
    left = LM + (k-1)*(W+HG) + 0.5*(W - Hsq);
    axBottom(k) = axes('Units','normalized', ...
        'Position',[left, yBottom, Hsq, Hsq], ...
        'PositionConstraint','innerposition', ...
        'ActivePositionProperty','position');
    imagesc(toImage(Xcl(:,:,:,frames(k))), [0 1]); axis off tight;
    set(axBottom(k),'LooseInset',[0 0 0 0]);
end

% ----- Column labels (above top row) -----
for k = 1:Kshow
    p = axTop(k).Position; xC = p(1)+p(3)/2; yA = p(2)+p(4)+0.012;
    annotation('textbox',[xC-0.04, yA, 0.08, 0.03], ...
        'String',sprintf('t=%d',frames(k)), 'EdgeColor','none', ...
        'HorizontalAlignment','center','VerticalAlignment','bottom', ...
        'FontWeight','bold','Color','k');
end

% ----- Row labels on the left -----
xLabel = max(LM-0.035, 0.001);
annotation('textbox',[xLabel, yTop+Hsq/2-0.02, 0.03, 0.04], ...
    'String','a','FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','right','VerticalAlignment','middle', 'EdgeColor','none');
annotation('textbox',[xLabel, yBottom+Hsq/2-0.02, 0.03, 0.04], ...
    'String','b','FontWeight','bold','FontSize',12, ...
    'HorizontalAlignment','right','VerticalAlignment','middle', 'EdgeColor','none');

% ===== Separators =====
% Thicker vertical separators between columns
sepw = 0.010;                           % <-- thicker vertical lines
y0   = yBottom;
y1   = yTop + Hsq;
for k = 1:Kshow-1
    leftCol  = LM + (k-1)*(W+HG);
    xSep = leftCol + W + HG/2 - sepw/2;
    annotation('rectangle',[xSep, y0, sepw, y1 - y0], ...
               'FaceColor','w','EdgeColor','w');
end

% Horizontal separator between the two rows
hsep = 0.012;                           % <-- thickness of the horizontal line
x0   = LM - 0.005;                      % slightly extend into margins
xW   = 1 - RM + 0.005 - x0;
yMid = yBottom + Hsq - hsep/2;          % boundary between rows
annotation('rectangle',[x0, yMid, xW, hsep], ...
           'FaceColor','w','EdgeColor','w');

%%
% Inputs: X, Xcl (n×1×r×T), toImage = @(T3) squeeze(T3(:,1,:))
frames = [2 4 8 11];
Tmax = min(size(X,4), size(Xcl,4));
frames = frames(frames <= Tmax);
Kshow  = numel(frames);

figure('Color','w','Toolbar','none','MenuBar','none');
colormap(gray(256));

% --- Layout (normalized) ---
LM=0.08; RM=0.04; TM=0.05; BM=0.06;
HG=0.02; VG=0.000;
W  = (1 - LM - RM - (Kshow-1)*HG) / Kshow;
Hmax = (1 - TM - BM - VG) / 2;
Hsq = min(W, Hmax);

rowBand   = 2*Hmax + VG;
usedH     = 2*Hsq + VG;
rowOffset = (rowBand - usedH)/2;
yBottom   = BM + rowOffset;
yTop      = yBottom + Hsq + VG;

axTop    = gobjects(1,Kshow);
axBottom = gobjects(1,Kshow);

% ----- Top row: observed -----
for k = 1:Kshow
    left = LM + (k-1)*(W+HG) + 0.5*(W - Hsq);
    axTop(k) = axes('Units','normalized', ...
        'Position',[left, yTop, Hsq, Hsq], ...
        'PositionConstraint','innerposition', ...
        'ActivePositionProperty','position');
    imagesc(toImage(X(:,:,:,frames(k))), [0 1]); axis off tight;
    set(axTop(k),'LooseInset',[0 0 0 0]);
end

% ----- Bottom row: closed-loop -----
for k = 1:Kshow
    left = LM + (k-1)*(W+HG) + 0.5*(W - Hsq);
    axBottom(k) = axes('Units','normalized', ...
        'Position',[left, yBottom, Hsq, Hsq], ...
        'PositionConstraint','innerposition', ...
        'ActivePositionProperty','position');
    imagesc(toImage(Xcl(:,:,:,frames(k))), [0 1]); axis off tight;
    set(axBottom(k),'LooseInset',[0 0 0 0]);
end

% ===== Larger labels =====
timeFS = 18;      % font size for t=…
rowFS  = 22;      % font size for a / b

% Column labels (above top row)
for k = 1:Kshow
    p = axTop(k).Position; xC = p(1)+p(3)/2; yA = p(2)+p(4)+0.012;
    annotation('textbox',[xC-0.05, yA, 0.10, 0.035], ...
        'String',sprintf('t = %d',frames(k)), 'EdgeColor','none', ...
        'HorizontalAlignment','center','VerticalAlignment','bottom', ...
        'FontWeight','bold','Color','k','FontSize',timeFS);
end

% Row labels on the left (a, b)
xLabel = max(LM-0.035, 0.001);
annotation('textbox',[xLabel, yTop+Hsq/2-0.025, 0.03, 0.05], ...
    'String','a','FontWeight','bold','FontSize',rowFS, ...
    'HorizontalAlignment','right','VerticalAlignment','middle', 'EdgeColor','none');
annotation('textbox',[xLabel, yBottom+Hsq/2-0.025, 0.03, 0.05], ...
    'String','b','FontWeight','bold','FontSize',rowFS, ...
    'HorizontalAlignment','right','VerticalAlignment','middle', 'EdgeColor','none');

% ===== Separators =====
% Thicker vertical separators
sepw = 0.010; y0 = yBottom; y1 = yTop + Hsq;
for k = 1:Kshow-1
    leftCol  = LM + (k-1)*(W+HG);
    xSep = leftCol + W + HG/2 - sepw/2;
    annotation('rectangle',[xSep, y0, sepw, y1 - y0], ...
               'FaceColor','w','EdgeColor','w');
end

% Horizontal separator between rows
hsep = 0.012; x0 = LM - 0.005; xW = 1 - RM + 0.005 - x0;
yMid = yBottom + Hsq - hsep/2;
annotation('rectangle',[x0, yMid, xW, hsep], ...
           'FaceColor','w','EdgeColor','w');

%% ---------------------------------------------------------------
%  Open-loop reconstruction with fitted U_t  (check A,B,U quality)
%  Uses variables already in workspace: A, B, U, X, tprod
% ---------------------------------------------------------------
assert(exist('A','var')==1 && exist('B','var')==1 ...
    && exist('U','var')==1 && exist('X','var')==1, ...
    'Need A, B, U, X in workspace.');

[n,m,r,T] = size(X);
toImage = @(T3) squeeze(T3(:,1,:));   % (n×1×r) -> (n×r), here 28×28

% Simulate forward from the first frame using the fitted inputs U_t
Xrec = zeros(n,m,r,T);
Xrec(:,:,:,1) = X(:,:,:,1);
for t = 1:T-1
    Xrec(:,:,:,t+1) = tprod(A, Xrec(:,:,:,t)) + tprod(B, U(:,:,:,t));
end

% Compare frames 2..9 (8 frames)
ids = 2:min(9,T);              % show t = 2..9 if available
Kshow = numel(ids);

% Per-frame RMSE (in image domain)
rmse = zeros(1,Kshow);
for k = 1:Kshow
    t = ids(k);
    E = toImage(X(:,:,:,t)) - toImage(Xrec(:,:,:,t));
    rmse(k) = norm(E,'fro') / sqrt(numel(E));
end
fprintf('RMSE (frames 2..%d): %s\n', ids(end), mat2str(rmse,3));

% Plot: top = original, bottom = reconstructed (same color limits)
figure('Name','Open-loop reconstruction with fitted U_t'); colormap(gray(256));
for k = 1:Kshow
    t = ids(k);
    subplot(2,Kshow,k);
    imagesc(toImage(X(:,:,:,t)), [0 1]); axis image off;
    title(sprintf('orig  t=%d', t-1));

    subplot(2,Kshow,k+Kshow);
    imagesc(toImage(Xrec(:,:,:,t)), [0 1]); axis image off;
    title(sprintf('recon t=%d\nRMSE=%.2e', t-1, rmse(k)));
end
sgtitle('Model replay:  \hatX_{t+1}=tprod(A,\hatX_t)+tprod(B,U_t)');

% (Optional) show absolute error maps
figure('Name','Absolute error (recon - orig)'); colormap(parula);
for k = 1:Kshow
    t = ids(k);
    errImg = abs(toImage(Xrec(:,:,:,t)) - toImage(X(:,:,:,t)));
    subplot(1,Kshow,k); imagesc(errImg); axis image off;
    title(sprintf('|err| t=%d', t-1));
end



%%
%% ===============================================================
%  UNFOLDING (block–circulant) controller on MNIST "1"
%  Use ONLY the first 3 frames to build the unfolding SDP (fast),
%  then roll out many future frames. Compares with your t-product run.
%
%  Requires these variables from your t-product script in workspace:
%     A (n×n×r), B (n×p×r), U (p×1×r×(T-1)), X (n×1×r×T),  (optional) Qhat
%     K (p×n×r) and Xcl (t-product rollout) are optional (only for plotting).
% ===============================================================

assert(exist('A','var')==1 && exist('B','var')==1 ...
    && exist('U','var')==1 && exist('X','var')==1, ...
    'Need A, B, U, X in workspace before running the unfolding method.');

[n,m,r,T] = size(X);          % here m=1 for MNIST reshape
p = size(B,2);
eps_pd = 1e-9;

%% ---------- Use only the first 3 frames to IDENTIFY the unfolding gain ----------
T_use = min(3, T);                     % << only 3 frames for the SDP
if T_use < 2, error('Need at least 2 frames in X.'); end
nT_small  = T_use - 1;                 % snapshot pairs = 2
nTeff_sm  = m * nT_small;              % = 2 when m=1

% Build small snapshot stacks from first T_use frames
X1_sm = zeros(n, nTeff_sm, r);
X2_sm = zeros(n, nTeff_sm, r);
V_sm  = zeros(p, nTeff_sm, r);
for t = 1:nT_small
    idx = (t-1)*m + (1:m);             % (m=1) -> idx = t
    X1_sm(:,idx,:) = X(:,:,:,t);
    X2_sm(:,idx,:) = X(:,:,:,t+1);
    V_sm (:,idx,:) = U(:,:,:,t);       % truncate U accordingly
end

% Block–circulant lift (small)
Ac = bcirc(A);                   % (nr × nr)
Bc = bcirc(B);                   % (nr × pr)
X1c_sm = bcirc(X1_sm);           % (nr × (nTeff_sm*r))
X2c_sm = bcirc(X2_sm);           % (nr × (nTeff_sm*r))
Vc_sm  = bcirc(V_sm);            % (pr × (nTeff_sm*r))

% Optional weighting Q (if you built Qhat in your t-product code)
if exist('Qhat','var')==1
    Q_td = real(ifft(Qhat,[],3));      % n×n×r
    Qc   = bcirc(Q_td);          % (nr × nr)
else
    Qc   = zeros(n*r);
end

%% ---------- Unfolding SDP on the SMALL stacks ----------
cvx_quiet true
cvx_begin sdp
    variable Theta_c(nTeff_sm*r, n*r)        % lifted Θ using only 2 columns (m=1)
    X1T = X1c_sm * Theta_c;                   % (nr × nr)
    X2T = X2c_sm * Theta_c;                   % (nr × nr)

    % PSD/Hermitian Lyapunov-like constraint
    M = [X1T, X2T; X2T', X1T];
    M == M';
    M >= eps_pd * eye(2*n*r);

    % Keep Qc*X2T small (soft); remove if Qc is zero
    minimize( norm(Qc * X2T, 'fro') )
cvx_end
fprintf('Unfolding SDP (3-frame ID) status: %s\n', cvx_status);

% Gain in lifted space, then closed loop in lifted space
YS   = X1T + eps_pd*eye(n*r);
Kc   = Vc_sm * (Theta_c / YS);         % (pr × nr)
Aclc = Ac + Bc * Kc;                   % (nr × nr)

%% ---------- Closed-loop rollout in UNFOLDED space ----------
toImage = @(T3) squeeze(T3(:,1,:));    % (n×1×r) -> (n×r)
Tsim = 12;

Xcl_unf = zeros(n,m,r,Tsim+1);
Xcl_unf(:,:,:,1) = X(:,:,:,1);         % start from first observed frame
xvec = unfold_local(Xcl_unf(:,:,:,1)); % (nr × 1)
for t = 1:Tsim
    xvec = Aclc * xvec;
    Xcl_unf(:,:,:,t+1) = fold_local(xvec, n, m, r);
end

%% ---------- If t-product rollout not present, create for comparison ----------
if ~exist('K','var') || ~exist('Xcl','var')
    warning('t-product K or Xcl not in workspace — creating t-product rollout for comparison.');
    Acl_t = A + tprod(B, K);   % if K not present this will error; otherwise fine
    if ~exist('Acl_t','var')
        % fallback: closed-loop with zero K
        warning('No K available — comparing unfolding to open-loop t-product.');
        Acl_t = A;
    end
    Xcl = zeros(n,m,r,Tsim+1);
    Xcl(:,:,:,1) = X(:,:,:,1);
    for t = 1:Tsim
        Xcl(:,:,:,t+1) = tprod(Acl_t, Xcl(:,:,:,t));
    end
end

%% ---------- Three-row comparison: Original / Unfolding / t-Product ----------
frames = 2:min(9, min(T, Tsim+1));
Kshow  = numel(frames);
clims  = [0 1];

figure('Name','Original vs Unfolding(3f-ID) vs t-Product');
tiledlayout(3, Kshow, 'TileSpacing','compact', 'Padding','compact');
colormap(gray(256)); set(gcf,'Color','w');

% Row 1: original
for k = 1:Kshow
    t = frames(k);
    nexttile(k);
    imagesc(toImage(X(:,:,:,t)), clims); axis image off;
    title(sprintf('orig  t=%d', t-1));
end

% Row 2: unfolding (bcirc) rollout
for k = 1:Kshow
    t = frames(k);
    nexttile(Kshow + k);
    imagesc(toImage(Xcl_unf(:,:,:,t)), clims); axis image off;
    title(sprintf('unfold  t=%d', t-1));
end

% Row 3: t-product rollout
for k = 1:Kshow
    t = frames(k);
    nexttile(2*Kshow + k);
    imagesc(toImage(Xcl(:,:,:,t)), clims); axis image off;
    title(sprintf('t-prod  t=%d', t-1));
end
sgtitle('MNIST "1": ID with 3 frames — Original vs Unfolding vs t-Product');



