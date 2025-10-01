clc;

% --- Load video data cube X: 20x20x20 ---
load('squares.mat','X');     % adjust name if needed
X = double(X);
[n, W, F] = size(X);                        % n=20, W=20, F=20
assert(F >= 20 && n==20 && W==20, 'Expecting 20x20x20 for this setup.');

% --- Build X1, X2 by concatenating along 2nd mode after reshaping each frame to 20x1x20 ---
% Each frame (n x W) -> (n x 1 x W) by mapping columns to tube dim (permute)
% Stack frames 1..19 and 2..20 along mode-2.
Xperm = permute(X, [1 3 2]);                % n x F x W  (rows x frames x width->tubes)
X1    = Xperm(:, 1:F-1, :);                 % n x 19 x W
X2    = Xperm(:, 2:F,   :);                 % n x 19 x W

% Dimensions for the t-product setup
nT = size(X1,2);                            % 19
r  = size(X1,3);                            % 20 (tube length = original width)

% --- Generate U (time domain) and Q,R (Fourier domain) "same way" ---
epsilon = 1e-8;

% Random input in time domain (n x nT x r), then FFT -> Uhat
U    = randn(n, nT, r);
Uhat = fft(U, [], 3);

% Qhat, Rhat constructed PSD/PD per Fourier slice (equiv. to your Qfourier/Rfourier)
Qhat = zeros(n,n,r);
Rhat = zeros(n,n,r);
for q = 1:r
    Ae = randn(n,n);
    Be = randn(n,n);
    Qhat(:,:,q) = Ae*Ae';                   % Hermitian PSD
    Rhat(:,:,q) = Be*Be' + 10^(-3)*eye(n);  % Hermitian PD
end

% --- FFT of X1, X2 along tube dimension (t-product diagonalization) ---
X1hat = fft(X1, [], 3);                     % n x nT x r
X2hat = fft(X2, [], 3);                     % n x nT x r
% --- Per-frequency SDPs (pure t-product; no unfolding/bcirc) ---
cvx_quiet true
% cvx_solver SDPT3  % or SeDuMi, MOSEK, etc., depending on your CVX setup

P_slices = cell(1,r);
status   = strings(1,r);
objval   = nan(1,r);

for q = 1:2
    X11 = X1hat(:,:,q);                     % n x nT
    X21 = X2hat(:,:,q);                     % n x nT
    Q11 = Qhat(:,:,q);   % enforce Hermitian numerically
    R11 = Rhat(:,:,q);
    U11 = Uhat(:,:,q);
    cvx_begin sdp
        variable P(n,n) hermitian
        % LMI in the slice (dimensions: nT x nT)
        L = X11' * P * X11 - X21' * P * X21 ...
            - X11' * Q11 * X11 - U11' * R11 * U11;

        maximize( trace(P) )
        subject to
            P >= 0;
            L <= 0;
    cvx_end

    P_slices{q} = P;
    status(q)   = string(cvx_status);
    objval(q)   = cvx_optval;
end

fprintf('Solved %d Fourier-slice SDPs (t-product only).\n', r);
disp(table((1:r).', status.', objval.', 'VariableNames', {'slice','status','traceP'}));
%%
clc;

% Load video data (expects X: 20x20x20)
load('squares.mat','X');
X = double(X);
[n, W, F] = size(X);
assert(all([n W F] == [20 20 20]), 'Expected X to be 20x20x20.');

%% Build X1, X2 by concatenating along mode-2 after reshaping each frame to 20x1x20
% Map columns to tube dim via permute; then take frames 1..19 and 2..20 along mode-2
% Result: X1, X2 in R^{20 x 19 x 20}
Xperm = permute(X, [1 3 2]);   % n x F x W  (rows x frames x width->tubes)
X1    = Xperm(:, 1:F-1, :);    % 20 x 19 x 20
X2    = Xperm(:, 2:F,   :);    % 20 x 19 x 20

nT = size(X1,2);               % 19  (second mode)
r  = size(X1,3);               % 20  (tube length / FFT length)
epsilon = 1e-9;                % PD margin

%% Build U (time domain) and Qhat (Fourier domain)
% U: random input, then FFT -> Uhat  (size n x nT x r)
U    = randn(n, nT, r);
Uhat = fft(U, [], 3);

% Qhat: per-slice PSD (n x n) in Fourier domain
Qhat = zeros(n,n,r);
for q = 1:r
    Ae = randn(n,n);
    Qhat(:,:,q) = Ae*Ae';                % Hermitian PSD
end

%% Storage
Thetas     = cell(1,r);                  % cell per slice
Theta_hat  = complex(zeros(nT, n, r));   % stacked spectral solutions
status     = strings(1,r);
times      = zeros(1,r);

cvx_quiet true
% cvx_solver SDPT3
% cvx_precision low   % you said high accuracy not needed

for q = 1:r
    tic;

    % (As requested) compute FFTs of X1, X2 inside the loop
    a1  = fft(X1, [], 3);    X11 = a1(:,:,q);        % n x nT (complex)
    a2  = fft(X2, [], 3);    X21 = a2(:,:,q);        % n x nT (complex)
    U11 = Uhat(:,:,q);                               % n x nT
    Q11 = (Qhat(:,:,q) + Qhat(:,:,q)')/2;            % Hermitianize numerically

    % Slice SDP
    cvx_begin sdp
        variable Theta(nT, n) complex

        X1Theta = X11 * Theta;           % n x n
        X2Theta = X21 * Theta;           % n x n

        minimize(0)
        subject to
            % Make the block matrix Hermitian by construction
            X1Theta == X1Theta';
            M = [X1Theta,  X2Theta;
                 X2Theta', X1Theta];
            M == M';                               % Hermitian
            M >= epsilon * eye(2*n);               % PD via epsilon

            % Additional constraints
            U11 * Theta      == 0;                 % n x n zero
            Q11 * X2Theta    == 0;                 % n x n zero
    cvx_end

    Thetas{q}        = Theta;
    Theta_hat(:,:,q) = Theta;
    status(q)        = string(cvx_status);
    times(q)         = toc;

    fprintf('Slice %2d/%2d | status: %s | time: %.3fs\n', q, r, cvx_status, times(q));
end

%% Also provide time-domain Theta tensor along tube dim
Theta_time = ifft(Theta_hat, [], 3);   % nT x n x r (complex)

%% Save everything
save('theta_solutions.mat', 'Thetas', 'Theta_hat', 'Theta_time', 'status', 'times', ...
     'n', 'nT', 'r', 'epsilon');

% Summary
disp(table((1:r).', status.', times.', 'VariableNames', {'slice','status','time_sec'}));


%% ---------- Recover K from solved S (Theta): K_q = V_q*S_q*(Y_q*S_q)^{-1} ----------
% Inputs required in workspace:
%   X1  -> Y  (size n x nT x r)
%   U   -> V  (size m x nT x r)
%   Theta_hat (nT x n x r)   OR   Thetas{q} each nT x n
% Optional: epsilon (regularization for (Y_q*S_q)^{-1})

if ~exist('epsilon','var'), epsilon = 1e-8; end

% If you saved Thetas in a cell, pack them into Theta_hat
if ~exist('Theta_hat','var') && exist('Thetas','var')
    r = numel(Thetas);
    [nT,n] = size(Thetas{1});
    Theta_hat = zeros(nT,n,r);
    for q = 1:r
        Theta_hat(:,:,q) = Thetas{q};
    end
end

[n, nT, r] = size(X1);
m          = size(U,1);

% FFT along mode-3 to decouple the problem
Yhat = fft(X1, [], 3);   % n x nT x r
Vhat = fft(U , [], 3);   % m x nT x r

% Allocate K in the frequency domain
Khat = complex(zeros(m, n, r));

for q = 1:r
    Yq = Yhat(:,:,q);           % n x nT
    Vq = Vhat(:,:,q);           % m x nT
    Sq = Theta_hat(:,:,q);      % nT x n

    YS = Yq * Sq;               % n x n

    % Regularized inverse for numerical robustness
    Khat(:,:,q) = Vq * (Sq / (YS + epsilon*eye(n)));
    % (Alternative)   Khat(:,:,q) = Vq * Sq * pinv(YS);
end

% Back to time/tube domain
K = ifft(Khat, [], 3);          % m x n x r
K = real(K);                    % data are real -> drop tiny imag parts

% (optional) save
save('tqr_gain_K.mat','K','Khat');

%% ======================= HAAR (pairwise) =======================
% Forward Haar (pairwise) along mode-3
Yh = fast_bdwt_haar_blocks(X1);   % n x nT x r
Zh = fast_bdwt_haar_blocks(X2);
Vh = fast_bdwt_haar_blocks(U);

% (Optional) Q in Haar domain (if you enforce the extra constraints)
if exist('Qhat','var')
    Q_time = ifft(Qhat, [], 3);                 % n x n x r
    Qh     = fast_bdwt_haar_blocks(Q_time);     % n x n x r
end

% Storage (mirrors your FFT section)
S_haar      = cell(1,r);                  % cell per slice
S_haar_hat  = complex(zeros(nT, n, r));   % stacked spectral solutions
status_haar = strings(1,r);
times_haar  = zeros(1,r);
Khat_haar   = complex(zeros(n, n, r));    % (here m=n in your setup)

cvx_quiet true
% cvx_solver SDPT3

for z = 1:r
    tic;

    Yz = Yh(:,:,z);   Zz = Zh(:,:,z);   Vz = Vh(:,:,z);
    if exist('Qh','var')
        Qz = 0.5*(Qh(:,:,z) + Qh(:,:,z)');  % Hermitianize defensively
    end

    % SDP for S_z (same structure as FFT; add the two zero constraints if desired)
    cvx_begin sdp
        variable Sz(nT, n) complex
        YS = Yz * Sz;     % n x n
        ZS = Zz * Sz;     % n x n

        minimize(0)
        subject to
            YS == YS';                            % Hermitian
            M = [YS,  ZS; ZS', YS];
            M == M';
            M >= epsilon * eye(2*n);

            % --- Optional: same two constraints as FFT variant ---
            % Vz * Sz == 0;
            % Qz * ZS == 0;
    cvx_end

    % Store & report
    S_haar{z}        = Sz;
    S_haar_hat(:,:,z)= Sz;
    status_haar(z)   = string(cvx_status);
    times_haar(z)    = toc;
    fprintf('HAAR  slice %2d/%2d | status: %s | time: %.3fs\n', z, r, cvx_status, times_haar(z));

    % K_z = V_z * S_z * (Y_z S_z)^{-1}
    Khat_haar(:,:,z) = Vz * ( Sz / (YS + epsilon*eye(n)) );
end

% Inverse Haar back to tube/time domain
K_haar = fast_bdwt_haar_blocks_inv(Khat_haar);
K_haar = real(K_haar);

% Save & summary
save('tqr_gain_K_haar.mat','K_haar','Khat_haar','S_haar','S_haar_hat','status_haar','times_haar');
disp(table((1:r).', status_haar.', times_haar.', 'VariableNames', {'slice','status','time_sec'}));

%% ========================= DCT (cosine) =========================
% If Qhat was specified per FFT-slice, bring to time then to cosine domain
if exist('Qhat','var')
    Q_time = ifft(Qhat, [], 3);         % n x n x r
    Qc     = cosine_transform(Q_time);  % n x n x r
end

Yc = cosine_transform(X1);              % n x nT x r
Zc = cosine_transform(X2);
Vc = cosine_transform(U);

% Storage
S_dct      = cell(1,r);
S_dct_hat  = complex(zeros(nT, n, r));
status_dct = strings(1,r);
times_dct  = zeros(1,r);
Kc_hat     = complex(zeros(n, n, r));

cvx_quiet true
% cvx_solver SDPT3

for y = 1:r
    tic;

    Yy = Yc(:,:,y);  Zy = Zc(:,:,y);  Vy = Vc(:,:,y);
    if exist('Qc','var')
        Qy = 0.5*(Qc(:,:,y) + Qc(:,:,y)');  % Hermitianize
    end

    cvx_begin sdp
        variable Sy(nT, n) complex
        YS = Yy * Sy;     % n x n
        ZS = Zy * Sy;     % n x n

        minimize(0)
        subject to
            YS == YS';
            M = [YS,  ZS; ZS', YS];
            M == M';
            M >= epsilon * eye(2*n);

            % --- Optional: same two constraints as FFT variant ---
            % Vy * Sy == 0;
            % Qy * ZS == 0;
    cvx_end

    % Store & report
    S_dct{y}        = Sy;
    S_dct_hat(:,:,y)= Sy;
    status_dct(y)   = string(cvx_status);
    times_dct(y)    = toc;
    fprintf('DCT   slice %2d/%2d | status: %s | time: %.3fs\n', y, r, cvx_status, times_dct(y));

    % K_y = V_y * S_y * (Y_y S_y)^{-1}
    Kc_hat(:,:,y) = Vy * ( Sy / (YS + epsilon*eye(n)) );
end

% Inverse: if you have the exact inverse of your cosine_transform, use it.
% Otherwise, approximate with an inverse DCT along mode-3:
try
    K_dct = inverse_cosine_transform(Kc_hat);  % your exact inverse, if available
catch
    K_dct = idct(Kc_hat, [], 3);               % fast approximation
end
K_dct = real(K_dct);

% Save & summary
save('tqr_gain_K_dct.mat','K_dct','Kc_hat','S_dct','S_dct_hat','status_dct','times_dct');
disp(table((1:r).', status_dct.', times_dct.', 'VariableNames', {'slice','status','time_sec'}));

%% ===============================================================
%  DWT (Haar) & DCT variants: solve Theta-LMI and check stability
%  Uses already-loaded A, B, Q, R, X1, X2, U
%  ---------------------------------------------------------------
% Required variables in workspace: A(nxnxr), B(nxpxr), X1(n x m*nT x r), X2(n x m*nT x r), U(p x m*nT x r)
% Optional: Q(nxnxr), R(p x pxr)
% Notes:
%   - We do NOT enforce U*Theta==0 (to allow nonzero K)
%   - Q-constraints are commented as optional (can tighten if desired)

clearvars -except A B Q R X1 X2 U; clc;

%% Dimensions + settings
[n, n2, r] = size(A);      assert(n==n2, 'A must be n x n x r');
p = size(B,2);
nTeff = size(X1,2);        % = m*nT
epsilon = 1e-9;
USE_U_EQ_ZERO = false;     % keep false to allow nonzero K
fprintf('Sizes: n=%d, p=%d, r=%d, snapshot width=%d\n', n, p, r, nTeff);

%% ========================= DCT (cosine) =========================
fprintf('\n=== DCT / Cosine variant ===\n');

% Cosine-domain data
Yc = cosine_transform(X1);              % n x nTeff x r
Zc = cosine_transform(X2);
Vc = cosine_transform(U);

% Transform A,B for stability check
Ac = cosine_transform(A);               % n x n x r
Bc = cosine_transform(B);               % n x p x r

% (Optional) Q in cosine domain
if exist('Q','var')
    Qc = cosine_transform(Q);           % n x n x r
end

% Storage
Sc_cell    = cell(1,r);
Sc_hat     = complex(zeros(nTeff, n, r));
Kc_hat     = complex(zeros(p    , n, r));
status_dct = strings(1,r);

cvx_quiet true
for y = 1:r
    Yy = Yc(:,:,y);  Zy = Zc(:,:,y);  Vy = Vc(:,:,y);

    cvx_begin sdp
        variable Sy(nTeff, n) complex
        YS = Yy * Sy;     % n x n
        ZS = Zy * Sy;     % n x n

        minimize(0)
        subject to
            YS == YS';
            M = [YS, ZS; ZS', YS];
            M == M';
            M >= epsilon * eye(2*n);

            % --- Optional cosine-domain constraints (uncomment if desired) ---
            % if exist('Qc','var'); Qy = 0.5*(Qc(:,:,y)+Qc(:,:,y)'); Qy*ZS == 0; end
            % if USE_U_EQ_ZERO;      Vy * Sy == 0; end
    cvx_end

    status_dct(y) = string(cvx_status);
    fprintf('DCT  slice %d/%d: %s\n', y, r, cvx_status);

    if contains(cvx_status,'Solved')
        Kc_hat(:,:,y) = Vy * ( Sy / (YS + epsilon*eye(n)) );
    else
        Kc_hat(:,:,y) = zeros(p,n);
    end

    Sc_cell{y}    = Sy;
    Sc_hat(:,:,y) = Sy;
end

% Stability check in cosine domain
rho_open_c = zeros(1,r);
rho_cl_c   = zeros(1,r);
for y = 1:r
    rho_open_c(y) = max(abs(eig(Ac(:,:,y))));
    rho_cl_c(y)   = max(abs(eig(Ac(:,:,y) + Bc(:,:,y)*Kc_hat(:,:,y))));
end
fprintf('DCT  open-loop  spectral radii : %s\n', mat2str(rho_open_c,3));
fprintf('DCT  closed-loop spectral radii : %s\n', mat2str(rho_cl_c,3));
if all(rho_cl_c < 1-1e-12)
    disp('DCT: All slices Schur-stable.');
else
    warning('DCT: Some slices are not strictly inside the unit circle.');
end

%% (Optional) Pack results to .mat (only if you want to inspect later)
% save('cvx_dwt_dct_results.mat', 'Kh_hat','Sh_hat','status_haar','rho_open_h','rho_cl_h', ...
%                                  'Kc_hat','Sc_hat','status_dct','rho_open_c','rho_cl_c', '-v7.3');
