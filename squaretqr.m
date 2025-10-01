%% ===============================================================
%  5x5x3 RGB t-product pipeline with RANDOM SHAPES in data
%    1) Build stable A,B,Q (per-slice) and real time-domain tensors
%    2) Generate data via X_{t+1} = tprod(A,X_t) + tprod(B,U_t)
%         - Each step picks a random bright target D_t
%         - U_t is solved in FFT domain by ridge LS per slice
%    3) Solve Θ–LMI per slice in FFT domain -> K
%    4) Closed-loop future: X_{t+1} = tprod(A + tprod(B,K), X_t)
%    5) Plots: open-loop frames, closed-loop frames, norm decay
% ===============================================================
clear; clc; close all; rng(7);

% ----------------------------- sizes -----------------------------
n   = 5;          % rows
m   = 5;          % cols
r   = 3;          % channels (tube length)
p   = n;          % input rows; choose p=n so B,K are 5x5 per slice
T   = 16;         % # data frames (t=0..T-1) -> nT transitions
nT  = T - 1;      
nTeff = m*nT;     % snapshot width after stacking along mode-2
epsilon = 1e-9;   % LMI PD margin & inverse regularizer
USE_U_EQ_ZERO = false;  % false to allow nonzero K
lambda_ridge  = 1e-6;   % ridge for per-slice LS when generating U_t

%% ---------------------- stable Ahat, Bhat, Qhat (diversified) -------------
Ahat = zeros(n,n,r);
Bhat = zeros(n,p,r);
Qhat = zeros(n,n,r);

% --- slice 1: stable, symmetric ---
[U1,~] = qr(randn(n));
evals1  = 0.25 + 0.65*rand(n,1);          % eigenvalues in (0,1)
A1      = U1*diag(evals1)*U1';  A1 = (A1 + A1')/2;
B1      = A1 * randn(n,p);

% --- slice 2: a different stable slice ---
[U2,~] = qr(randn(n));
evals2  = 0.25 + 0.65*rand(n,1);
A2      = U2*diag(evals2)*U2';  A2 = (A2 + A2')/2;
B2      = A2 * randn(n,p);

% --- assign slices; copy slice 2 -> slice 3 to keep time-domain real ---
Ahat(:,:,1) = A1;  Bhat(:,:,1) = B1;
Ahat(:,:,2) = A2;  Bhat(:,:,2) = B2;
Ahat(:,:,3) = A2;  Bhat(:,:,3) = B2;

% Q per slice: projector onto (col([Aq Bq]))^⊥  so Q_q A_q = Q_q B_q = 0
for q = 1:r
    Mq = [Ahat(:,:,q), Bhat(:,:,q)];   % n x (n+p)
    [U,S,~] = svd(Mq,'econ');
    rk = rank(S, 1e-10*norm(S,2));
    if rk < n
        Uperp = U(:, rk+1:end);
        Qq = Uperp * Uperp.';
    else
        Qq = zeros(n);
    end
    Qhat(:,:,q) = 0.5*(Qq + Qq');      % symmetrize
end

% Back to time/tube domain (real)
A = real(ifft(Ahat,[],3));   % n x n x r
B = real(ifft(Bhat,[],3));   % n x p x r
Q = real(ifft(Qhat,[],3));

% ---------------------- data generation (open loop) ----------------
% X_{t+1} = tprod(A, X_t) + tprod(B, U_t)
X = zeros(n,m,r,T);
U = zeros(p,m,r,T-1);

% First RGB frame (truly color)
X(:,:,:,1) = rand_shape_rgb_color(n,m,r, 1.0);

for t = 1:nT
    D_t = rand_shape_rgb_color(n,m,r, 1.0);
    U(:,:,:,t)   = track_one_step(A, B, X(:,:,:,t), D_t, lambda_ridge);
    X(:,:,:,t+1) = tprod(A, X(:,:,:,t)) + tprod(B, U(:,:,:,t));
end

% ---------------------- build snapshots X1, X2, V -----------------
X1 = zeros(n, nTeff, r);
X2 = zeros(n, nTeff, r);
V  = zeros(p, nTeff, r);
for t = 1:nT
    idx = (t-1)*m + (1:m);
    X1(:, idx, :) = X(:,:,:, t);
    X2(:, idx, :) = X(:,:,:, t+1);
    V (:, idx, :) = U(:,:,:, t);
end

% FFT along tubes (decouples slices)
Yhat1 = fft(X1, [], 3);        % n x nTeff x r
Yhat2 = fft(X2, [], 3);        % n x nTeff x r
Vhat  = fft(V , [], 3);        % p x nTeff x r

% ---------------------- Θ–LMI per slice -> Khat -------------------
cvx_quiet true
Theta_hat = complex(zeros(nTeff, n, r));
Khat      = complex(zeros(p    , n, r));
status    = strings(1,r);

fprintf('--- CVX solve per slice ---\n');
for q = 1:r
    X11 = Yhat1(:,:,q);
    X21 = Yhat2(:,:,q);
    U11 = Vhat(:,:,q);
    Q11 = Qhat(:,:,q);

    if norm(X11,'fro') < 1e-10 || norm(X21,'fro') < 1e-10
        warning('slice %d: snapshot energy too small -> set K_q=0', q);
        Khat(:,:,q) = zeros(p,n);
        status(q)   = "Skipped-low-energy";
        continue;
    end

    cvx_begin sdp
        variable Theta(nTeff, n) complex
        X1Theta = X11 * Theta;
        X2Theta = X21 * Theta;

        minimize(0)
        subject to
            X1Theta == X1Theta';
            M = [X1Theta,  X2Theta;
                 X2Theta', X1Theta];
            M == M';
            M >= epsilon * eye(2*n);
            Q11 * X2Theta == 0;
            if USE_U_EQ_ZERO
                U11 * Theta == 0;
            end
    cvx_end

    status(q) = string(cvx_status);
    fprintf('slice %d: %s\n', q, cvx_status);

    if contains(cvx_status,'Solved')
        YS = X11 * Theta;
        Khat(:,:,q) = U11 * (Theta / (YS + epsilon*eye(n)));
        fprintf('  ||K_q||_F = %.3e\n', norm(Khat(:,:,q),'fro'));
    else
        warning('  slice %d infeasible -> set K_q = 0', q);
        Khat(:,:,q) = zeros(p,n);
    end
end


% Back to time/tube domain
K = real(ifft(Khat, [], 3));        % p x n x r  (= 5 x 5 x 3)

% ------------------- stability check (per slice) -------------------
rho_open = zeros(1,r);
rho_cl   = zeros(1,r);
for q = 1:r
    rho_open(q) = max(abs(eig(Ahat(:,:,q))));
    rho_cl(q)   = max(abs(eig(Ahat(:,:,q) + Bhat(:,:,q)*Khat(:,:,q)*0.01)));
end
fprintf('\nOpen-loop  |A_q|           : %s\n', mat2str(rho_open,3));
fprintf('Closed-loop|A_q + B_q K_q| : %s\n', mat2str(rho_cl,3));

%% ------------------- closed-loop future frames --------------------
beta = 0.3;                 % 0<beta<=1, smaller => closer to open-loop
Acl  = A + tprod(B, beta*K);
        % autonomous left map in time domain
Tsim = 6;
Xcl  = zeros(n,m,r,Tsim+1);
Xcl(:,:,:,1) = X(:,:,:,10);  % start from last data frame

for t = 1:Tsim
    Xcl(:,:,:,t+1) = tprod(Acl, Xcl(:,:,:,t));
end

%% ------------------- Visualizations -------------------------------
% Open-loop (data) t=0..min(15,T-1)
figure('Name','Open-loop data (random shapes)'); 
for t = 0:min(15, T-1)
    subplot(4,4,t+1);
    img = X(:,:,:,t+1);
    image(min(max(img,0),1)); axis image off;
    title(sprintf('t=%d', t));
end
sgtitle('Open-loop: X_{t+1} = tprod(A, X_t) + tprod(B, U_t) (random bright shapes)');

% Closed-loop future t=1..16
figure('Name','Closed-loop future (autonomous) t=1..16');
for t = 1:Tsim
    subplot(2,3,t);
    img = Xcl(:,:,:,t);
    image(min(max(img,0),1)); axis image off;
    title(sprintf('t=%d', t));
end
Acl_hat = fft(Acl,[],3);
rho_acl = max(abs(eig(Acl_hat(:,:,1))));
sgtitle(sprintf('Closed-loop: X_{t+1} = tprod(A+BK, X_t)  (spectral radius %.3f)', rho_acl));

% Norm decay
nf = zeros(1,Tsim+1);
for t = 1:Tsim+1
    nf(t) = norm(reshape(Xcl(:,:,:,t),[],1),2);
end
figure('Name','Closed-loop Frobenius norm decay');
semilogy(0:Tsim, nf/nf(1), 'o-','LineWidth',2); grid on;
xlabel('t'); ylabel('||X_t||_F / ||X_0||_F');
title('Decay under autonomous t-product closed loop');


%%
% pick the open-loop frame with the largest energy as X0
[~,t0] = max( vecnorm(reshape(X, [], size(X,4)), 2, 1) );
Xcl = zeros(n,m,r,Tsim+1);
Xcl(:,:,:,1) = X(:,:,:,t0);


% rollout (already done) …

% fixed scale based on the initial closed-loop frame
M0 = max(abs(Xcl(:,:,:,1)), [], 'all');  if M0==0, M0=1; end
figure('Name','Closed-loop (fixed scale)'); colormap(gray(256));
for t=1:Tsim
    subplot(4,4,t);
    imagesc(abs(Xcl(:,:,:,t)), [0 M0]); axis image off;
    title(sprintf('t=%d',t));
end
sgtitle('Fixed scale: true decay to black');

% per-frame normalization just for visualization
figure('Name','Closed-loop (normalized per frame)'); colormap(gray(256));
for t=1:Tsim
    subplot(4,4,t);
    F = abs(Xcl(:,:,:,t));
    m = max(F,[],'all'); if m==0, m=1; end
    imagesc(F/m, [0 1]); axis image off;
    title(sprintf('t=%d',t));
end
sgtitle('Per-frame normalized: structure visible while magnitude shrinks');


