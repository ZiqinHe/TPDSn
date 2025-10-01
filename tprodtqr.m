clear
clc
epsilon = 1e-3; % Small positive constant
n = 2;
nT = 4;
T = nT/n;
% Define the matrices
n1 = 10;
ite = 5;
t = zeros(ite,min(5,n1));
t1 = zeros(ite,n1);
t2 = zeros(ite,n1);
t3 = zeros(ite,n1);
for j = 1:n1
for g = 1:ite

r = 2^j;
U = rand(n,nT,r);
x0 = rand(n,n,r);
Qfourier = zeros(n,n,r);
Rfourier = zeros(n,n,r);
for e = 1:r
    Ae = rand(n,n);
    Be = rand(n,n);
    Qfourier(:,:,e) = Ae*Ae';
    Rfourier(:,:,e) = Be*Be' + epsilon*eye(n,n);
end
Q = ifft(Qfourier, [], 3);
R = ifft(Rfourier, [], 3);
% Dimensions and initial state matrices

% Initialize X1 and X2
X1 = randn(n, nT,r);
X2 = zeros(n, nT,r);
% Populate X1 and X2 based on the system dynamics
for i = 2:T
    X2(:,(i-2)*n+1:(i-1)*n ,:) = X1(:, (i-1)*n+1:i*n ,:);
end
%method1
X2(:,(T-1)*n+1:T*n ,:) = randn(n,n,r);
if j < 6
tic
X1c = bcirc(X1);
X2c = bcirc(X2);
Qc = bcirc(Q);
Rc = bcirc(R);
Uc = bcirc(U);
% Use CVX to define and solve the LMI constraints
epsilon = 1e-3; % Small positive constant
cvx_begin sdp
    variable P(n, n, r) symmetric % Define Theta as a T x n matrix variable
    BP = bcirc(P);
    L = X1c'*BP * X1c - X2c'*BP*X2c - X1c'*Qc*X1c- Uc'*Rc*Uc;

     % Objective: Maximize trace(P)
    maximize(trace(BP))

    % LMI constraints
    BP =BP';
    BP >= 0;
    L <= 0;
cvx_end

% Display the solution for Theta if feasible
if strcmp(cvx_status, 'Solved')
    %P_full = full(P); % Convert Theta to a full matrix to see its values
    disp('Solution for Theta found:');
    %disp(P_full);
else
    disp('No feasible solution found.');
end
t(g,j) = toc
end

tic
a1 = fft(X1,[],3);
a2 = fft(X2,[],3);
a3 = fft(Q,[],3);
a4 = fft(U,[],3);
a5 = fft(R,[],3);
for q = 1:r
    X11 = a1(:,:,q);
    X21 = a2(:,:,q);
    Q11 = a3(:,:,q);
    U11 = a4(:,:,q);
    R11 = a5(:,:,q);
    cvx_begin sdp
    variable P(n,n) symmetric% Define Theta as a T x n matrix variable
    % Define the symmetric matrix constraint
    L = X11'*P*X11 - X21'*P*X21 - X11'*Q11*X11 - U11'*R11*U11;

    % Objective: Minimize a placeholder objective (since we're only interested in feasibility)
    maximize(trace(P))

    % LMI constraints
    P == P';
    P >= 0;
    L <= 0;
    cvx_end

% Display the solution for Theta if feasible
if strcmp(cvx_status, 'Solved')
    %Theta_full = full(Theta); % Convert Theta to a full matrix to see its values
    disp('Solution for Theta found:');
    %disp(P)
else
    disp('No feasible solution found.');
end
end
t1(g,j) = toc

tic
X1_haar = fast_bdwt_haar_blocks(X1);
X2_haar = fast_bdwt_haar_blocks(X2);
Q_haar  = fast_bdwt_haar_blocks(Q);
R_haar  = fast_bdwt_haar_blocks(R);
U_haar  = fast_bdwt_haar_blocks(U);

for z = 1:r
    X1z = X1_haar(:,:,z);
    X2z = X2_haar(:,:,z);
    Qz  = Q_haar(:,:,z);
    Rz  = R_haar(:,:,z);  % or block: R_haar(rows, rows) if needed
    Uz  = U_haar(:,:,z);  % same idea

    cvx_begin sdp
        variable P(n,n) symmetric
        L = X1z'*P*X1z - X2z'*P*X2z - X1z'*Qz*X1z - Uz'*Rz*Uz;
        maximize(trace(P))
        P == P';
        P >= 0;
        L <= 0;
    cvx_end

    if strcmp(cvx_status, 'Solved')
        disp(['Haar solution at z = ', num2str(z), ' found.']);
    else
        disp(['No Haar solution at z = ', num2str(z)]);
    end
end
t2(g,j) = toc

tic
X1_cos = cosine_transform(X1);
X2_cos = cosine_transform(X2);
Q_cos  = cosine_transform(Q);
R_cos  = cosine_transform(R);
U_cos  = cosine_transform(U);

for y = 1:r
    X1y = X1_cos(:,:,y);
    X2y = X2_cos(:,:,y);
    Qy  = Q_cos(:,:,y);
    Ry  = R_cos(:,:,y);
    Uy  = U_cos(:,:,y);

    cvx_begin sdp
        variable P(n,n) symmetric
        L = X1y'*P*X1y - X2y'*P*X2y - X1y'*Qy*X1y - Uy'*Ry*Uy;
        maximize(trace(P))
        P == P';
        P >= 0;
        L <= 0;
    cvx_end

    if strcmp(cvx_status, 'Solved')
        disp(['Cosine solution at y = ', num2str(y), ' found.']);
    else
        disp(['No cosine solution at y = ', num2str(y)]);
    end
end
t3(g,j) = toc

end
end
%%
% Column-wise mean
mean_vals = mean(t, 1);

% Column-wise standard deviation
std_vals = std(t, 0, 1);

% Display results
disp('Column-wise mean:');
disp(mean_vals);

disp('Column-wise std:');
disp(std_vals);
%%
% Column-wise mean
mean_vals = mean(t1, 1);

% Column-wise standard deviation
std_vals = std(t1, 0, 1);

% Display results
disp('t-product Column-wise mean:');
disp(mean_vals);

disp('t-product Column-wise std:');
disp(std_vals);
%%
% Column-wise mean
mean_vals = mean(t2, 1);

% Column-wise standard deviation
std_vals = std(t2, 0, 1);

% Display results
disp('haar wavelet transform Column-wise mean:');
disp(mean_vals);

disp('haar wavelet transform Column-wise std:');
disp(std_vals);
%%
% Column-wise mean
mean_vals = mean(t3, 1);

% Column-wise standard deviation
std_vals = std(t3, 0, 1);

% Display results
disp('cosine transform Column-wise mean:');
disp(mean_vals);

disp('cosine transform Column-wise std:');
disp(std_vals);