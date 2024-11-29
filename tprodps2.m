epsilon = 1e-3; % Small positive constant
% Define the matrices
n1 = 10;
ite = 5;
t = zeros(ite,5);
t1 = zeros(ite,n1);
for j = 1:n1
for g = 1:ite
r = 2^j;
A = rand(2,2,r);
B = rand(2,2,r);
U = rand(2,4,r);
x0 = rand(2,2,r);
Qfourier = zeros(2,2,r);
Rfourier = zeros(2,2,r);
for e = 1:r
    Ae = rand(2,2);
    Be = rand(2,2);
    Qfourier(:,:,e) = Ae*Ae';
    Rfourier(:,:,e) = Be*Be' + epsilon*eye(2,2);
end
Q = ifft(Qfourier, [], 3);
R = ifft(Rfourier, [], 3);
% Dimensions and initial state matrices
[n, ~,~] = size(A);
[~, nT,~] = size(U);
T = nT/n;

% Initialize X1 and X2
X1 = zeros(n, nT,r);
X1(:,1:n,:) = x0;
X2 = zeros(n, nT,r);

% Populate X1 and X2 based on the system dynamics
for i = 2:T
    X1(:,(i-1)*n+1:i*n ,:) = tprod(A,X1(:, n*(i-2)+1:(i-1)*n,:)) + tprod(B ,U(:,  n*(i-2)+1:(i-1)*n,:));
    X2(:,(i-2)*n+1:(i-1)*n ,:) = X1(:, (i-1)*n+1:i*n ,:);
end
%method1
X2(:,(T-1)*n+1:T*n ,:) = tprod(A, X1(:,(T-1)*n+1:T*n ,:)) + tprod(B, U(:, (T-1)*n+1:T*n ,:));
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
for q = 1:r
    tic
    a1 = fft(X1,[],3);
    X11 = a1(:,:,q);
    a2 = fft(X2,[],3);
    X21 = a2(:,:,q);
    a3 = fft(Q,[],3);
    Q11 = a3(:,:,q);
    a4 = fft(U,[],3);
    U11 = a4(:,:,q);
    a5 = fft(R,[],3);
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
t1(g,j) = t1(g,j) +toc
end
end
end
%%
for q = 1:r
    tic
    a1 = fft(X1,[],3);
    X11 = a1(:,:,q);
    a2 = fft(X2,[],3);
    X21 = a2(:,:,q);
    cvx_begin sdp
    variable Theta(nT,n) % Define Theta as a T x n matrix variable
    % Define the symmetric matrix constraint
    X1Theta = X11 * Theta;
    X2Theta = X21 * Theta;

    % Objective: Minimize a placeholder objective (since we're only interested in feasibility)
    minimize(0)

    % LMI constraints
    X1Theta == X1Theta'; % Symmetry constraint on X1 * Theta
    [X1Theta, X2Theta; X2Theta', X1Theta] >= epsilon * eye(2 * n); % Positive definite constraint with small epsilon
cvx_end

% Display the solution for Theta if feasible
if strcmp(cvx_status, 'Solved')
    %Theta_full = full(Theta); % Convert Theta to a full matrix to see its values
    disp('Solution for Theta found:');
    disp(Theta_full);
else
    disp('No feasible solution found.');
end
t1(q,j) = t1(q,j) +toc
end
%%
set(gca, 'FontSize', 14);
width = 1000;  % Width of the figure in pixels
height = 300; % Height of the figure in pixels
set(gcf, 'Position', [100, 100, width, height]);

y1 = 1:n1;
R1 = 2.^y1;
y = 1:5;
R = 2.^y;
T = mean(t);
T1 = mean(t1);
variance_t = var(t)
variance_t1 = var(t1)
% First figure: loglog plot
subplot(1,3,1);  % Create the first subplot
loglog(R, T, 'b-*', R1, T1, 'r-*');
xlabel('third mode dimension: r', 'FontSize', 16);
ylabel('time', 'FontSize', 16);
title('Time vs Third Mode Dimension', 'FontSize', 16);
% Second figure: ratio of time and r
a = T ./ R.^3*500;
a1 = T1 ./ R1;

subplot(1,3,2);  % Create the second subplot
semilogx(R(1:end), a(1:end), 'b-*', R1(1:end), a1(1:end), 'r-*');
xlabel('third mode dimension: r', 'FontSize', 16);
ylabel('ratio of time to r', 'FontSize', 16);
title('Ratio of Time to Third Mode Dimension', 'FontSize', 16);

subplot(1,3,3);
semilogy(R, variance_t, 'b-*',R1, variance_t1, 'r-*');
xlabel('third mode dimension: r','FontSize', 16);
ylabel('Variance of time','FontSize', 16);
title('Variance of unfolding method','FontSize', 16);