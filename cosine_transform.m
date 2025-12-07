function A_hat = cosine_transform(A)
% Applies the cosine-domain transform to A (along 3rd mode)
% such that A_hat = A ×_3 M, where M = W^{-1} * C * (I + Z)

[m, l, n] = size(A);

% Step 1: Construct DCT matrix C and circulant upshift Z
C = dctmtx(n);              % DCT matrix
Z = diag(ones(n-1,1),1);      % circulant upshift: 1 on superdiagonal
W = diag(C(:,1));             % W = diag(first column of C)
M = W \ (C * (eye(n) + Z));   % M = W^{-1} * C * (I + Z)

% Step 2: Mode-3 unfolding
A3 = reshape(permute(A, [3, 1, 2]), n, []);  % size n × (m*l)

% Step 3: Apply M (left-multiply)
A3_hat = M * A3;

% Step 4: Fold back to tensor
A_hat = permute(reshape(A3_hat, n, m, l), [2, 3, 1]);  % size m × l × n
end
