function D = fast_bdwt_haar_blocks(A)
%FAST_BDWT_HAAR_BLOCKS
% Compute the diagonal blocks of (H ⊗ I_n) * bdwt(A) * (H' ⊗ I_m)
% where bdwt(A) uses 2x2 symmetric blocks on adjacent slices.
%
% INPUT:
%   A : n x m x r tensor, r must be a power of 2
% OUTPUT:
%   D : n x m x r tensor in the SAME block order as haar_diag_bdwt(A)

[n,m,r] = size(A);

if mod(r,2) ~= 0
    error('r must be even to match the 2x2 symmetric bdwt pairing.');
end

% ---- Step 1: Pairwise Haar transform (sum/diff for each adjacent slice) ----
tmp = zeros(n,m,r,class(A));
isqrt2 = 1;

for p = 1:(r/2)
    X = A(:,:,2*p-1);
    Y = A(:,:,2*p);
    tmp(:,:,2*p-1) = (X + Y) * isqrt2;  % sum block
    tmp(:,:,2*p)   = (X - Y) * isqrt2;  % diff block
end

% ---- Step 2: Permute block order to match kron(H) * bdwt * kron(H') ----
D = zeros(size(tmp),class(A));

% For r=2: order is already correct
if r == 2
    D = tmp;
else
    % For r >= 4, reorder within each group of 4
    % Fast order: [avg p1, avg p2, diff p1, diff p2]
    % Target order: [avg p1, diff p1, avg p2, diff p2]
    for g = 1:4:r
        D(:,:,g  ) = tmp(:,:,g  );   % avg p1
        D(:,:,g+1) = tmp(:,:,g+2);   % diff p1
        D(:,:,g+2) = tmp(:,:,g+1);   % avg p2
        D(:,:,g+3) = tmp(:,:,g+3);   % diff p2
    end
end

end
