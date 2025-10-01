% 20x20 video, 20 frames, 5x5 square moving fast left->right
height = 20; 
width  = 20;
numFrames = 20;
squareSize = 5;
speed = 2;                  % columns per frame (tweak for faster/slower)

% Preallocate and background
X = zeros(height, width, numFrames, 'uint8');
bg = randi([0, 50], height, width, 'uint8');

% Vertical placement
rowStart = 8; 
rowEnd   = rowStart + squareSize - 1;  % 8:12

% Start left edge just off-screen so it slides in
leftEdge0 = -squareSize + 1;  % = -4 for size 5

for f = 1:numFrames
    X(:,:,f) = bg;  % static background

    leftEdge = round(leftEdge0 + (f-1)*speed);
    rightEdge = leftEdge + squareSize - 1;

    % In-bounds horizontal indices
    c1 = max(1, leftEdge);
    c2 = min(width, rightEdge);

    if c1 <= c2
        X(rowStart:rowEnd, c1:c2, f) = 255; % bright square
    end
end

% Write MP4
v = VideoWriter('squares.mp4', 'MPEG-4');
v.FrameRate = 30;
open(v);
for f = 1:numFrames
    writeVideo(v, X(:,:,f));
end
close(v);

% Save the data cube
save('squares.mat', 'X');
