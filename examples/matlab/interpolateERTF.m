function dataH = interpolateERTF( structERTF, sourceName, dirVecs, freqsHz )
% Transfer function of one source at arbitrary directions and frequencies
%   structERTF: dataset of one species (loadBatERTF)
%   sourceName: 'left', 'right', 'emitter', or an alternative such as 'left_alt1'
%   dirVecs: unit vectors in the frame of the file [nDirections x 3]
%   freqsHz: frequencies [nFreqs x 1], default the grid of the file (zero outside the simulated band)
%   dataH: complex transfer function [nDirections x nFreqs]
% Bilinear in direction and linear in frequency, on the real and imaginary parts of H * exp( -j k u.p ), with p the
% source landmark: removing this delay makes the data smooth between the grid nodes (as batertf/dataset.py).

speedOfSound = 343;
posSource = calculateSourceCentre( structERTF, sourceName );
dirVecs = dirVecs ./ sqrt( sum( dirVecs .^ 2, 2 ) );
numEl = length( structERTF.elVec );
numAz = length( structERTF.azVec );

% Remove the delay of the source landmark on the grid
[ azMesh, elMesh ] = meshgrid( structERTF.azVec, structERTF.elVec );
delayMesh = cosd( elMesh ) .* cosd( azMesh ) * posSource( 1 ) + cosd( elMesh ) .* sind( azMesh ) * posSource( 2 ) + ...
            sind( elMesh ) * posSource( 3 );
waveNumbers = reshape( 2 * pi * structERTF.freqsHz / speedOfSound, 1, 1, [ ] );
dataGrid = structERTF.H.( sourceName ) .* exp( -1i * waveNumbers .* delayMesh );
dataGrid = reshape( dataGrid, numEl * numAz, [ ] );

% Bilinear interpolation on the grid, azimuth wraps around
azDirections = atan2d( dirVecs( :, 2 ), dirVecs( :, 1 ) );
elDirections = asind( max( -1, min( 1, dirVecs( :, 3 ) ) ) );
posAz = mod( azDirections - structERTF.azVec( 1 ), 360 ) / ( structERTF.azVec( 2 ) - structERTF.azVec( 1 ) );
posEl = min( max( ( elDirections - structERTF.elVec( 1 ) ) / ( structERTF.elVec( 2 ) - structERTF.elVec( 1 ) ), 0 ), numEl - 1 - 1e-9 );
idxAz0 = floor( posAz );
idxEl0 = floor( posEl );
weightAz = posAz - idxAz0;
weightEl = posEl - idxEl0;
idxAz0 = mod( idxAz0, numAz ) + 1;
idxAz1 = mod( idxAz0, numAz ) + 1;
idxEl0 = idxEl0 + 1;
idxEl1 = min( idxEl0 + 1, numEl );
dataDirections = ( 1 - weightEl ) .* ( ( 1 - weightAz ) .* dataGrid( idxEl0 + ( idxAz0 - 1 ) * numEl, : ) + ...
                                       weightAz .* dataGrid( idxEl0 + ( idxAz1 - 1 ) * numEl, : ) ) + ...
                 weightEl .* ( ( 1 - weightAz ) .* dataGrid( idxEl1 + ( idxAz0 - 1 ) * numEl, : ) + ...
                               weightAz .* dataGrid( idxEl1 + ( idxAz1 - 1 ) * numEl, : ) );

% Linear interpolation in frequency
if( nargin < 4 || isempty( freqsHz ) )
    freqsHz = structERTF.freqsHz;
else
    dataDirections = interp1( structERTF.freqsHz, dataDirections.', freqsHz( : ), 'linear', 0 ).';
end

% Restore the delay of the source landmark
dataH = dataDirections .* exp( 1i * 2 * pi * freqsHz( : ).' / speedOfSound .* ( dirVecs * posSource( : ) ) );

end


function posSource = calculateSourceCentre( structERTF, sourceName )
% Landmark of a source: ear canal entrance, mean of the nostrils, or mouth

nameBase = regexprep( sourceName, '_alt\d+$', '' );
landmarks = structERTF.landmarks;
switch nameBase
    case 'left'
        posSource = landmarks.ear_left;
    case 'right'
        posSource = landmarks.ear_right;
    case 'emitter'
        if( isfield( landmarks, 'nostrils' ) )
            posSource = mean( landmarks.nostrils, 1 );
        else
            posSource = landmarks.mouth;
        end
    otherwise
        error( 'Source not known' )
end
posSource = posSource( : ).';

end
