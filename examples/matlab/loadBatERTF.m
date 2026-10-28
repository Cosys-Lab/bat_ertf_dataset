function structERTF = loadBatERTF( folderSpecies, sourceNames, frameName )
% Load the transfer functions of one species folder of the dataset (ertf_<frame>.h5 and landmarks.json)
%   folderSpecies: species folder of the dataset, e.g. '...\Myotis_daubentonii'
%   sourceNames: cell array of sources, default { 'left', 'right', 'emitter' } (alternatives: 'left_alt1', ...)
%   frameName: 'ct' (default), or 'reference' for the frame of De Mey et al. (P. discolor only)
%   structERTF.freqsHz [nFreqs x 1], .azVec, .elVec [deg], .H.<source> complex [nEl x nAz x nFreqs],
%   .landmarks (ear_left, ear_right, nostrils or mouth; metres, frame of the file), .meta (landmarks.json)

if( nargin < 2 || isempty( sourceNames ) )
    sourceNames = { 'left', 'right', 'emitter' };
end
if( nargin < 3 || isempty( frameName ) )
    frameName = 'ct';
end
if( ischar( sourceNames ) )
    sourceNames = { sourceNames };
end

fileName = fullfile( folderSpecies, [ 'ertf_' frameName '.h5' ] );
structERTF.fileName = fileName;
structERTF.freqsHz = double( h5read( fileName, '/freqs' ) );
structERTF.azVec = double( h5read( fileName, '/azimuth' ) );
structERTF.elVec = double( h5read( fileName, '/elevation' ) );
structERTF.freqsHz = structERTF.freqsHz( : );
structERTF.azVec = structERTF.azVec( : );
structERTF.elVec = structERTF.elVec( : );

% h5py stores complex numbers as a compound { r, i }, and MATLAB reverses the dimension order: [ f, az, el ]
for cntSource = 1 : length( sourceNames )
    dataH5 = h5read( fileName, [ '/' sourceNames{ cntSource } '/H' ] );
    if( isstruct( dataH5 ) )
        dataH5 = complex( double( dataH5.r ), double( dataH5.i ) );
    end
    structERTF.H.( sourceNames{ cntSource } ) = permute( double( dataH5 ), [ 3 2 1 ] );
end

% Landmarks and metadata
fileJson = fullfile( folderSpecies, 'landmarks.json' );
structERTF.landmarks = struct( );
if( isfile( fileJson ) )
    structERTF.meta = jsondecode( fileread( fileJson ) );
    structERTF.landmarks = structERTF.meta.frames.( frameName ).landmarks_m;
    structERTF.species = structERTF.meta.taxon;
end

end
