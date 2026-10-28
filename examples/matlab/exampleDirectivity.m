% Directivity maps (azimuth/elevation) of the left ear, the emitter and the two-way product at one frequency.
% Set folderSpecies to a species folder of the dataset before running, or see findSpeciesFolder.
% The SOFA files can be read with the SOFA toolbox instead:
%   Obj = SOFAload( fullfile( folderSpecies, 'ears_ct.sofa' ) ); H = complex( Obj.Data.Real, Obj.Data.Imag );

%% Settings
if( ~exist( 'folderSpecies', 'var' ) || isempty( folderSpecies ) )
    folderSpecies = findSpeciesFolder( );
end
freqShow = 60e3;
rangeDb = 30;

%% Load the data
structERTF = loadBatERTF( folderSpecies );
[ ~, idxFreq ] = min( abs( structERTF.freqsHz - freqShow ) );

%% Plot the left ear, the emitter and the two-way ERTF
dataMaps = { structERTF.H.left( :, :, idxFreq ), structERTF.H.emitter( :, :, idxFreq ), ...
             structERTF.H.left( :, :, idxFreq ) .* structERTF.H.emitter( :, :, idxFreq ) };
titleMaps = { 'Left ear', 'Emitter', 'Two-way (emitter x left ear)' };
figure( 'Position', [ 100 100 1400 380 ] );
for cntMap = 1 : 3
    subplot( 1, 3, cntMap );
    mapDb = 20 * log10( abs( dataMaps{ cntMap } ) / max( abs( dataMaps{ cntMap }( : ) ) ) );
    imagesc( structERTF.azVec, structERTF.elVec, mapDb, [ -rangeDb 0 ] );
    axis xy equal tight;
    hold on;
    contour( structERTF.azVec, structERTF.elVec, mapDb, [ -6 -3 ], 'w' );
    set( gca, 'XDir', 'reverse' );          % the bat's left (+ azimuth) on the left
    xlabel( 'Azimuth (deg)' );
    ylabel( 'Elevation (deg)' );
    title( titleMaps{ cntMap } );
end
colorbar;
sgtitle( sprintf( '%s, %g kHz', structERTF.species, structERTF.freqsHz( idxFreq ) / 1e3 ) );
