% Two-way ERTF of the left ear in the head frame and with the head rotated (yaw to the left, pitch nose down).
% The rotation matrix holds the head axes in the world frame (columns), so a world direction u_w is the head
% direction u_h = R' * u_w, or for row vectors u_h = u_w * R (as ERTF.rotate in batertf/dataset.py).
% Set folderSpecies to a species folder of the dataset before running, or see findSpeciesFolder.

%% Settings
if( ~exist( 'folderSpecies', 'var' ) || isempty( folderSpecies ) )
    folderSpecies = findSpeciesFolder( );
end
angleYaw = 30;
anglePitch = 15;
freqShow = 60e3;
rangeDb = 30;

%% Rotation and directions on a world grid
rotationYaw = [ cosd( angleYaw ) -sind( angleYaw ) 0 ; sind( angleYaw ) cosd( angleYaw ) 0 ; 0 0 1 ];
rotationPitch = [ cosd( anglePitch ) 0 sind( anglePitch ) ; 0 1 0 ; -sind( anglePitch ) 0 cosd( anglePitch ) ];
rotationHead = rotationYaw * rotationPitch;
azVec = -180 : 2 : 178;
elVec = -90 : 2 : 90;
[ azMesh, elMesh ] = meshgrid( azVec, elVec );
dirWorld = calculateUnitVectors( azMesh( : ), elMesh( : ) );

%% Calculate and plot
structERTF = loadBatERTF( folderSpecies );
titleMaps = { 'Head frame', sprintf( 'Yaw %g deg, pitch %g deg', angleYaw, anglePitch ) };
allDirHead = { dirWorld, dirWorld * rotationHead };
figure( 'Position', [ 100 100 1000 400 ] );
for cntMap = 1 : 2
    dataTwoWay = interpolateERTF( structERTF, 'emitter', allDirHead{ cntMap }, freqShow ) .* ...
                 interpolateERTF( structERTF, 'left', allDirHead{ cntMap }, freqShow );
    mapDb = reshape( 20 * log10( abs( dataTwoWay ) / max( abs( dataTwoWay ) ) ), size( azMesh ) );
    subplot( 1, 2, cntMap );
    imagesc( azVec, elVec, mapDb, [ -rangeDb 0 ] );
    axis xy equal tight;
    set( gca, 'XDir', 'reverse' );
    xlabel( 'Azimuth (deg)' );
    ylabel( 'Elevation (deg)' );
    title( [ 'Two-way, left ear: ' titleMaps{ cntMap } ] );
end
sgtitle( sprintf( '%s, %g kHz', structERTF.species, freqShow / 1e3 ) );
