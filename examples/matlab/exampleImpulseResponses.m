% Impulse responses and transfer functions of both ears for one direction.
% Set folderSpecies to a species folder of the dataset before running, or see findSpeciesFolder.

%% Settings
if( ~exist( 'folderSpecies', 'var' ) || isempty( folderSpecies ) )
    folderSpecies = findSpeciesFolder( );
end
azSource = 30;
elSource = -10;
structIR.sampleRate = 500e3;
structIR.numSamples = 1024;
structIR.preDelay = 0.2e-3;

%% Load the data and calculate the responses
structERTF = loadBatERTF( folderSpecies );
dirSource = calculateUnitVectors( azSource, elSource );
earNames = { 'left', 'right' };

figure( 'Position', [ 100 100 1200 400 ] );
for cntEar = 1 : 2
    [ dataIR, timeVec ] = calculateImpulseResponse( structERTF, earNames{ cntEar }, dirSource, structIR );
    dataH = interpolateERTF( structERTF, earNames{ cntEar }, dirSource );
    subplot( 1, 2, 1 );
    plot( timeVec * 1e3, dataIR );
    hold on;
    subplot( 1, 2, 2 );
    plot( structERTF.freqsHz / 1e3, 20 * log10( abs( dataH ) ) );
    hold on;
end

%% Labels
subplot( 1, 2, 1 );
xlim( [ -0.1 0.4 ] );
grid on;
legend( earNames );
xlabel( 'Time (ms), 0 = arrival at the origin' );
ylabel( 'Impulse response' );
subplot( 1, 2, 2 );
grid on;
legend( earNames );
xlabel( 'Frequency (kHz)' );
ylabel( 'Magnitude (dB re free field at the origin)' );
sgtitle( sprintf( '%s, azimuth %g deg, elevation %g deg', structERTF.species, azSource, elSource ) );
