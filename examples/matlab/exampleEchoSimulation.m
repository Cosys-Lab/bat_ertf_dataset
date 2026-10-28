% Echoes of three point reflectors at both ears of the bat, for a linear FM call.
% Set folderSpecies to a species folder of the dataset before running, or see findSpeciesFolder.
% The spectrograms need the Signal Processing Toolbox, the rest is base MATLAB (R2016b or newer).

%% Settings
if( ~exist( 'folderSpecies', 'var' ) || isempty( folderSpecies ) )
    folderSpecies = findSpeciesFolder( );
end
structSonar.sampleRate = 500e3;
structSonar.speedOfSound = 343;
structSonar.structAir.temperature = 25;
structSonar.structAir.humidity = 60;

% Emission: 2 ms linear FM sweep from 90 to 35 kHz with a Hann window
durationCall = 2e-3;
freqStart = 90e3;
freqStop = 35e3;
timeCall = ( 0 : round( durationCall * structSonar.sampleRate ) - 1 ) / structSonar.sampleRate;
structSonar.emissionSignal = sin( 2 * pi * ( freqStart * timeCall + ( freqStop - freqStart ) / ( 2 * timeCall( end ) ) * timeCall .^ 2 ) ) .* ...
                             ( 0.5 - 0.5 * cos( 2 * pi * timeCall / timeCall( end ) ) );

% Reflectors: x y z [m] in the head frame (X forward, Y left, Z up) and echo amplitude re 1 m
dataReflectors = [ 0.60  0.15  0.00 0.05 ;
                   0.90 -0.30  0.10 0.10 ;
                   1.40  0.00 -0.20 0.30 ];

%% Simulate the echoes
structERTF = loadBatERTF( folderSpecies );
[ dataEchoes, timeVec ] = simulateEchoes( structERTF, structSonar, dataReflectors );

%% Plot
earNames = { 'Left ear', 'Right ear' };
figure( 'Position', [ 100 100 1200 600 ] );
for cntEar = 1 : 2
    subplot( 2, 2, cntEar );
    plot( timeVec * 1e3, dataEchoes( cntEar, : ) );
    xlabel( 'Time (ms)' );
    title( earNames{ cntEar } );
    subplot( 2, 2, 2 + cntEar );
    spectrogram( dataEchoes( cntEar, : ), 256, 224, 256, structSonar.sampleRate, 'yaxis' );
    ylim( [ 20 120 ] );
end
sgtitle( structERTF.species );
