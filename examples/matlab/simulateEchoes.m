function [ dataEchoes, timeVec ] = simulateEchoes( structERTF, structSonar, dataReflectors )
% Echoes of point reflectors at both ears: call x emitter x ear, spreading 1/r^2, delay 2r/c, air absorption
%   structERTF: dataset of one species (loadBatERTF, with the sources left, right and emitter)
%   structSonar: .sampleRate [Hz], .emissionSignal [1 x nSamplesCall], .speedOfSound [m/s] (343),
%                .structAir (calculateAirAbsorption), .taperWidth [Hz] (5e3)
%   dataReflectors: x y z [m] in the head frame and echo amplitude re 1 m [nReflectors x 4]
%   dataEchoes: signals at the left and right ear [2 x nSamples]; timeVec [s]
% Far-field model, valid for reflectors at distances much larger than the head (as simulate_echoes in dataset.py).

if( ~isfield( structSonar, 'speedOfSound' ) ), structSonar.speedOfSound = 343; end
if( ~isfield( structSonar, 'structAir' ) ), structSonar.structAir = struct( ); end
if( ~isfield( structSonar, 'taperWidth' ) ), structSonar.taperWidth = 5e3; end

% Directions and distances of the reflectors
distReflectors = sqrt( sum( dataReflectors( :, 1 : 3 ) .^ 2, 2 ) );
dirVecs = dataReflectors( :, 1 : 3 ) ./ distReflectors;

% Perform zero-padding to fit the longest echo
numSamples = 2 ^ ceil( log2( length( structSonar.emissionSignal ) + ...
             ( 2 * max( distReflectors ) / structSonar.speedOfSound + 2e-3 ) * structSonar.sampleRate ) );
freqsFFT = ( 0 : numSamples / 2 ).' * structSonar.sampleRate / numSamples;
spectrumCall = fft( structSonar.emissionSignal, numSamples );
spectrumCall = spectrumCall( 1 : numSamples / 2 + 1 ) .* calculateBandTaper( freqsFFT.', structERTF.freqsHz( 1 ), ...
               min( structERTF.freqsHz( end ), structSonar.sampleRate / 2 ), structSonar.taperWidth );

% Propagation: spreading, round-trip delay and absorption [nReflectors x nFreqs]
absorptionDb = calculateAirAbsorption( freqsFFT.', structSonar.structAir );
gainPropagation = dataReflectors( :, 4 ) ./ distReflectors .^ 2 .* ...
                  exp( -1i * 2 * pi * freqsFFT.' .* ( 2 * distReflectors / structSonar.speedOfSound ) ) .* ...
                  10 .^ ( -absorptionDb .* ( 2 * distReflectors ) / 20 );

% Emitter and ears
dataEmitter = interpolateERTF( structERTF, 'emitter', dirVecs, freqsFFT );
earNames = { 'left', 'right' };
dataEchoes = zeros( 2, numSamples );
for cntEar = 1 : 2
    dataEar = interpolateERTF( structERTF, earNames{ cntEar }, dirVecs, freqsFFT );
    spectrumEcho = sum( dataEmitter .* dataEar .* gainPropagation, 1 ) .* spectrumCall;
    dataEchoes( cntEar, : ) = ifft( [ spectrumEcho, conj( spectrumEcho( end - 1 : -1 : 2 ) ) ], numSamples, 'symmetric' );
end
timeVec = ( 0 : numSamples - 1 ) / structSonar.sampleRate;

end
