function [ dataIR, timeVec ] = calculateImpulseResponse( structERTF, sourceName, dirVecs, structIR )
% Impulse responses of one source towards a set of directions
%   structERTF: dataset of one species (loadBatERTF)
%   sourceName: 'left', 'right', 'emitter', ...
%   dirVecs: unit vectors [nDirections x 3]
%   structIR: .sampleRate [Hz] (500e3), .numSamples (1024, even), .preDelay [s] (0.2e-3),
%             .taperWidth [Hz] (5e3, raised cosine at both band edges, as there are no data outside the band)
%   dataIR: impulse responses [nDirections x numSamples]; timeVec [s], 0 = arrival of the free-field wave at the origin

if( nargin < 4 )
    structIR = struct( );
end
if( ~isfield( structIR, 'sampleRate' ) ), structIR.sampleRate = 500e3; end
if( ~isfield( structIR, 'numSamples' ) ), structIR.numSamples = 1024; end
if( ~isfield( structIR, 'preDelay' ) ), structIR.preDelay = 0.2e-3; end
if( ~isfield( structIR, 'taperWidth' ) ), structIR.taperWidth = 5e3; end
if( mod( structIR.numSamples, 2 ) ~= 0 )
    error( 'numSamples must be even' )
end

% Transfer functions on the FFT grid, band-limited, with a pre-delay (the response starts before the arrival at
% the origin for directions on the side of the source)
freqsFFT = ( 0 : structIR.numSamples / 2 ).' * structIR.sampleRate / structIR.numSamples;
dataH = interpolateERTF( structERTF, sourceName, dirVecs, freqsFFT );
windowBand = calculateBandTaper( freqsFFT.', structERTF.freqsHz( 1 ), min( structERTF.freqsHz( end ), structIR.sampleRate / 2 ), structIR.taperWidth );
dataH = dataH .* windowBand .* exp( -1i * 2 * pi * freqsFFT.' * structIR.preDelay );

% Inverse FFT of the conjugate-symmetric spectrum (exp( +j w t ) convention)
dataSpectrum = [ dataH, conj( dataH( :, end - 1 : -1 : 2 ) ) ];
dataIR = ifft( dataSpectrum, structIR.numSamples, 2, 'symmetric' );
timeVec = ( 0 : structIR.numSamples - 1 ) / structIR.sampleRate - structIR.preDelay;

end
