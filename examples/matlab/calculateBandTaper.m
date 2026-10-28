function windowBand = calculateBandTaper( freqsHz, freqMin, freqMax, taperWidth )
% Raised-cosine band window: 0 outside [ freqMin, freqMax ], rising and falling over taperWidth [Hz] inside the band
%   freqsHz: frequencies [Hz], any size
%   windowBand: window, same size as freqsHz

rampUp = min( max( ( freqsHz - freqMin ) / taperWidth, 0 ), 1 );
rampDown = min( max( ( freqMax - freqsHz ) / taperWidth, 0 ), 1 );
windowBand = sin( pi / 2 * rampUp ) .^ 2 .* sin( pi / 2 * rampDown ) .^ 2;

end
