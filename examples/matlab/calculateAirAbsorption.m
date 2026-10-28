function absorptionDb = calculateAirAbsorption( freqsHz, structAir )
% Atmospheric absorption [dB/m], ISO 9613-1 pure-tone formula (stated for 50 Hz to 10 kHz, extrapolated above, as
% usual in bioacoustics; expect 10-20 % uncertainty). Same as batertf/dataset.py.
%   freqsHz: frequencies [Hz], any size
%   structAir: .temperature [deg C], .humidity [% RH], .pressure [kPa]; default 20 deg C, 50 %, 101.325 kPa

if( nargin < 2 )
    structAir = struct( );
end
if( ~isfield( structAir, 'temperature' ) ), structAir.temperature = 20; end
if( ~isfield( structAir, 'humidity' ) ), structAir.humidity = 50; end
if( ~isfield( structAir, 'pressure' ) ), structAir.pressure = 101.325; end

tempKelvin = structAir.temperature + 273.15;
tempRef = 293.15;
tempTriple = 273.16;
pressureRel = structAir.pressure / 101.325;

% Molar concentration of water vapour and the relaxation frequencies of oxygen and nitrogen
concentrationWater = structAir.humidity * 10 ^ ( -6.8346 * ( tempTriple / tempKelvin ) ^ 1.261 + 4.6151 ) / pressureRel;
freqOxygen = pressureRel * ( 24 + 4.04e4 * concentrationWater * ( 0.02 + concentrationWater ) / ( 0.391 + concentrationWater ) );
freqNitrogen = pressureRel * ( tempKelvin / tempRef ) ^ -0.5 * ...
               ( 9 + 280 * concentrationWater * exp( -4.170 * ( ( tempKelvin / tempRef ) ^ ( -1 / 3 ) - 1 ) ) );

absorptionDb = 8.686 * freqsHz .^ 2 .* ( 1.84e-11 / pressureRel * ( tempKelvin / tempRef ) ^ 0.5 + ...
               ( tempKelvin / tempRef ) ^ -2.5 * ( 0.01275 * exp( -2239.1 / tempKelvin ) ./ ( freqOxygen + freqsHz .^ 2 / freqOxygen ) + ...
               0.1068 * exp( -3352.0 / tempKelvin ) ./ ( freqNitrogen + freqsHz .^ 2 / freqNitrogen ) ) );

end
