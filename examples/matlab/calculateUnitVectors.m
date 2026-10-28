function dirVecs = calculateUnitVectors( azVec, elVec )
% Unit vectors of directions, X forward, Y left (azimuth +), Z up (elevation +)
%   azVec, elVec: azimuth and elevation [deg], vectors of equal length
%   dirVecs: [ cos(el) cos(az), cos(el) sin(az), sin(el) ] [nDirections x 3]

azVec = azVec( : );
elVec = elVec( : );
dirVecs = [ cosd( elVec ) .* cosd( azVec ), cosd( elVec ) .* sind( azVec ), sind( elVec ) ];

end
