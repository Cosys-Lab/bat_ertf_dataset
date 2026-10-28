function folderSpecies = findSpeciesFolder( speciesPreferred )
% Find a species folder of the dataset (a folder that contains ertf_ct.h5)
%   speciesPreferred: folder name to prefer, default 'Phyllostomus_discolor'
% Looks in the environment variable BAT_ERTF_DATA (a species folder or the dataset root), then relative to this
% file as in the Zenodo layout (<root>\code\examples\matlab). The examples use a variable folderSpecies from the
% workspace instead, if it is set.

if( nargin < 1 )
    speciesPreferred = 'Phyllostomus_discolor';
end

folderThis = fileparts( mfilename( 'fullpath' ) );
allRoots = { };
if( ~isempty( getenv( 'BAT_ERTF_DATA' ) ) )
    allRoots{ end + 1 } = getenv( 'BAT_ERTF_DATA' );
end
allRoots{ end + 1 } = fullfile( folderThis, '..', '..', '..' );

for cntRoot = 1 : length( allRoots )
    folderRoot = allRoots{ cntRoot };
    if( isfile( fullfile( folderRoot, 'ertf_ct.h5' ) ) )
        folderSpecies = folderRoot;
        return
    end
    if( isfile( fullfile( folderRoot, speciesPreferred, 'ertf_ct.h5' ) ) )
        folderSpecies = fullfile( folderRoot, speciesPreferred );
        return
    end
    allSubFolders = dir( folderRoot );
    allSubFolders = allSubFolders( [ allSubFolders.isdir ] & ~startsWith( { allSubFolders.name }, '.' ) );
    for cntFolder = 1 : length( allSubFolders )
        folderCandidate = fullfile( folderRoot, allSubFolders( cntFolder ).name );
        if( isfile( fullfile( folderCandidate, 'ertf_ct.h5' ) ) )
            folderSpecies = folderCandidate;
            return
        end
    end
end

% Report where we looked (a species folder needs ertf_ct.h5; with Dropbox, wait until it has synced)
allRootsFull = cellfun( @( x ) char( java.io.File( x ).getCanonicalPath( ) ), allRoots, 'UniformOutput', false );
error( [ 'No species folder (with ertf_ct.h5) found in: ' strjoin( allRootsFull, ', ' ) '. Set folderSpecies = ' ...
         '''<path to a species folder>'' before running an example, or set the environment variable BAT_ERTF_DATA.' ] );

end
