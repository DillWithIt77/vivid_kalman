% coarsen raw data set


function [qhc, tsave] = coarsenqh(datafolder,dtsave)

savefilename = fullfile(datafolder,sprintf('coarseqh_dt%.2g.mat',dtsave));

searchfilename = dir(fullfile(datafolder,sprintf('coarseqh_dt%.2g.mat',dtsave))); % arbitrarily load regardless of dtsave size
if ~isempty(searchfilename)
    data = load(fullfile(searchfilename.folder,searchfilename.name));
    qhc = data.qhc;
    tsave = data.tsave;
    return 
end
    

fnameqh = fullfile(datafolder, 'qh.bin');
fnamet = fullfile(datafolder, 't.bin');

fidqh = fopen(fnameqh,'r');
fidt = fopen(fnamet,'r');

t = fread(fidt,'double'); 
n_t = length(t);
dt = t(2) - t(1);

dtsave = round(dtsave/dt)*dt; % round to a factor of dt
nsavef = dtsave/dt;
tsave = t(1:nsavef:length(t));
n_tsave = length(tsave);

dataext = load(fullfile(datafolder, 'diagnostics.mat'));
Ntruth =  dataext.params.N;
Ntotal = Ntruth*Ntruth;


qhc = zeros(Ntotal, n_tsave);
ii = 0;
for i = 1:n_t
    qq = fread(fidqh,2*Ntotal,'double');
    qq = complex(qq(1:Ntotal), qq(Ntotal+1:2*Ntotal));
    if ~mod(i-1,nsavef)
        ii = ii + 1;
        qhc(:,ii) = qq(:);
    end
end
fclose(fidqh); fclose(fidt);


save(savefilename,'qhc','tsave','-v7.3');

end