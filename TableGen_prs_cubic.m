% comparing GEP, Newton_prs, RW_prs for p = 3 , i.e, crs
% testing instances in three cases:  
%       case == 1    --  easy case instances
%       case == 2    --  near hard case instances
%       else             --  hard case instances
clear;
randn('seed', 2021);
rand('seed', 2021);

p = 3;
theta = 1.2;        % theta is used in M = theta*norm(H)

casesarray = [1 2 3];
narray = 25000 : 25000 : 100000;
density = 0.005;
repeats = 20;

maxiter = 10;
ntoler = 1e-9;
rwtoler =  1e-12;
delta = 1e-5;     % for checking hard case in gep
warning off

a = clock;
fname = ['Results\prs--cubic'  '--'  date  '-'  int2str(a(4))  '-'  int2str(a(5))  '.txt'];
fid = fopen(fname, 'w');
fprintf(fid,'p = %3g                      theta = %3g\n', p, theta);
fprintf(fid, 'maxiter_nt = %3g        density = %6g   \n', maxiter, density);
fprintf(fid, ' %6s & %10s & %10s & %10s & %10s & %10s & %10s\n', ...
        'n','CPU(iter)', 'ratio-gep',  'CPU(iter)', ' ratio-nt', 'CPU(iter)', 'ratio-RW');
    
for cii = 1:length(casesarray)
    cases = casesarray(cii);
    fprintf(fid,'cases == %2g \n', cases);
    for nii = 1:length(narray)
        n = narray(nii);
        progress = [];
        for rr = 1: repeats
            if cases == 1       %  easy case instances          
                % generate random easy instances
                g = randn(n,1);
                H = sprandsym(n,density);
                M = abs(eigs(H,1,'LM'))*theta;
                fprintf('Initialization end.\n');
                
            elseif cases == 2       % near hard case instances
                % generate random near hard instances
                lambdamin = 1;
                while lambdamin >= 0   % lambdamin>=0
                    H = sprandsym(n,density);
                    opts.maxit = 5000;
                    opts.issym = 1;
                    opts.fail = 'keep';
                    [v0,lambdamin] = eigs(H,1,'SA',opts);
                end
                M =  abs(eigs(H,1,'LM'))*theta;
                u = randn(n,1);
                v = H*u - lambdamin*u;
                v = 1.1*v/norm(v)*(-2/M*lambdamin)^(1/(p-2));
                g = H*v - lambdamin*v;
                fprintf('Initialization ends \n');
            
            else                % hard case instances
                % generate random hard instances
                lambdamin = 1;
                while lambdamin >= 0   %  lambdamin >= 0
                    H = sprandsym(n,density);
                    opts.maxit = 5000;
                    opts.issym = 1;
                    opts.fail = 'keep';
                    [v0,lambdamin] = eigs(H,1,'SA',opts);
                end
                M =  abs(eigs(H,1,'LM'))*theta;
                u = randn(n,1);
                v = H*u - lambdamin*u;
                v = 0.9*v/norm(v)*(-2/M*lambdamin)^(1/(p-2));
                g = H*v - lambdamin*v;
                fprintf('Initialization ends \n');              
            end

            % GEP for crs
            fprintf('Start of gep for p = %g \n ', p)
            start = tic;
            [x_gep, fx_gep, time1] = gep(H, g, M, delta);
            time_gep = toc(start);
            fval_gep = 2* fx_gep;
            
            % Newton_prs
            fprintf('Start of Newton_prs for p = %g \n', p)
            start = tic;
            [x_nt, ~, lambda_nt, iter_nt, dval_nt] = Newton_prs(p,M,H,g,ntoler,maxiter);
            time_nt = toc(start);
            pval_nt = 2*g'*x_nt + x_nt'*H*x_nt + M/p*norm(x_nt)^p;
            
            % RW_prs
            fprintf('Start of RW_prs for p = %g \n', p)
            start = tic;
            [x_rw,iter_rw,pval_rw, dval,k_slope,time,lambda_rw] = RW_prs(p, M, H,g, rwtoler);
            time_rw = toc(start);
            
            fbest = min([ fval_gep, pval_nt, pval_rw ]);
            progress = [progress; time_gep, 1, (fval_gep - fbest)/abs(fbest), time_nt, iter_nt, (pval_nt - fbest)/abs(fbest), time_rw, iter_rw,  (pval_rw - fbest)/abs(fbest)];
                
        end
        
        fprintf(fid,  ' %5.0f & %5.1f(%2.0f) & %2.1e &  %5.1f(%2.0f) & %2.1e & %5.1f(%2.0f) & %2.1e \n',  n,  mean(progress));
    end
end

fclose(fid);

    

