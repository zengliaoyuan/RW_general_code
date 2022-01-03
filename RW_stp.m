function varargout = RW_stp(p, s, M, H, g, gtol)
% s: the radius of trust region
% p: the power of p regularizer

tstart = tic;

zerotol = 1e-8;
n = length(g);

%% max rw_dual
tinit = tic;

Hhandle = @(x) H*x;
normg = norm(g);
normH = normest(H, 1e-2);

opts.maxit = 5000;
opts.v0 = sum(H)';
opts.issym = 1;
opts.tol = zerotol;
opts.fail = 'keep';
[v0, lambdamin] = eigs(sparse(H), 1, 'SA',  opts);    

ubbeta = max(normg/sqrt(s),  lambdamin + M/2*s^((p-2)/2));
ubalpha = min(sqrt(s), max( (4*p*normg/M)^(1/(p-1)),  (2*p*normH/M)^(1/(p-2)) ) );
low = lambdamin - ubbeta;
high = lambdamin + normg * ubalpha;

fprintf(' time for initialization = %g\n', toc(tinit));
fprintf(' Initial end points: %16.10f  %16.10f\n', low, high);

%% check for hard case 2: when lambdamin > 0, we have ||H^(-1)g||^2 \in argmin rho = (infty, 0] in hard case 2, which implies g = 0
fprintf(' ------------- Case check -------------\n');
if lambdamin > 0
    if normg < zerotol
       x = zeros(n,1);
       fval = 0;
       dval = 0;
       lambdaout = 0;
       t_final = toc(tstart);
       % print output
       fprintf('Positive definite H, lambdamin = %g >0\n', lambdamin);
       fprintf(' Primal value       %16.10f    CPU time           %16.10f\n', fval,  t_final);
       if nargout >= 1; varargout{1} = x; end
       if nargout >= 2; varargout{2} = 0; end
       if nargout >= 3; varargout{3} = fval; end
       if nargout >= 4; varargout{4} = dval; end
       if nargout >= 5; varargout{5} = 0; end
       if nargout >= 6; varargout{6} = t_final; end
       if nargout >= 7; varargout{7} = lambdaout; end
       return
    end
end

hc2 = 0;        % tag for hard case
if lambdamin <= 0      
    vag = g' *v0/normg;
    if vag < 0
        vag = -vag;
    end
    
    %% check if g \in Range(H - lambdamin I)
    if vag < zerotol
        hc2 = 1;    % tag for possible hard case
        
        Hhandletest = @(x) H*x - lambdamin*x;
        [xtmp,~,~,~,~,residual] = minresQLP(Hhandletest,-g,0.01*zerotol);
        fprintf(' relative minres residual %g\n', residual);
        if residual >= zerotol
            hc2 = 0;
        end
        
        %% check and solve hard case 2
        if hc2 == 1  % possible hard case
            if norm(xtmp) <= sqrt( drhoplus_stp(-lambdamin, p, s, M) ) % hard case; explicit solution
                % solving quadratic
                b0 = 2*xtmp' * v0;
                c0 = norm(xtmp)^2 -  drhoplus_stp(-lambdamin, p, s, M) ;
                if b0 < 0
                    alpha = 0.5*(-b0 + sqrt(b0^2 - 4*c0));
                else
                    alpha = 0.5*(-b0 - sqrt(b0^2 - 4*c0));
                end
                x = xtmp + alpha*v0;
                pfeas = max( 0, norm(x)^2 - s);
                lambdaout = lambdamin;
                fval = 2*(g'*x) + x'*Hhandle(x)+ rho_stp( norm(x)^2 , p, s, M);
                dval = fval;
                t_final = toc(tstart);
                fprintf(' Hard case. explicit solutions\n')
                %  print output
                fprintf('\n ------------- Results of RW_stp-------------\n')
                
                fprintf(' Primal value       %16.10f  lambda       %16.10f\n', fval, lambdaout)
                fprintf(' Rel Opt Cond.     %16.10f  eigdiff     %16.10f\n', norm(Hhandle(x) - lambdaout*x + g), max(lambdaout + M/2*norm(x)^(p-2), 0) );
                fprintf(' Pfeas                   %2.1e                      CPU time           %16.10f \n',pfeas,  t_final)
                if nargout >= 1; varargout{1} = x; end
                if nargout >= 2; varargout{2} = 0; end
                if nargout >= 3; varargout{3} = fval; end
                if nargout >= 4; varargout{4} = dval; end
                if nargout >= 5; varargout{5} = 0; end
                if nargout >= 6; varargout{6} = t_final; end
                if nargout >= 7; varargout{7} = lambdaout; end
                return
            end
        end
    end
end

%% maximize tilde k(t) - t. tilde k will be differentialble at its maximizer.
v_init = [1; v0];
%% possibly shrink interval for near hard cases
if hc2 == 1
    high  = min(high, lambdamin - g'*xtmp);
end

side = 0;
iter = 0;

gap = inf;
fval = inf;
dval = -inf;
k_slope = 1;
aa = 1;     % used in inverse interpolation; has to be >=1
ptgood = []; ptbad = [];

%% Enter main loop
fprintf(' ------------- Enter main loop -------------\n')

while  ~( abs(gap)/(abs(fval) + 1) < gtol || (high - low)/(abs(high) + abs(low)) < gtol ) && iter < 50
    iter = iter + 1;
    
    gindex = size(ptgood, 1);
    bindex = size(ptbad, 1);
    
    ss1 = 1/sqrt(sqrt(aa));     % parameter corresponding to 1/4 ???  how about 1/aa?
    % strategies for maximizing k
    
    tt = (high + low)/2;        % midpoint rule
    
    [tt,high,low] = vertical_cut(gindex,bindex,ptgood,ptbad,high,low,tt); % vertical cut; update both t and the interval, if possible
    
    tt = trig_interpol(gindex,bindex,ptgood,ptbad,high,low,tt); % triangle interpolation; overwrites the previous only when triangle lies between low and high

    tt = inv_interpol(gindex,bindex,ptgood,ptbad,side,ss1,high,low,tt); % inverse interpolation; overwrites the previous only when inv lies between low and high

    if side > 0
        v_init = gv_init;
    elseif side < 0
        v_init = bv_init;
    end
    
    opts.v0 = v_init; % warm start
    opts.tol = zerotol;
    H0handle = @(x) [tt*x(1) + g'*x(2:end); g*x(1) + H*x(2:end)];
    eigtime = tic;
    [v,lambdaout] = eigs(H0handle,n+1,1,'SA',opts);
    eigtimeend = toc(eigtime);
    
    if lambdaout >= 0
        gamma = 0;
    else
        gamma = drhoplus_stp(-lambdaout, p, s, M)+1;
    end
    
    k_slope = gamma*v(1)^2 - 1;
    dval = rho_stp( gamma-1, p, s, M )+ gamma*lambdaout - tt;    
    
    %% recover x
    x0 = v(2:end);
    x =  ( sqrt( drhoplus_stp(-lambdaout, p, s, M) ) )/norm(x0)*sign(v(1))*x0;
    if abs(v(1)) < zerotol
        fprintf('warning: first entry of v is small \n');
    end
    fval =  2*(x'*g) + x'*Hhandle(x) + rho_stp(norm(x)^2, p, s, M);
    
    %% duality and optimality conditions
    gap = fval - dval;

    %% record good side/bad side points   
    if k_slope > 0 
        ptbad = [tt sqrt(sqrt(k_slope + aa)) k_slope dval; ptbad];
        low = tt;
        side = min(-1, side -1);
        bv_init = v;
        
        fprintf('%3d  bad  : ', iter);
        
    elseif k_slope < 0
        ptgood = [tt sqrt(sqrt(k_slope + aa)) k_slope dval;ptgood];
        high = tt;
        side = max(1, side+1);
        gv_init = v;

        fprintf('%3d  good  : ', iter);
    end
    fprintf('primal value % 6.5e, gap % 3.2e, kslope % 3.2e,  eigstime  %4.1f \n',...
        fval, gap, k_slope, eigtimeend)
    
end

t_final = toc(tstart);

% print output

fprintf('\n ------------- Results of RW_stp -------------\n')

fprintf(' Primal value       % 16.10f  Dual value       % 16.10f\n', fval, dval);
fprintf(' Gap                % 2.1e    interval width     % 2.1e   \n', gap, high - low);
fprintf(' k_slope (approx)   % 2.1e          CPU time         % 4.1f\n', k_slope, t_final);

if nargout >= 1; varargout{1} = x; end
if nargout >= 2; varargout{2} = iter; end
if nargout >= 3; varargout{3} = fval; end
if nargout >= 4; varargout{4} = dval; end
if nargout >= 5; varargout{5} = k_slope; end
if nargout >= 6; varargout{6} = t_final; end
if nargout >= 7; varargout{7} = lambdaout; end

%% end of main function




%% nested functions
    function tmid = trig_interpol(gindex,bindex,ptgood,ptbad,high,low,tmid)
        
        if bindex > 0 && gindex > 0
            
            % k'(t) from the last point on either side
            tbslope = ptbad(1,3);
            tgslope = ptgood(1,3);
            
            % k(t) from the last point on either side
            ktb = ptbad(1,4);
            ktg = ptgood(1,4);
            
            ttemp = (tbslope*ptbad(1,1) + ktg - tgslope*ptgood(1,1) - ktb)/(tbslope - tgslope);
            
            if low <= ttemp && ttemp <= high
                tmid = ttemp;
                %     fprintf('triang used\n')
            else
                %     fprintf('triang fails: outside\n')
            end
        end
        
    end

    function [tmid,high,low] = vertical_cut(gindex,bindex,ptgood,ptbad,high,low,tmid)
        
        if gindex > 0 && bindex > 0
            
            ktg = ptgood(1,4);
            ktb = ptbad(1,4);
            
            if ktg > ktb % value on good side > bad side
                tmplow = ptbad(1,1) + (ktg - ktb)/ptbad(1,3);
                
                if low <= tmplow && tmplow <= high
                    low = tmplow;
                    %       fprintf(' vertical cut: interval reduced\n')
                    if tmid < low
                        tmid = low;
                        %         fprintf(' vertical cut used\n')
                    end
                end
            else
                tmphigh = ptgood(1,1) + (ktb - ktg)/ptgood(1,3);
                
                if low <= tmphigh && tmphigh <= high
                    high = tmphigh;
                    %       fprintf(' vertical cut: interval reduced\n')
                    if tmid > high
                        tmid = high;
                        %         fprintf(' vertical cut used\n')
                    end
                end
            end
        end
        
    end

    function tmid = inv_interpol(gindex,bindex,ptgood,ptbad,side,ss1,high,low,tmid)
        
        if side < 0 && bindex > 1 % inverse interpolation
            tint = ptbad(1:2,1);
            vint = ptbad(1:2,2);
            
            f1 = -1/vint(1) + ss1;
            f2 = -1/vint(2) + ss1;
            
            slope = (tint(1) - tint(2))/(f1 - f2);
            ttemp = tint(1) - f1*slope; % linear inverse interpolation
            if (low <= ttemp) && (ttemp <= high)
                tmid = ttemp;
                %     fprintf(' inv interpol used\n')
            else
                %     fprintf(' inv interpol failed\n')
            end
        else
            if side > 0 && gindex > 1 % inverse interpolation
                tint = ptgood(1:2,1);
                vint = ptgood(1:2,2);
                
                f1 = -1/vint(1) + ss1;
                f2 = -1/vint(2) + ss1;
                
                slope = (tint(1) - tint(2))/(f1 - f2);
                ttemp = tint(1) - f1*slope; % linear inverse interpolation
                if (low <= ttemp) && (ttemp <= high)
                    tmid = ttemp;
                    %       fprintf(' inv interpol used\n')
                else
                    %       fprintf(' inv interpol failed\n')
                end
            end
        end
        
    end

    function rho = rho_stp(t, p, s, M)
        %  the regularization term: sum of indicator function and p regularizer
        %    rho(t) = M/p*max(0,t)^(p/2) + delta_(-infty, s](t)
            rho = M/p*max(0,t)^(p/2);
    end

    function drhoplus = drhoplus_stp(u, p, s, M)
        %  the derivative of rhoplus for stp
        if u<= M/2*s^( (p-2)/2 )
            drhoplus  = ( 2*max(0,u)/M )^( 2/(p-2) );
        else
            drhoplus = s;
        end
    end

end
