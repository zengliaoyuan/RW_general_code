function [x, pval, lambda, iters,  dval] = Newton_stp(p, s, M, H, g, toler, maxiter)
%  min x'Hx + 2g'x + M/p * \|x\|^p  s.t. \|x\|^2 \leq s

% Using Newton method + Armijo line search for solving its Lagrangian dual problem
% min d(lambda) = g'(H - lambda I)^(\dagger)g + rhoplus( -lambda ) s.t. lambda <= min(0, lambdamin(H) )

zerotol = 1e-8;
opts.maxit = 5000;
opts.v0 = sum(H)';
opts.issym = 1;
opts.tol = zerotol;
opts.fail = 'keep';
[u0, lambdamin] = eigs(sparse(H),1, 'SA',  opts);
lambdatilde = min(0, lambdamin);
vref = sqrt(drhoplus_stp(-lambdatilde, p, s, M));
hc2 = 0;

iters = 0;
Hhandle = @(x) H*x;
lambda = lambdatilde - 1;     % heuristic choice; the choice of initial point makes difference   

Hlhandle = @(z) H*z - lambda*z;
[v, ~] = pcg(Hlhandle, g,  [ ], 5000);
xtilde = -v;
normxtilde = norm(xtilde);
if normxtilde > sqrt(s)
    x = (sqrt(s)/norm(xtilde))*xtilde;
    normx = sqrt(s);
else
    x = xtilde;
    normx = normxtilde;
end
Hx = Hhandle(x);
minusdval =  g'*v + rhoplus_stp(-lambda, p, s, M);
dslope = v'*v - drhoplus_stp(-lambda, p, s, M);
pval = x'*Hx + 2*g'*x + M/p*normx^p;

while abs(pval + minusdval)/(abs(pval) + 1) >= toler && iters < maxiter
    iters = iters +1;
    if iters == 1
        [tmpvec, ~] = pcg(Hlhandle, v, [ ], 5000);
    else
        [tmpvec, ~] = pcg(Hlhandle, v, [ ], 5000, [], [], tmpvec);
    end
    dhess =  2*v'*tmpvec + ddrhoplus_stp(-lambda, p, s, M);
    dir = -dslope/dhess;
    stepsize = 1;
    if dir > 0
        stepsize = min(1,0.95*(lambdatilde - lambda)/dir); % backtrack
    end
    dderiv = dir*dslope;
    
    % Armijo line search
    lambdanew = lambda + stepsize*dir;
    Hlhandle = @(z) H*z - lambdanew*z;
    [vnew, ~] = pcg(Hlhandle, g, [ ], 5000, [], [], v);
    dnew = g'*vnew +  rhoplus_stp(-lambdanew, p, s, M);
    while dderiv < 0 && dnew > minusdval + 1e-4*stepsize*dderiv && stepsize > 1e-10
        stepsize = stepsize/2;
        lambdanew  = lambda + stepsize*dir;
        Hlhandle = @(z) H*z - lambdanew*z;
        [vnew, ~] = pcg(Hlhandle, g,  [ ], 5000, [], [], v);
        dnew = g'*vnew +  rhoplus_stp(-lambdanew, p, s, M);
    end
    
    if stepsize <= 1e-10 || dderiv >= 0
        fprintf(' Newton Method terminates at iterate %g: stepsize %g dderiv %g \n', iters, stepsize, dderiv);
        if abs(lambda - lambdatilde) < 1e-10 && (normx - vref)/(abs(vref) + 1) <= 1e-10
            hc2 = 1;
        end
        break
    end
    
    lambda = lambdanew;   
    v = vnew;
    xtilde = -v; 
    normxtilde = norm(xtilde);
    if normxtilde > sqrt(s)
        x = (sqrt(s)/norm(xtilde))*xtilde;
        normx = sqrt(s);
    else
        x = xtilde;
        normx = normxtilde;
    end
    
    Hx = Hhandle(x); 
    minusdval =  dnew;
    dslope = v'*v - drhoplus_stp(-lambda, p, s, M);
    pval = x'*Hx +2*g'*x + M/p*normx^p;
    
    if abs(lambda - lambdatilde) < 1e-10 && (normx - vref)/(abs(vref) + 1) <= 1e-10
        hc2 = 1;
        break
    end
    
end
dval = -minusdval;

if hc2 == 1
    % solving quadratic
    b0 = 2*(x'*u0);
    c0 = normx^2 - vref^2;
    if b0 < 0
        alpha = 0.5*(-b0 + sqrt(b0^2 - 4*c0));
    else
        alpha = 0.5*(-b0 - sqrt(b0^2 - 4*c0));
    end
    x = x + alpha*u0;
    pval = x'*(Hx + alpha*u0*lambdatilde) + 2*(g'*x) + M/p*norm(x)^p;
end

fprintf('--------------- Results of NT_stp ---------------------\n');
fprintf(' Primal value       % 16.10f  Dual value       % 16.10f\n', pval, dval);
fprintf(' Newton Method terminates at iterate %g : lambda = %g,  dslope = %g \n', iters, lambda, dslope)


    function rhoplus = rhoplus_stp(u, p, s, M)
        if u <=  M/2*s^((p-2)/2)
            rhoplus = (p-2)*M/2/p*( 2*max(u, 0)/M )^(p/(p-2));
        else
            rhoplus = s*u - M/p*s^(p/2);
        end
    end

    function drhoplus = drhoplus_stp(u, p, s, M)
        if u <= M/2*s^((p-2)/2)
            drhoplus = (2*max(u,0)/M)^(2/(p-2));
        else
            drhoplus = s;
        end
    end

    function ddrhoplus = ddrhoplus_stp(u, p, s, M)
        %  drhoplus is not differentiable at u_bp = M/2*s^((p-2)/2); we set ddrhoplus(u_bp) = ddprs(u_bp) 
        if u <= M/2*s^((p-2)/2)
            ddrhoplus = 4/M/(p-2)*( 2*max(0,u)/M )^((4-p)/(p-2));
        else
            ddrhoplus = 0;
        end
    end

end
