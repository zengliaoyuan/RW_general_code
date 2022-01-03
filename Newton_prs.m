function [x, pval, lambda, iters, dval] = Newton_prs(p, M, H, g,  toler, maxiter)
%  min x'Hx + 2g'x + M/p * \|x\|^p

% Using Newton method + Armijo line search for solving its Lagrangian dual problem 
% min d(lambda) = g'(H - lambda I)^(\dagger)g + (p-2)*M/2/p*( -2*lambda/M )^( p/(p-2) ) s.t. lambda <= min(0, lambdamin(H) )

zerotol = 1e-8;
opts.maxit = 5000;
opts.v0 = sum(H)';
opts.tol = zerotol;
opts.issym = 1;
opts.fail = 'keep';
[u0, lambdamin] = eigs(sparse(H),1, 'SA',  opts);
lambdatilde = min(0, lambdamin);
vref = (-2*lambdatilde/M)^(1/(p-2));
hc2 = 0;

iters = 0;
Hhandle = @(x) H*x;
lambda = lambdatilde - 1;     % hueristic choice; the choice of initial point makes difference

Hlhandle = @(z) H*z - lambda*z;
[v, ~] = pcg(Hlhandle, g, [ ], 5000);
x = -v;
normx = norm(x);
Hx = Hhandle(x);
minusdval =  g'*v + (p-2)*M/2/p*( -2*lambda/M )^( p/(p-2) );
dslope = v'*v - (-2*lambda/M)^(2/(p-2));     
pval = x'*Hx + 2*g'*x + M*normx^p/p;

while  abs(pval + minusdval)/(abs(pval) + 1) >= toler && iters < maxiter
    iters = iters +1;
    if iters == 1
        [tmpvec, ~] = pcg(Hlhandle, v, [ ], 5000);
    else
        [tmpvec, ~] = pcg(Hlhandle, v, [ ], 5000, [], [], tmpvec);
    end
    dhess =  2*v'*tmpvec + 2/(p-2)*(2/M)^(2/(p-2))*(-lambda)^((4-p)/(p-2));   
    dir = -dslope/dhess;
    stepsize = 1;
    if dir > 0
        stepsize = min(1,0.95*(lambdatilde - lambda)/dir); % backtrack
    end
    dderiv = dir*dslope;
    
    % Armijo line search
    lambdanew = lambda + stepsize*dir;
    Hlhandle = @(z) H*z - lambdanew*z;
    [vnew, ~] = pcg(Hlhandle, g,  [ ], 5000, [], [], v);
    dnew = g'*vnew +  (p-2)*M/2/p*( -2*lambdanew/M )^( p/(p-2) );
    while dderiv < 0 && dnew > minusdval + 1e-4*stepsize*dderiv && stepsize > 1e-10
        stepsize = stepsize/2;
        lambdanew  = lambda + stepsize*dir;
        Hlhandle = @(z) H*z - lambdanew*z;
        [vnew, ~] = pcg(Hlhandle, g,  [ ], 5000, [], [], v);
        dnew = g'*vnew +  (p-2)*M/2/p*( -2*lambdanew/M )^( p/(p-2) );
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
    x = -v;
    normx = norm(x);
    Hx = Hhandle(x);
    minusdval =  dnew;
    dslope = v'*v - (-2*lambda/M)^(2/(p-2));
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

fprintf('--------------- Results of NT_prs ---------------------\n');
fprintf(' Primal value       % 16.10f  Dual value       % 16.10f\n', pval, dval);
fprintf(' Newton Method terminates at iterate %g : lambda = %g,  dslope = %g  \n', iters, lambda, dslope)

