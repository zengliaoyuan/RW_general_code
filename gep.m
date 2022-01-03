function [x_gep, fx_gep, time_gep] = gep(H, g, M, delta)
% generalized eigenvalue based approach for cubic regularized subproblems
n = length(g);
sigma = M/2;        % In the original paper of GEP, they consider min 0.5*x'*H*x + g'*x + sigma/3*\|x\|^3
normg = norm(g);

tstart = tic;
MatPen = @(v) [sigma*v(n+2); -v(1)*g-H*v(2:n+1); -g'*v(n+3:end); sigma*v(2:n+1) - H*v(n+3:end)];
[v, lambda2] = eigs(MatPen, 2*(n+1), 1,'largestreal', 'FailureTreatment', 'keep');
if abs(v(1)) < 1e-12
    v = real(v);     % without these two line, eigs gives complex lambda2 and v for hard case instances
    lambda2 = real(lambda2);
end
v2 =  v(2:n+1);
normv2 = norm(v2);
v4 = v(n+3:2*(n+1));
normv4 = norm(v4);
if abs(lambda2) < 1e-8 || normv2 < 1e-8
    x_gep = zeros(n,1);
else
    x_gep = -sign(g'*v4) * lambda2 * v2 / (sigma * normv2);
end
fx_gep = g'*x_gep + 0.5*x_gep'*H*x_gep + sigma*norm(x_gep)^3/3;

if abs(g'*v4) <= delta*normg*normv4         %  hard case checking
%     [v4,lambdamin] = eigs(H, 1, 'smallestreal', 'Startvector', v4/normv4, 'FailureTreatment', 'keep');
%     normv4 = norm(v4);
%     lambda2 = max(-lambdamin,0);
    Hlhandle = @(x) H*x + lambda2 *x;
    d = minresQLP(Hlhandle, -g);
    a0 = normv4^2;
    b0 = 2*(d'*v4);
    c0 = norm(d)^2- (lambda2/sigma)^2;
    discr = b0^2 - 4*a0*c0;
    if discr <0
          fprintf('discr.  %g not >=0 error ?????? \n',discr)
          keyboard
    end
    if b0 < 0
        t = 0.5*(-b0 + sqrt(discr))/a0;
    else
        t = 0.5*(-b0 - sqrt(discr))/a0;
    end
    xbar = d + t*v4;
    fx_bar = g'*xbar + 0.5*xbar'*H*xbar + sigma*norm(xbar)^3/3;
    if  fx_bar < fx_gep
        x_gep = xbar;
        fx_gep = fx_bar;
    end
end
%     normxgep = norm(x_gep);
%     tmphandle = @(x) H*x + sigma*normxgep*x + sigma*x_gep*(x_gep'*x)/normxgep;   
%     nts = pcg(tmphandle, H*x_gep+sigma*normxgep*x_gep+g, 1e-14, 1000);
%     x_gep = x_gep - nts;
time_gep = toc(tstart);

fprintf('GEP method done: fval = %16.10f, CPU time = %4.1f \n', fx_gep, time_gep)



