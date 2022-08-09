function [x,f]=minimize_steepest(x0,fg,optparam)
%
% Jean-Philippe Vert
% June 4, 2006
%
% function [x,f]=minimize_steepest(x0,fg,optparam)
%
% Input:    x0          initial point
%           fg          name of a matlab function [f,g]=fg(x) which returns the
%                       objective function and its gradient
%           optparam    optional parameters
%           optparam.tol            stopping tolerance: the algorithm stops when
%                                   ||g(x)||<=tol*min(1,||g(x0)||)
%           optparam.armijo_alpha   alpha parameter for Armijo rule
%           optparam.armijo_beta    beta parameter for Armijo rule
%           optparam.display        display information is =1%
%           optparam.plot           plot the successive iterates (x[1] and
%                                   x[2])
%
% Output:   xn      final point after optimization
%           fn      objective at the final point

if nargin<3
    optparam=0;
end

if isfield(optparam,'armijo_alpha')
    armijo_alpha=optparam.armijo_alpha;
else
    armijo_alpha=0.1;
end
if isfield(optparam,'armijo_beta')
    armijo_beta=optparam.armijo_beta;
else
    armijo_beta=0.5;
end
if isfield(optparam,'tol')
    tol=optparam.tol;
else
    tol=1e-6;
end
if isfield(optparam,'display')
    display=optparam.display;
else
    display=0;
end
if isfield(optparam,'plot')
    plotit=optparam.plot;
else
    plotit=0;
end
x=x0;
[f,g]=feval(fg,x);
nmg0=norm(g);
nmg=nmg0;
it=0;
if plotit
    hold on
end

% main loop
while (nmg>tol*min(1,nmg0))
    it=it+1;
    if display
        fprintf('it=%3.d  f=%e   ||g||=%e\n',it,f,nmg);
    end
    
    % Descent direction = -gradient
    descent=-g;
    
    % Compute the new point with Armijo rule
    [t,xnew,fnew]=armijo(x,descent,dot(descent,g),fg,1,f,armijo_alpha,armijo_beta);
    if plotit
        plot([x(1),xnew(1)],[x(2),xnew(2)],'-')
    end
    x=xnew;
    
    % Compute the new gradient (next descent direction)
    [f,g]=feval(fg,x);
    nmg=norm(g);
    
end

it=it+1;
if display
    fprintf('it=%3.d  f=%e   ||g||=%e\n',it,f,nmg);
    fprintf('Successful termination!')
end
