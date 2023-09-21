function kernel = skewNormalKernel(x, xi, omega, alpha)
    % Skew Normal Kernel function for PNN
    
    % Calculate the standard normal distribution function value
    phi = normcdf(alpha * ((x - xi) / omega));
    
    dot_prod = (x - xi)*(x - xi)';

    % Calculate the skew normal kernel function
    kernel = (2 / (omega * sqrt(2*pi))) * (exp((-1*dot_prod) / (2 * omega^2)) * norm(phi));
end