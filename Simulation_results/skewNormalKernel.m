function kernel = skewNormalKernel(x, xi, omega, alpha)
    % Skew Normal Kernel function for PNN
    % disp(size(x))
    % disp(size(xi))
    % Calculate the standard normal distribution function value

    phi = normcdf(alpha * ((x - xi) / omega));

    dot_prod = sum((x-xi).^2,2);

    % Calculate the skew normal kernel function
    kernel = sum((2 / (omega * sqrt(2*pi))) * (exp((-1*dot_prod) / (2 * omega^2)) .* sqrt(sum(phi.^2,2))));
    %disp(kernel)
end