% Multivariate Skew Normal Distribution function
function pdf = multivariateSkewNormal(x, mu, sigma, alpha)
    d = size(x, 1);
    phi = mvnpdf(x, mu, sigma); % Multivariate normal PDF
    phi_alpha = normcdf(alpha' * x); % CDF of the univariate standard normal distribution
    
    pdf = 2 * phi * phi_alpha;
end