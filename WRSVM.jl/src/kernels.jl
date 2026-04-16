"""
    rbf_kernel(X1, X2; gamma=0.5)

Compute the RBF kernel `K(x, y) = exp(-gamma * ||x - y||^2)`.
Returns a matrix of shape `(size(X1, 1), size(X2, 1))`.
"""
function rbf_kernel(X1::AbstractMatrix, X2::AbstractMatrix; gamma::Real = 0.5)
    sq1 = sum(X1 .^ 2, dims = 2)
    sq2 = sum(X2 .^ 2, dims = 2)'
    D2 = sq1 .+ sq2 .- 2.0 .* (X1 * X2')
    @. D2 = max(D2, 0.0)
    return exp.(-gamma .* D2)
end
