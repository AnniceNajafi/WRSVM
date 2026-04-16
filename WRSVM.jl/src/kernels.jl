"""
    compute_kernel(X1, X2; kernel="rbf", gamma=0.5, degree=3, coef0=0.0)

Unified kernel dispatcher. Supported values of `kernel`:

- `"rbf"`       : `exp(-gamma * ||x - y||^2)` (default)
- `"linear"`    : `x . y`
- `"poly"`      : `(gamma * x . y + coef0)^degree`
- `"sigmoid"`   : `tanh(gamma * x . y + coef0)`
- `"laplacian"` : `exp(-gamma * ||x - y||_1)`

Returns a matrix of shape `(size(X1, 1), size(X2, 1))`.
"""
function compute_kernel(X1::AbstractMatrix, X2::AbstractMatrix;
                        kernel::AbstractString = "rbf",
                        gamma::Real = 0.5, degree::Integer = 3,
                        coef0::Real = 0.0)
    if kernel == "rbf"
        return rbf_kernel(X1, X2; gamma = gamma)
    elseif kernel == "linear"
        return linear_kernel(X1, X2)
    elseif kernel == "poly"
        return poly_kernel(X1, X2; gamma = gamma, degree = degree, coef0 = coef0)
    elseif kernel == "sigmoid"
        return sigmoid_kernel(X1, X2; gamma = gamma, coef0 = coef0)
    elseif kernel == "laplacian"
        return laplacian_kernel(X1, X2; gamma = gamma)
    else
        throw(ArgumentError("Unknown kernel '$kernel'. " *
              "Expected one of: rbf, linear, poly, sigmoid, laplacian."))
    end
end

"""
    rbf_kernel(X1, X2; gamma=0.5)

`K(x, y) = exp(-gamma * ||x - y||^2)`.
"""
function rbf_kernel(X1::AbstractMatrix, X2::AbstractMatrix; gamma::Real = 0.5)
    sq1 = sum(X1 .^ 2, dims = 2)
    sq2 = sum(X2 .^ 2, dims = 2)'
    D2 = sq1 .+ sq2 .- 2.0 .* (X1 * X2')
    @. D2 = max(D2, 0.0)
    return exp.(-gamma .* D2)
end

"""
    linear_kernel(X1, X2)

`K(x, y) = x . y`.
"""
function linear_kernel(X1::AbstractMatrix, X2::AbstractMatrix)
    return X1 * X2'
end

"""
    poly_kernel(X1, X2; gamma=0.5, degree=3, coef0=0.0)

`K(x, y) = (gamma * x . y + coef0)^degree`.
"""
function poly_kernel(X1::AbstractMatrix, X2::AbstractMatrix;
                     gamma::Real = 0.5, degree::Integer = 3,
                     coef0::Real = 0.0)
    return (gamma .* (X1 * X2') .+ coef0) .^ degree
end

"""
    sigmoid_kernel(X1, X2; gamma=0.5, coef0=0.0)

`K(x, y) = tanh(gamma * x . y + coef0)`.
"""
function sigmoid_kernel(X1::AbstractMatrix, X2::AbstractMatrix;
                        gamma::Real = 0.5, coef0::Real = 0.0)
    return tanh.(gamma .* (X1 * X2') .+ coef0)
end

"""
    laplacian_kernel(X1, X2; gamma=0.5)

`K(x, y) = exp(-gamma * ||x - y||_1)`.
"""
# Project a symmetric matrix onto the PSD cone by clipping negative
# eigenvalues to zero. Used for non-Mercer kernels (e.g. sigmoid) so the
# Gram matrix is accepted by Clarabel's QP path.
function _psd_project(A::AbstractMatrix)
    F = eigen(Symmetric(Matrix{Float64}(A)))
    vals = max.(F.values, 0.0)
    M = F.vectors * Diagonal(vals) * F.vectors'
    return Matrix{Float64}(0.5 .* (M .+ M'))
end

function laplacian_kernel(X1::AbstractMatrix, X2::AbstractMatrix;
                          gamma::Real = 0.5)
    n1, n2 = size(X1, 1), size(X2, 1)
    D1 = Matrix{Float64}(undef, n1, n2)
    @inbounds for j in 1:n2
        for i in 1:n1
            D1[i, j] = sum(abs, @view(X1[i, :]) .- @view(X2[j, :]))
        end
    end
    return exp.(-gamma .* D1)
end
