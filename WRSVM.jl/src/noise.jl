"""
    inject_outliers_majority(X, y; outlier_rate=0.14, seed=nothing)

Flip labels of majority-class samples farthest from the majority centroid.
Each flipped sample is reassigned to a uniformly random minority class.
"""
function inject_outliers_majority(X::AbstractMatrix, y::AbstractVector;
                                    outlier_rate::Real = 0.14,
                                    seed::Union{Nothing,Integer} = nothing)
    outlier_rate <= 0 && return copy(y)
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    classes = sort(unique(y))
    sizes = [count(==(c), y) for c in classes]
    maj = classes[argmax(sizes)]
    mins = classes[classes .!= maj]
    isempty(mins) && return copy(y)

    n = length(y)
    n_flip = round(Int, n * outlier_rate)
    maj_idx = findall(==(maj), y)
    centroid = mean(X[maj_idx, :], dims = 1)
    dists = vec(sum((X[maj_idx, :] .- centroid) .^ 2, dims = 2))
    n_flip = min(n_flip, length(maj_idx))
    local_order = partialsortperm(dists, 1:n_flip, rev = true)
    flip_global = maj_idx[local_order]

    y_out = copy(y)
    for i in flip_global
        y_out[i] = rand(rng, mins)
    end
    return y_out
end

"""
    inject_outliers_minority(X, y; outlier_rate=0.3, seed=nothing)

For each minority class, flip `round(outlier_rate * n_k)` random labels
to a uniformly random other class. Majority class is left untouched.
"""
function inject_outliers_minority(X::AbstractMatrix, y::AbstractVector;
                                    outlier_rate::Real = 0.3,
                                    seed::Union{Nothing,Integer} = nothing)
    outlier_rate <= 0 && return copy(y)
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    classes = sort(unique(y))
    sizes = [count(==(c), y) for c in classes]
    maj = classes[argmax(sizes)]
    mins = classes[classes .!= maj]
    isempty(mins) && return copy(y)

    y_out = copy(y)
    for mc in mins
        idx = findall(==(mc), y)
        n_flip = round(Int, length(idx) * outlier_rate)
        n_flip == 0 && continue
        flip_idx = Random.randperm(rng, length(idx))[1:n_flip]
        others = classes[classes .!= mc]
        for i in flip_idx
            y_out[idx[i]] = rand(rng, others)
        end
    end
    return y_out
end
