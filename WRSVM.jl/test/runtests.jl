using WRSVM
using Test
using Random
using LinearAlgebra
using Statistics

function toy_data(; n = 90, K = 3, d = 5, seed = 0)
    rng = MersenneTwister(seed)
    per_class_base = div(n, K)
    sizes = [per_class_base * k for k in K:-1:1]
    total = sum(sizes)
    X = zeros(total, d)
    y = zeros(Int, total)
    pos = 1
    for k in 1:K
        for _ in 1:sizes[k]
            X[pos, :] = randn(rng, d) .+ (k - 1) * 2.5
            y[pos] = k
            pos += 1
        end
    end
    perm = randperm(rng, total)
    return X[perm, :], y[perm]
end

@testset "WRSVM.jl" begin
    X, y = toy_data()

    @testset "RBF kernel" begin
        K = rbf_kernel(X, X; gamma = 0.5)
        @test size(K) == (size(X, 1), size(X, 1))
        @test all(diag(K) .≈ 1.0)
        @test all(K .>= 0.0)
    end

    @testset "Noise injection" begin
        y_min = inject_outliers_minority(X, y; outlier_rate = 0.3, seed = 0)
        classes = sort(unique(y))
        sizes = [count(==(c), y) for c in classes]
        maj = classes[argmax(sizes)]
        @test count(==(maj), y_min) >= count(==(maj), y)
    end

    @testset "Crammer-Singer" begin
        m = solve_crammer_singer(X, y; C = 10.0, gamma = 0.5, upsilon = 0.2)
        preds = predict_cs(m, X)
        @test length(preds) == length(y)
        @test Set(preds) ⊆ Set(m.classes)
        @test mean(preds .== y) > 0.5
    end

    @testset "SimMSVM" begin
        m = solve_simmsvm(X, y; C = 10.0, gamma = 0.5, upsilon = 0.2)
        preds = predict_simmsvm(m, X)
        @test length(preds) == length(y)
        @test mean(preds .== y) > 0.5
    end
end
