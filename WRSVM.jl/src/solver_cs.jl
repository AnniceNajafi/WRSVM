"""
    WRSVMModel

Solution of the Crammer-Singer (or SimMSVM) WRSVM dual QP.
"""
struct WRSVMModel
    alpha::Matrix{Float64}
    theta::Matrix{Float64}
    beta::Vector{Float64}
    b::Vector{Float64}
    X_train::Matrix{Float64}
    classes::Vector
    K_cls::Int
    n_c::Vector{Float64}
    gamma::Float64
    C::Float64
    upsilon::Float64
    kernel::String
    degree::Int
    coef0::Float64
end

"""
    solve_crammer_singer(X, y; C, gamma, upsilon=0.3)

Solve the Crammer-Singer WRSVM dual QP using Clarabel.
"""
function solve_crammer_singer(X::AbstractMatrix, y::AbstractVector;
                               C::Real, gamma::Real,
                               upsilon::Real = 0.3,
                               kernel::AbstractString = "rbf",
                               degree::Integer = 3,
                               coef0::Real = 0.0)
    Xf = Matrix{Float64}(X)
    classes = sort(unique(y))
    K_cls = length(classes)
    N = size(Xf, 1)
    y_idx = [searchsortedfirst(classes, yi) for yi in y]
    n_c = zeros(Float64, K_cls)
    for k in y_idx
        n_c[k] += 1.0
    end

    K_mat = compute_kernel(Xf, Xf; kernel = kernel, gamma = gamma,
                           degree = degree, coef0 = coef0)
    if kernel == "sigmoid"
        # Sigmoid kernel is not Mercer; project onto PSD cone.
        K_mat = _psd_project(Symmetric(0.5 .* (K_mat .+ K_mat')))
    end
    K_mat = K_mat + I(N) * 1e-6
    pos_mask = zeros(Float64, N, K_cls)
    for i in 1:N
        pos_mask[i, y_idx[i]] = 1.0
    end
    neg_mask = 1.0 .- pos_mask

    w_diag = n_c[y_idx] ./ C

    P = _build_hessian_cs(K_mat, pos_mask, w_diag)
    A_eq = _build_equality_cs(pos_mask, neg_mask)
    NK = N * K_cls

    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, alpha[1:NK] >= 0)

    for i in 1:N
        fix(alpha[y_idx[i] * N - N + i], 0.0; force = true)
    end

    @constraint(model, A_eq * alpha .== 0)

    obj_quad = 0.5 * (alpha' * P * alpha)
    obj_lin = -sum(alpha)

    @variable(model, beta[1:K_cls] >= 0)
    for k in 1:K_cls
        idx_k = findall(==(k), y_idx)
        for j in 1:K_cls
            for i in idx_k
                @constraint(model, alpha[(j - 1) * N + i] <= beta[k])
            end
        end
    end
    @objective(model, Min,
                obj_quad + obj_lin + sum(n_c[k] * (K_cls - 1) * upsilon * beta[k] for k in 1:K_cls))

    optimize!(model)
    status = termination_status(model)
    if status ∉ (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED)
        error("Solver failed: $status")
    end

    alpha_flat = JuMP.value.(alpha)
    alpha_mat = zeros(Float64, N, K_cls)
    for j in 1:K_cls
        alpha_mat[:, j] = alpha_flat[(j - 1) * N + 1:(j * N)]
    end
    alpha_mat[alpha_mat .< 1e-7] .= 0.0

    beta_vals = JuMP.value.(model[:beta])

    theta = pos_mask .* sum(alpha_mat, dims = 2) .- neg_mask .* alpha_mat
    b = _recover_biases_cs(alpha_mat, y_idx, K_mat, theta, n_c, C, beta_vals)

    return WRSVMModel(alpha_mat, theta, Vector{Float64}(beta_vals), b,
                      Xf, classes, K_cls, n_c, Float64(gamma),
                      Float64(C), Float64(upsilon),
                      String(kernel), Int(degree), Float64(coef0))
end

function _build_hessian_cs(K_mat, pos_mask, w_diag)
    N, K_cls = size(pos_mask)
    NK = N * K_cls
    P = zeros(Float64, NK, NK)
    for k in 1:K_cls
        p_k = pos_mask[:, k]
        s_k = 2 .* p_k .- 1
        pp = (p_k * p_k') .* K_mat
        ps = (p_k * s_k') .* K_mat
        sp = (s_k * p_k') .* K_mat
        ss = (s_k * s_k') .* K_mat
        for j1 in 1:K_cls, j2 in 1:K_cls
            r_rng = (j1 - 1) * N + 1:j1 * N
            c_rng = (j2 - 1) * N + 1:j2 * N
            blk = if j1 != k && j2 != k; pp
            elseif j1 != k && j2 == k; ps
            elseif j1 == k && j2 != k; sp
            else; ss
            end
            P[r_rng, c_rng] .+= blk
        end
    end
    for i in 1:NK
        P[i, i] += w_diag[mod1(i, N)]
    end
    return Symmetric(0.5 .* (P .+ P'))
end

function _build_equality_cs(pos_mask, neg_mask)
    N, K_cls = size(pos_mask)
    NK = N * K_cls
    A = zeros(Float64, K_cls, NK)
    for k in 1:K_cls, j in 1:K_cls
        c_rng = (j - 1) * N + 1:j * N
        if j != k
            A[k, c_rng] .+= pos_mask[:, k]
        else
            A[k, c_rng] .+= pos_mask[:, k] .- neg_mask[:, k]
        end
    end
    return A
end

function _recover_biases_cs(alpha, y_idx, K_mat, theta, n_c, C, beta)
    N, K_cls = size(alpha)
    K_scores = K_mat * theta
    w_xi = n_c[y_idx] ./ C

    diff_sum = zeros(Float64, K_cls, K_cls)
    diff_count = zeros(Int, K_cls, K_cls)
    for i in 1:N
        ci = y_idx[i]
        bc = beta[ci]
        for k in 1:K_cls
            k == ci && continue
            aik = alpha[i, k]
            is_free = aik > 1e-6 && (!isfinite(bc) || aik < bc - 1e-4)
            if is_free
                xi_ik = w_xi[i] * aik
                obs = (1.0 - xi_ik) - (K_scores[i, ci] - K_scores[i, k])
                diff_sum[ci, k] += obs
                diff_count[ci, k] += 1
            end
        end
    end

    active = [(i, j) for i in 1:K_cls, j in 1:K_cls if diff_count[i, j] > 0]
    if isempty(active)
        return zeros(K_cls)
    end

    n_eqs = length(active) + 1
    M_sys = zeros(Float64, n_eqs, K_cls)
    rhs = zeros(Float64, n_eqs)
    for (r, (ci, j)) in enumerate(active)
        M_sys[r, ci] = 1.0
        M_sys[r, j] = -1.0
        rhs[r] = diff_sum[ci, j] / diff_count[ci, j]
    end
    M_sys[end, :] .= 1.0
    b = try
        M_sys \ rhs
    catch
        zeros(K_cls)
    end
    any(isnan, b) && return zeros(K_cls)
    return Vector{Float64}(b)
end

"""
    predict_cs(model, X_new)

Predict class labels from a Crammer-Singer `WRSVMModel`.
"""
function predict_cs(model::WRSVMModel, X_new::AbstractMatrix)
    Xn = Matrix{Float64}(X_new)
    K_new = compute_kernel(Xn, model.X_train;
                           kernel = model.kernel, gamma = model.gamma,
                           degree = model.degree, coef0 = model.coef0)
    scores = K_new * model.theta .+ model.b'
    scores[isnan.(scores) .| isinf.(scores)] .= 0.0
    pred_idx = [argmax(scores[i, :]) for i in 1:size(Xn, 1)]
    return [model.classes[j] for j in pred_idx]
end
