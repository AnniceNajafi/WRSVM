"""
    solve_simmsvm(X, y; C, gamma, upsilon=0.3)

Solve the SimMSVM WRSVM dual QP (one dual per sample, specially-constructed
Gram matrix).
"""
function solve_simmsvm(X::AbstractMatrix, y::AbstractVector;
                        C::Real, gamma::Real, upsilon::Real = 0.3)
    Xf = Matrix{Float64}(X)
    classes = sort(unique(y))
    K_cls = length(classes)
    N = size(Xf, 1)
    y_idx = [searchsortedfirst(classes, yi) for yi in y]
    n_c = zeros(Float64, K_cls)
    for k in y_idx
        n_c[k] += 1.0
    end

    K_mat = rbf_kernel(Xf, Xf; gamma = gamma)
    same_class = [y_idx[i] == y_idx[j] for i in 1:N, j in 1:N]
    coef_same = K_cls / (K_cls - 1)
    coef_diff = -K_cls / (K_cls - 1)^2
    G = similar(K_mat)
    @. G = ifelse(same_class, coef_same * K_mat, coef_diff * K_mat)

    w_diag = n_c[y_idx] ./ C
    for i in 1:N
        G[i, i] += w_diag[i] + 1e-7
    end
    G = Symmetric(0.5 .* (G .+ G'))

    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, alpha[1:N] >= 0)

    @variable(model, beta[1:K_cls] >= 0)
    for k in 1:K_cls
        idx_k = findall(==(k), y_idx)
        for i in idx_k
            @constraint(model, alpha[i] <= beta[k])
        end
    end
    @objective(model, Min,
                0.5 * (alpha' * G * alpha) - sum(alpha) +
                sum(n_c[k] * upsilon * beta[k] for k in 1:K_cls))

    optimize!(model)
    status = termination_status(model)
    if status ∉ (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED)
        error("SimMSVM solver failed: $status")
    end

    alpha_vals = JuMP.value.(alpha)
    alpha_vals[alpha_vals .< 1e-7] .= 0.0

    beta_vals = Vector{Float64}(JuMP.value.(model[:beta]))

    V = fill(-1.0 / (K_cls - 1), N, K_cls)
    for i in 1:N
        V[i, y_idx[i]] = 1.0
    end
    alpha_mat = V .* alpha_vals
    theta = alpha_mat

    b = _recover_biases_simmsvm(alpha_vals, y_idx, K_mat, theta, n_c, C,
                                  beta_vals, K_cls)

    return WRSVMModel(reshape(alpha_vals, N, 1), theta, beta_vals, b,
                      Xf, classes, K_cls, n_c, Float64(gamma),
                      Float64(C), Float64(upsilon))
end

function _recover_biases_simmsvm(alpha, y_idx, K_mat, theta, n_c, C, beta, K_cls)
    N = length(alpha)
    K_scores = K_mat * theta
    sv = findall(>(1e-6), alpha)
    isempty(sv) && return zeros(K_cls)

    eq_rows = Vector{Vector{Float64}}()
    rhs = Float64[]
    for i in sv
        ci = y_idx[i]
        bc = beta[ci]
        is_free = alpha[i] > 1e-6 && (!isfinite(bc) || alpha[i] < bc - 1e-4)
        is_free || continue
        s_own = K_scores[i, ci]
        s_others = sum(K_scores[i, :]) - s_own
        composite = s_own - s_others / (K_cls - 1)
        xi_i = n_c[ci] * alpha[i] / C
        rhs_i = 1.0 - xi_i - composite
        row = fill(-1.0 / (K_cls - 1), K_cls)
        row[ci] = 1.0
        push!(eq_rows, row)
        push!(rhs, rhs_i)
    end
    isempty(rhs) && return zeros(K_cls)

    A = reduce(vcat, [reshape(r, 1, K_cls) for r in eq_rows])
    A = vcat(A, ones(1, K_cls))
    push!(rhs, 0.0)
    b = try
        A \ rhs
    catch
        zeros(K_cls)
    end
    any(isnan, b) && return zeros(K_cls)
    return Vector{Float64}(b)
end

"""
    predict_simmsvm(model, X_new)
"""
function predict_simmsvm(model::WRSVMModel, X_new::AbstractMatrix)
    Xn = Matrix{Float64}(X_new)
    K_new = rbf_kernel(Xn, model.X_train; gamma = model.gamma)
    scores = K_new * model.theta .+ model.b'
    scores[isnan.(scores) .| isinf.(scores)] .= 0.0
    pred_idx = [argmax(scores[i, :]) for i in 1:size(Xn, 1)]
    return [model.classes[j] for j in pred_idx]
end
