module WRSVM

using LinearAlgebra
using Random
using Statistics
using JuMP
import Clarabel

export compute_kernel,
       rbf_kernel, linear_kernel, poly_kernel,
       sigmoid_kernel, laplacian_kernel,
       solve_crammer_singer, predict_cs,
       solve_simmsvm, predict_simmsvm,
       inject_outliers_majority, inject_outliers_minority,
       WRSVMModel

include("kernels.jl")
include("noise.jl")
include("solver_cs.jl")
include("solver_simmsvm.jl")

end
