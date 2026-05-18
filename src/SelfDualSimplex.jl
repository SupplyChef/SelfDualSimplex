module SelfDualSimplex

import  Base.*
import  Base.size
import  Base.getindex
import  Base.setindex

using Dates
using DelimitedFiles
using LinearAlgebra
using SparseArrays
using SuiteSparse

include("MPSParser.jl")
include("LU.jl")
include("Presolve.jl")

export PFI
export ETAMatrix
export ftran!
export btran!
export solve

export parseMPS
export Problem
export add_upper_bounds!
export add_lower_bounds!
export handle_negative_lowerbound_variables!
export add_slack_variables!

export LUelimination!
export LUdecomposition

function get_column!(c::Array{Float64, 1}, m::SparseMatrixCSC{Float64, Int64}, j::Int64)
    fill!(c, 0.0)
    @inbounds for k in m.colptr[j]:(m.colptr[j+1]-1)
        c[m.rowval[k]] = m.nzval[k]  
    end
end

function max_argmax(data)
    max_value = -Inf64
    max_index = -1
    @inbounds for i in 1:length(data)
        value = data[i]
        if value > max_value
            max_value = value
            max_index = i
        end
    end
    return (max_value, max_index)
end

function min_argmin(data)
    min_value = Inf64
    min_index = -1
    @inbounds for i in 1:length(data)
        value = data[i]
        if value < min_value
            min_value = value
            min_index = i
        end
    end
    return (min_value, min_index)
end

function computeΔc!(Δc::Array{Float64}, A::SparseMatrixCSC{Float64,Int64}, el::Array{Float64}, pfi::PFI, is_basic::Array{Bool}, l::Int64)
    c = size(A)[2]
    b = size(A)[1]

    fill!(el, 0.0)
    el[l] = 1.0
    btran!(pfi, el)
    #mul!(Δc,  A', el)
    
    @inbounds @simd for i in 1:c
        # if is_basic[i]
        #     Δc[i] = 0.0
        # else
        if i > c - b
            Δc[i] = el[i - (c - b)]
            if abs(Δc[i]) < 1e-12
                Δc[i] = 0
            end
        else 
            Δc[i] = dot(A, i, el)
            if abs(Δc[i]) < 1e-12
                Δc[i] = 0
            end
        end
    end
end

# Computes Δc = B⁻ᵀ eₗ scattered through A.
# Δc_nz/Δc_flag are scratch: on entry they describe the previous call's
# nonzero set; on exit they describe the current call's nonzero set.
# Using them avoids fill!(Δc, 0) over all n columns every iteration.
function computeΔc2!(Δc::Array{Float64}, Δc_nz::Vector{Int}, Δc_flag::Vector{Bool},
                     tA::SparseMatrixCSC{Float64,Int64}, el::Array{Float64},
                     pfi::PFI, l::Int64, eps::Float64)
    n_cols = size(tA)[1]
    m      = size(tA)[2]

    fill!(el, 0.0)
    el[l] = 1.0
    btran!(pfi, el)

    # Sparse reset: zero only entries written last call.
    @inbounds for k in Δc_nz
        Δc[k]      = 0.0
        Δc_flag[k] = false
    end
    empty!(Δc_nz)

    # Scatter B⁻ᵀeₗ through A (via tA = Aᵀ stored column-major).
    @inbounds for i in 1:m
        eli = el[i]
        abs(eli) > eps || continue
        for ptr in tA.colptr[i]:(tA.colptr[i+1]-1)
            j = tA.rowval[ptr]
            Δc[j] += eli * tA.nzval[ptr]
            if !Δc_flag[j]
                Δc_flag[j] = true
                push!(Δc_nz, j)
            end
        end
    end

    # Slack columns: Δc[n_cols-m+i] = el[i].  The scatter above already
    # produces this via the identity block in tA, but we overwrite for
    # exactness (avoids accumulating eps-filtered approximation error).
    slack_base = n_cols - m
    @inbounds for i in 1:m
        v = el[i]
        v == 0.0 && continue
        j = slack_base + i
        if !Δc_flag[j]
            Δc_flag[j] = true
            push!(Δc_nz, j)
        end
        Δc[j] = v
    end
end

function solve(p; time_limit=-1)
    p = deepcopy(p)
    print("$(size(p.A)) ")
    presolution = Dict{String, Float64}()
    col_scaling = presolve!(p, presolution)
    print("$(size(p.A)) ")
    (flipped, split_pairs) = handle_negative_lowerbound_variables!(p)
    add_upper_bounds!(p)
    add_lower_bounds!(p)
    #add_free_variables!(p)
    add_slack_variables!(p)
    print("$(size(p.A)) ")
    solution = solve(p.A, p.c, p.b; time_limit=time_limit)
    solution2 = Dict{String, Float64}()
    for i in 1:length(p.c_names)
        raw = get(solution, i, 0.0)
        if i in flipped
            solution2[p.c_names[i]] = -raw * col_scaling[i]
        elseif haskey(split_pairs, i)
            raw_neg = get(solution, split_pairs[i], 0.0)
            solution2[p.c_names[i]] = (raw - raw_neg) * col_scaling[i]
        else
            solution2[p.c_names[i]] = raw * col_scaling[i]
        end
    end
    merge!(solution2, presolution)
    return solution2
end

function solve(A::SparseMatrixCSC{Float64, Int64},c::Array{Float64,1},b::Array{Float64,1}; time_limit=-1)
    try; SuiteSparse.UMFPACK.umf_ctrl[8] = 0; catch; end  # silence UMFPACK output; field removed in Julia 1.11+

    @assert size(A)[1] == length(b) "Size of b and size of A do not match: $(size(A)[1]) != $(length(b))"
    @assert size(A)[2] == length(c) "Size of c and size of A do not match: $(size(A)[2]) != $(length(c))"
    @assert count(x -> isnan(x), b) == 0
    @assert count(x -> isnan(x), c) == 0

    (Is, Js, Vs) = findnz(A)
    tA = sparse(Js, Is, Vs, size(A)[2], size(A)[1])

    eps = 1e-8
    t = 0.0

    n_constraints = length(b)
    refactor_pivot_cap = max(100, min(500, n_constraints ÷ 4))
    refactor_fill_threshold = 10 * n_constraints

    b_scale = max(1.0, norm(b) / sqrt(n_constraints))
    c_nz = length(c) - n_constraints
    c_scale = c_nz > 0 ? max(1.0, norm(c[1:c_nz]) / sqrt(c_nz)) : 1.0

    basis = collect(length(c) - length(b) + 1:length(c))
    is_basic = zeros(Bool, length(c))
    for b in basis
        is_basic[b] = true
    end
    pfi = PFI()
    #LUelimination!(pfi, A, basis)
    pfi = LUdecomposition(A, basis)
    basis = pfi.basis

 
    b_hat = copy(b) # values of basic variables
    c_hat = copy(c) # values of dual variables

    # Initial c-perturbation scaled by column norms of A.  At initialization B=I,
    # so ||B⁻¹Aⱼ|| = ||Aⱼ||, and perturbation_c[j] ∝ ||Aⱼ|| gives exact steepest-edge
    # pricing for primal steps: argmax(-ĉ[j]/perturbation_c[j]) = argmax(|ĉ[j]|/||Aⱼ||).
    col_norms = zeros(Float64, c_nz)
    @inbounds for j in 1:c_nz
        for k in A.colptr[j]:(A.colptr[j+1]-1)
            col_norms[j] += A.nzval[k]^2
        end
        col_norms[j] = sqrt(col_norms[j])
    end
    perturbation_c = vcat(c_scale .* max.(1.0, col_norms) .* (1.0 .+ 0.1 .* rand(Float64, c_nz)),
                          zeros(Float64, n_constraints))
    perturbation_c_hat = copy(perturbation_c)
    perturbation_b = b_scale .* (1.0 .+ 5.0 .* rand(Float64, n_constraints))
    perturbation_b_hat = copy(perturbation_b)
    
    old_t = Inf64
    iter = 0

    leaving = -1
    j = -1
    el = zeros(length(b))
    Δb = zeros(length(b))
    Δc = zeros(length(c))
    # Sparse-Δc tracking: Δc_nz holds the indices written last iteration;
    # Δc_flag[j] is true iff Δc[j] is currently nonzero.
    Δc_nz   = Int[]
    sizehint!(Δc_nz, length(c))
    Δc_flag = zeros(Bool, length(c))

    # Devex weight estimates.  devex_w_row[i] ≈ ||B⁻ᵀeᵢ||²,  devex_w_col[j] ≈ ||B⁻¹Aⱼ||².
    # Updated via recurrence after each pivot; used at refactorization to set the next
    # phase's perturbation ∝ sqrt(weight), biasing pricing toward steepest-edge pivots.
    devex_w_row = ones(Float64, n_constraints)
    devex_w_col = vcat(max.(1.0, col_norms) .^ 2, ones(Float64, n_constraints))

    forced_refactoring = false
    primal_count = 0
    dual_count = 0
    total_eta_nnz = 0
    start_ns = time_ns()
    time_limit_ns = time_limit > 0 ? UInt64(time_limit) * UInt64(1_000_000_000) : typemax(UInt64)
    while true
        #@assert sum(is_basic) == length(basis)
        iter = iter + 1

        # Pricing: compute t_b, t_c, and the argmax candidates simultaneously.
        # leaving/j are set to the exact argmax (the variable achieving the
        # maximum parametric ratio).  The SDS dual step requires the leaving row
        # to achieve t_b — any smaller value stalls progress because the basis
        # update step size b_hat[leaving]/Δb[leaving] ≈ 0.
        t_b = 0.0; t_c = 0.0; leaving = -1; j = -1
        @inbounds for i in 1:length(b_hat)
            bi = b_hat[i]
            if bi < 0.0
                v = -bi / perturbation_b_hat[i]
                if v > t_b; t_b = v; leaving = i; end
            end
        end
        @inbounds for i in 1:length(c_hat)
            ci = c_hat[i]
            if ci < 0.0 && !is_basic[i]
                v = -ci / perturbation_c_hat[i]
                if v > t_c; t_c = v; j = i; end
            end
        end
        t = max(t_b, t_c)
        
        #@assert t <= old_t "$t $old_t"
        old_t = t

        if iter % 15000 == 0
            r = zeros(length(c))
            for (i,v) in Dict(zip(basis, b_hat))
                r[i] = v
                @assert !isnan(v) "($(i), $(v))"
            end
            println("$iter $(sum(c .* r)) t:$(max(t_b,t_c)) t_b:$t_b t_c:$t_c primals:$primal_count duals:$dual_count") 
        end
        
        if t_b <= eps && t_c <= eps
            @debug "$iter t_b: $t_b t_c: $t_c "#b_hat: $(b_hat[leaving]) perturbation_b_hat: $(perturbation_b_hat[leaving]) c_hat: $(c_hat) perturbation_c_hat: $(perturbation_c_hat)"
            println("primal: $primal_count dual: $dual_count")
            return Dict(zip(basis, b_hat))
        end
        
        try
        #if t_b > t_c
        if t_b > t_c
            dual_count += 1
            #dual simplex step   
            #computeΔc!(Δc, A, el, pfi, is_basic, leaving)
            #Δc2 = copy(Δc)
            computeΔc2!(Δc, Δc_nz, Δc_flag, tA, el, pfi, leaving, eps)
            # for i in 1:length(Δc)
            #     @assert Δc[i] ≈ Δc2[i] "$(Δc[i]) ≈ $(Δc2[i])"
            # end
                
            # Dual ratio test: scan only Δc_nz (entries written by computeΔc2!).
            j = -1; min_ratio_j = Inf64
            @inbounds for i in Δc_nz
                if !is_basic[i] && Δc[i] < -eps
                    r = (c_hat[i] + t * perturbation_c_hat[i]) / -Δc[i]
                    if r < min_ratio_j; min_ratio_j = r; j = i; end
                end
            end
            if j == -1
                @debug "$iter t_b: $t_b t_c: $t_c entering:? leaving:$(basis[leaving])"
                throw(ErrorException("Infeasible/Unbounded (minJ)"))
            end

            get_column!(Δb, A, j)
            ftran!(pfi, Δb)
        else
            primal_count += 1
            #primal simplex step
            get_column!(Δb, A, j)
            ftran!(pfi, Δb)
            
            leaving = -1; minL = Inf64
            @inbounds for i in 1:length(b_hat)
                if Δb[i] > eps
                    r = (b_hat[i] + t * perturbation_b_hat[i]) / Δb[i]
                    if r < minL; minL = r; leaving = i; end
                end
            end
            if leaving == -1
                @debug "$iter t_b: $t_b t_c: $t_c entering: $j leaving:?"
                throw(ErrorException("Infeasible/Unbounded (minL)"))
            end

            #computeΔc!(Δc, A, el, pfi, is_basic, leaving)
            computeΔc2!(Δc, Δc_nz, Δc_flag, tA, el, pfi, leaving, eps)
        end
        catch
            #forced_refactoring = true
            rethrow()
        end

        @debug "$iter t_b: $t_b t_c: $t_c entering:$j leaving:$(basis[leaving]) b_hat: $(b_hat[leaving]) perturbation_b_hat: $(perturbation_b_hat[leaving]) Δb: $(Δb[leaving]) c_hat: $(c_hat[j]) perturbation_c_hat: $(perturbation_c_hat[j]) Δc: $(Δc[j])"

        if Δb[leaving] == 0 || Δc[j] == 0
            println("$iter Changing perturbation")
            p_scale = 1.0 / 100.0
            perturbation_c = vcat((c_scale * p_scale) .* (1.0 .+ 5.0 .* rand(Float64, c_nz)),
                                  zeros(Float64, n_constraints))
            perturbation_b = (b_scale * p_scale) .* (1.0 .+ 5.0 .* rand(Float64, n_constraints))
            perturbation_b_hat = copy(perturbation_b)
            ftran!(pfi, perturbation_b_hat)
            multipliers = c[basis]
            perturbation_multipliers = perturbation_c[basis]
            btran!(pfi, multipliers)
            btran!(pfi, perturbation_multipliers)
            mul!(c_hat, tA, multipliers)
            @inbounds for i in 1:length(c_hat)
                c_hat[i] = c[i] - c_hat[i]
            end
            mul!(perturbation_c_hat, tA, perturbation_multipliers)
            @inbounds for i in 1:length(perturbation_c_hat)
                perturbation_c_hat[i] = perturbation_c[i] - perturbation_c_hat[i]
            end
            fill!(devex_w_row, 1.0)
            fill!(devex_w_col, 1.0)
            continue
        end
        if !forced_refactoring
            #update basis
            sΔb = sparsevec(Δb)
            droptol!(sΔb, 1e-10)
            η = ETAMatrix(leaving, sΔb)
            push!(pfi.eta_matrices, η)
            total_eta_nnz += nnz(sΔb)

            leaving_var = basis[leaving]
            db_sq = 0.0
            @inbounds begin
                t_step   = b_hat[leaving]              / Δb[leaving]
                t_step_p = perturbation_b_hat[leaving] / Δb[leaving]
                for i in 1:length(Δb)
                    d = Δb[i]
                    b_hat[i]              -= t_step   * d
                    perturbation_b_hat[i] -= t_step_p * d
                    db_sq += d * d
                end
                b_hat[leaving]              = t_step
                perturbation_b_hat[leaving] = t_step_p
            end

            # Devex recurrence: update row weights (devex_w_row[i] ≈ ||B⁻ᵀeᵢ||²)
            # and set column weight for the newly-nonbasic variable.
            @inbounds begin
                piv     = Δb[leaving]
                piv_sq  = piv * piv
                w_new   = max(1.0, db_sq / piv_sq)
                devex_w_row[leaving] = w_new
                inv_piv = 1.0 / piv
                for i in 1:length(Δb)
                    i == leaving && continue
                    r = Δb[i] * inv_piv
                    approx = r * r * w_new
                    if approx > devex_w_row[i]; devex_w_row[i] = approx; end
                end
                devex_w_col[leaving_var] = w_new
            end

            # Sparse c_hat update: only iterate over Δc_nz (the nonzero
            # indices of Δc filled by computeΔc2!), skipping the ~(n-nnz_Δc)
            # zero entries that the old dense loop wasted work on.
            @inbounds begin
                s_step   = c_hat[j]              / Δc[j]
                s_step_p = perturbation_c_hat[j] / Δc[j]
                for k in Δc_nz
                    d = Δc[k]
                    c_hat[k]              -= s_step   * d
                    perturbation_c_hat[k] -= s_step_p * d
                end
                c_hat[leaving_var]              = -s_step
                perturbation_c_hat[leaving_var] = -s_step_p
            end

            is_basic[leaving_var] = false
            is_basic[j] = true
            basis[leaving] = j

            for i in basis
                c_hat[i] = 0
                perturbation_c_hat[i] = 0
            end
        end

        if total_eta_nnz > refactor_fill_threshold || length(pfi.eta_matrices) >= refactor_pivot_cap || forced_refactoring
            forced_refactoring = false
            total_eta_nnz = 0

            # Update perturbation using Devex weights accumulated since the last refactor.
            # perturbation_b[i] ∝ sqrt(devex_w_row[i]) ≈ ||B⁻ᵀeᵢ|| makes pricing equivalent
            # to steepest-edge: argmax(-b̂[i]/perturbation_b̂[i]) = argmax(|b̂[i]|/||B⁻ᵀeᵢ||).
            @inbounds for i in 1:n_constraints
                perturbation_b[i] = b_scale * sqrt(devex_w_row[i]) * (1.0 + 0.1 * rand())
            end
            @inbounds for j in 1:c_nz
                perturbation_c[j] = c_scale * sqrt(devex_w_col[j]) * (1.0 + 0.1 * rand())
            end
            fill!(devex_w_row, 1.0)
            fill!(devex_w_col, 1.0)

            #LUelimination!(pfi, A, basis)
            pfi = LUdecomposition(A, basis)
            basis = pfi.basis

            b_hat = copy(b)
            ftran!(pfi, b_hat)
            @assert count(x -> isnan(x), b_hat) == 0

            perturbation_b_hat = copy(perturbation_b)
            ftran!(pfi, perturbation_b_hat)

            multipliers = c[basis]
            perturbation_multipliers = perturbation_c[basis]
            btran!(pfi, multipliers)
            btran!(pfi, perturbation_multipliers)
            mul!(c_hat, tA, multipliers)
            @inbounds for i in 1:length(c_hat)
                c_hat[i] = c[i] - c_hat[i]
            end
            mul!(perturbation_c_hat, tA, perturbation_multipliers)
            @inbounds for i in 1:length(perturbation_c_hat)
                v = perturbation_c[i] - perturbation_c_hat[i]
                perturbation_c_hat[i] = abs(v) < eps ? 0.0 : v
            end

            for i in basis
                #@assert abs(c_hat[i]) < 1e-12 "$(abs(c_hat[i]))"
            end
        end
        if iter % 1000 == 0 && time_ns() - start_ns > time_limit_ns
            break
        end
    end
    throw(ErrorException("Time limit reached after $iter iterations (t_b=$t_b, t_c=$t_c)"))
end

function updateBasicVariables(b_hat, Δb, leaving)
    @inbounds t = b_hat[leaving] / Δb[leaving]
    #@assert !isnan(t) & !isinf(t) "t_b: $t_b t_c: $t_c b_hat: $(b_hat[leaving]) perturbation_b_hat: $(perturbation_b_hat[leaving]) Δb: $(Δb[leaving]) c_hat: $(c_hat[j]) perturbation_c_hat: $(perturbation_c_hat[j]) Δc: $(Δc[j]) leaving: $(basis[leaving]) entering: $j"
    axpy!(-t, Δb, b_hat)
    @inbounds b_hat[leaving] = t
end

function updateDualVariables(c_hat, Δc, j, leaving, basis)
    @inbounds s = c_hat[j] / Δc[j]
    #@assert !isnan(s) & !isinf(s) "t_b:$t_b t_c:$t_c c_hat: $(c_hat[j]) Δc: $(Δc[j]) b_hat: $(b_hat[leaving]) Δb: $(Δb[leaving]) leaving: $(basis[leaving]) entering: $j"
    axpy!(-s, Δc, c_hat)
    @inbounds c_hat[basis[leaving]] = -s
end

end # module
