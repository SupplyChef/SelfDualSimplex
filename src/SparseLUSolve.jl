using SparseArrays
using SuiteSparse.UMFPACK

# Hyper-sparse triangular solvers.
#
# UMFPACK factorization identity: L * U = Rs .* A[p, q]
# where (Rs .* A[p,q])[i,j] = Rs[i] * A[p[i], q[j]]
#
# FTRAN (solve B*y = x in-place):
#   work[i] = Rs[i] * x[p[i]]   (gather + scale)
#   sparse_lsolve_unit!(L, work)  (forward, unit lower tri)
#   sparse_usolve!(U, work)       (backward, upper tri)
#   x[q[i]] = work[i]             (scatter by q)
#
# BTRAN (solve B^T*y = x in-place):
#   work[i] = x[q[i]]             (gather by q)
#   sparse_lsolve!(Ut, work)      (forward on U^T, lower tri)
#   sparse_usolve_unit!(Lt, work) (backward on L^T, unit upper tri)
#   x[p[i]] = Rs[i] * work[i]    (scatter + scale)
#
# Hyper-sparse benefit: skip column j when x[j] == 0.
# For btran from e_l: reach is O(sqrt(m)) vs O(m) for dense solve.
# For ftran from sparse A column: similar savings.

mutable struct SparseLUSolver
    L::SparseMatrixCSC{Float64,Int64}   # unit lower triangular
    U::SparseMatrixCSC{Float64,Int64}   # upper triangular
    Ut::SparseMatrixCSC{Float64,Int64}  # U transposed (lower tri, same diag as U)
    Lt::SparseMatrixCSC{Float64,Int64}  # L transposed (unit upper tri)
    p::Vector{Int64}                    # row permutation (1-indexed)
    q::Vector{Int64}                    # col permutation (1-indexed)
    Rs::Vector{Float64}                 # row scaling
    u_diag::Vector{Float64}            # diagonal of U (and Ut), precomputed
    work::Vector{Float64}              # reusable work buffer (length n)

    function SparseLUSolver(luf::UmfpackLU{Float64,Int64})
        L  = luf.L
        U  = luf.U
        p  = luf.p
        q  = luf.q
        Rs = luf.Rs
        n  = size(L, 1)

        Ut = SparseMatrixCSC{Float64,Int64}(permutedims(U))
        Lt = SparseMatrixCSC{Float64,Int64}(permutedims(L))

        # Precompute diagonal of U. U is upper triangular CSC, so the
        # diagonal entry of column j is the last stored entry (largest row = j).
        u_diag = zeros(Float64, n)
        @inbounds for j in 1:n
            last_k = U.colptr[j+1] - 1
            if last_k >= U.colptr[j] && U.rowval[last_k] == j
                u_diag[j] = U.nzval[last_k]
            else
                # Fallback: linear scan (should not happen for valid UMFPACK output)
                for k in U.colptr[j]:(U.colptr[j+1]-1)
                    if U.rowval[k] == j
                        u_diag[j] = U.nzval[k]
                        break
                    end
                end
            end
        end

        work = zeros(Float64, n)
        return new(L, U, Ut, Lt, p, q, Rs, u_diag, work)
    end
end

# Forward solve: L * x = x in-place. L is unit lower triangular (CSC).
# Skip column j entirely when x[j] == 0 (hyper-sparse).
@inline function sparse_lsolve_unit!(L::SparseMatrixCSC{Float64,Int64}, x::Vector{Float64})
    @inbounds for j in 1:size(L, 1)
        xj = x[j]
        iszero(xj) && continue
        for k in L.colptr[j]:(L.colptr[j+1]-1)
            row = L.rowval[k]
            row > j && (x[row] -= L.nzval[k] * xj)
        end
    end
end

# Backward solve: U * x = x in-place. U is upper triangular (CSC), non-unit diagonal.
# Skip column j when x[j] == 0 (hyper-sparse from the right).
@inline function sparse_usolve!(U::SparseMatrixCSC{Float64,Int64}, u_diag::Vector{Float64}, x::Vector{Float64})
    @inbounds for j in size(U, 1):-1:1
        xj = x[j]
        iszero(xj) && continue
        xj = xj / u_diag[j]
        x[j] = xj
        for k in U.colptr[j]:(U.colptr[j+1]-1)
            row = U.rowval[k]
            row < j && (x[row] -= U.nzval[k] * xj)
        end
    end
end

# Forward solve: Ut * x = x in-place. Ut = U^T is lower triangular (CSC), non-unit diagonal.
@inline function sparse_lsolve!(Ut::SparseMatrixCSC{Float64,Int64}, u_diag::Vector{Float64}, x::Vector{Float64})
    @inbounds for j in 1:size(Ut, 1)
        xj = x[j]
        iszero(xj) && continue
        xj = xj / u_diag[j]
        x[j] = xj
        for k in Ut.colptr[j]:(Ut.colptr[j+1]-1)
            row = Ut.rowval[k]
            row > j && (x[row] -= Ut.nzval[k] * xj)
        end
    end
end

# Backward solve: Lt * x = x in-place. Lt = L^T is unit upper triangular (CSC).
@inline function sparse_usolve_unit!(Lt::SparseMatrixCSC{Float64,Int64}, x::Vector{Float64})
    @inbounds for j in size(Lt, 1):-1:1
        xj = x[j]
        iszero(xj) && continue
        for k in Lt.colptr[j]:(Lt.colptr[j+1]-1)
            row = Lt.rowval[k]
            row < j && (x[row] -= Lt.nzval[k] * xj)
        end
    end
end

# FTRAN: solve B*y = x in-place using hyper-sparse triangular solves.
function slu_ftran!(slu::SparseLUSolver, x::Vector{Float64})
    n  = length(x)
    p  = slu.p
    Rs = slu.Rs
    q  = slu.q
    w  = slu.work

    @inbounds for i in 1:n
        pi = p[i]
        w[i] = Rs[pi] * x[pi]
    end
    sparse_lsolve_unit!(slu.L, w)
    sparse_usolve!(slu.U, slu.u_diag, w)
    @inbounds for i in 1:n
        x[q[i]] = w[i]
    end
end

# BTRAN: solve B^T*y = x in-place using hyper-sparse triangular solves.
function slu_btran!(slu::SparseLUSolver, x::Vector{Float64})
    n  = length(x)
    p  = slu.p
    Rs = slu.Rs
    q  = slu.q
    w  = slu.work

    @inbounds for i in 1:n
        w[i] = x[q[i]]
    end
    sparse_lsolve!(slu.Ut, slu.u_diag, w)
    sparse_usolve_unit!(slu.Lt, w)
    @inbounds for i in 1:n
        pi = p[i]
        x[pi] = Rs[pi] * w[i]
    end
end
