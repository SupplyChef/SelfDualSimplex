using SparseArrays
using LinearAlgebra
using Printf

function LUdecomposition(m::SparseMatrixCSC{Float64, Int64}, basis)::Tuple{PFI, Array{Int64}}
    luf  = lu(m[:,basis])
    return PFI(ETAMatrix[], luf), basis
end

function LUelimination(m::SparseMatrixCSC{Float64, Int64}, basis)::Tuple{PFI, Array{Int64}}
    tmp = [get_column(m, b) for b in basis]
    map_to_new_indexes = collect(1:length(basis))

    tmp_row_nzs = [Array{Int64,1}() for row in 1:length(basis)]
    for j in 1:length(basis)
        for i in tmp[j].nzind
            push!(tmp_row_nzs[i], j)
        end
    end
    
    ls = ETAMatrix[]
    us = ETAMatrix[]

    col_nnzs = zeros(Int64, length(basis))
    @inbounds for t in 1:length(basis)
        col_nnzs[t] = get_nnz(tmp[t])
    end

    row_j = zeros(length(basis))
    @inbounds for j in 1:length(basis)
        # println(j)
        # A = zeros(Float64, (length(tmp),length(tmp)))
        # for u in 1:length(tmp)
        #     A[:, u] = tmp[u]
        # end
        # for v in 1:length(tmp)
        #     println(A[v, :])
        # end

        max_pivot = 0.0
        @inbounds for tmp_j in tmp_row_nzs[j]
            r = tmp[tmp_j][j]
            if r > max_pivot || -r > max_pivot
                max_pivot = abs(r)
            end
            row_j[tmp_j] = r
        end

        if max_pivot == 0
            throw(ErrorException("Singular basis"))# $j $basis"))
        end

        min_nnz = Inf64
        tmp_k = -1
        @inbounds for tmp_j in tmp_row_nzs[j]
            if abs(row_j[tmp_j]) >= 0.05 * max_pivot #|| abs(row_j[tmp_j]) >= 0.1
                if col_nnzs[tmp_j] < min_nnz
                    min_nnz = col_nnzs[tmp_j]
                    tmp_k = tmp_j
                end
            end
        end
        
        map_to_new_indexes[j] = tmp_k
        pivot = row_j[tmp_k]
        
        lfactors = spzeros(length(basis))
        ufactors = spzeros(length(basis))
        @inbounds for k in 1:length(tmp[tmp_k].nzind)
            v = tmp[tmp_k].nzval[k]
            if abs(v) < 1e-15
                continue
            end
            i = tmp[tmp_k].nzind[k]
            if i < j
                ufactors[i] = v
            elseif i > j
                lfactors[i] = v / pivot
            end
        end
        
        tmp_k2 = copy(tmp[tmp_k])
        @inbounds for l in tmp_row_nzs[j]
            t = row_j[l] / pivot 
            if t != 0
                tmp[l] = eliminate2(tmp_k2, tmp[l], j, t, tmp_row_nzs, l)
            end
        end

        if length(lfactors.nzind) > 0
            push!(ls, ETAMatrix(j, 1.0, lfactors))
        end
        if pivot != 1.0 || length(ufactors.nzind) > 0
            push!(us, ETAMatrix(j, pivot, ufactors))
        end

        for tmp_j in tmp_row_nzs[j]
            row_j[tmp_j] = 0.0
        end
    end
    basis = basis[map_to_new_indexes]

    @debug "LU factorization: $(length(ls)) lower Eta matrices, $(length(us)) upper Eta matrices."
    pfi = PFI(vcat(ls,reverse(us)), nothing)
    return pfi, basis
end

function push_if_not_present!(a, new)
    found = false
    for old in a
        if old == new
            found = true
            break
        end
    end
    if !found 
        push!(a, new)
    end
end

function eliminate(a::SparseVector{Float64, Int64}, b::SparseVector{Float64, Int64}, j::Int64, f::Float64, nzs, l)::SparseVector{Float64, Int64}
    Ia = a.nzind
    Va = a.nzval
    Ib = b.nzind
    Vb = b.nzval
    lia = length(Ia)
    lib = length(Ib)
    Ic = Array{Int64}(undef, lia + lib)
    Vc = Array{Float64}(undef, lia + lib)
    ia = 1
    ib = 1
    ic = 1
    while ia <= lia || ib <= lib
        if ia > lia
            # we drain the b vector
            @inbounds Ic[ic] = Ib[ib]
            @inbounds Vc[ic] = Vb[ib]
            ib += 1
            ic += 1
            continue
        end
        if ib > lib
            # we drain the a vector
            if Ia[ia] > j
                @inbounds Ic[ic] = Ia[ia]
                @inbounds Vc[ic] = 0 - Va[ia] * f
                ic += 1
                push_if_not_present!(nzs[Ia[ia]], l)
            end
            ia += 1
            continue
        end
        Iib = Ib[ib]
        Iia = Ia[ia]
        if Iia < Iib
            if Iia > j
                @inbounds Ic[ic] = Iia
                @inbounds Vc[ic] = 0 - Va[ia] * f
                ic += 1
                push_if_not_present!(nzs[Iia], l)
            end
            ia += 1        
            continue
        end
        if Iia == Iib
            if Iia > j
                if abs(Vb[ib] - Va[ia] * f) > 1e-15
                    @inbounds Ic[ic] = Iib
                    @inbounds Vc[ic] = Vb[ib] - Va[ia] * f
                    ic += 1
                end
            else
                @inbounds Ic[ic] = Iib
                @inbounds Vc[ic] = Vb[ib]
                ic += 1
            end
            ia += 1
            ib += 1
            continue
        end
        if Iia > Iib
            @inbounds Ic[ic] = Iib
            @inbounds Vc[ic] = Vb[ib]
            ib += 1
            ic += 1
            continue
        end
    end
    resize!(Ic, ic - 1)
    resize!(Vc, ic - 1)
    return SparseVector(length(b), Ic, Vc)
end

function eliminate2(a::SparseVector{Float64, Int64}, b::SparseVector{Float64, Int64}, j::Int64, f::Float64, nzs, l)::SparseVector{Float64, Int64}
    @inbounds for ia in 1:length(a.nzind)
        i = a.nzind[ia] 
        if i > j
            v = a.nzval[ia]
            old_v = b[i]
            new_v = old_v - f * v
            if abs(new_v) < 1e-15
                b[i] = 0
            else
                b[i] = new_v 

                push_if_not_present!(nzs[i], l)
            end
        end
    end
    return b
end