using LinearAlgebra

struct ETAMatrix <: AbstractArray{Float64, 2}
    column_index::Int64
    eta_pivot::Float64
    eta_other_vector::SparseVector{Float64, Int64}
    function ETAMatrix(column_index::Int64, eta_vector::SparseVector{Float64, Int64})
        eta_pivot = eta_vector[column_index]
        eta_other_vector = eta_vector
        eta_other_vector[column_index] = 0

        e = new(column_index, eta_pivot, eta_other_vector)
        return e
    end
    function ETAMatrix(column_index::Int64, eta_pivot::Float64, eta_other_vector::SparseVector{Float64, Int64})
        e = new(column_index, eta_pivot, eta_other_vector)
        return e
    end
end

function inv(e::ETAMatrix)::ETAMatrix
    return ETAMatrix(e.column_index, 1.0 / e.eta_pivot, -1.0 / e.eta_pivot * e.eta_other_vector)
end

function size(e::ETAMatrix)
    return (size(e.eta_other_vector)[1], size(e.eta_other_vector)[1])
end

function getindex(e::ETAMatrix, I::Vararg{Int, 2})
    i = I[1]
    j = I[2]
    if j != e.column_index
        return i == j ? 1.0 : 0.0
    else
        if i == e.column_index
            return e.eta_pivot
        else
            return e.eta_other_vector[i]
        end
    end
end

function setindex!(e::ETAMatrix, v, I::Vararg{Int, 2})
    throw(ErrorException("Cannot set an element of an ETA matrix"))
end

function get_column(m::SparseMatrixCSC{Float64, Int64}, j::Int64)::SparseVector{Float64,Int64}
    return sparsevec(m[:,j])
end

function get_nz(m::SparseMatrixCSC{Float64, Int64}, j::Int64)::Tuple{Array{Int64,1},Array{Float64,1}}
    return (m.rowval[m.colptr[j]:(m.colptr[j+1]-1)], m.nzval[m.colptr[j]:(m.colptr[j+1]-1)])
end

function get_nz(m::SparseVector{Float64, Int64})::Tuple{Array{Int64,1},Array{Float64,1}}
    return (copy(m.nzind), copy(m.nzval))
end

function get_nnz(m::SparseMatrixCSC{Float64, Int64}, j::Int64)::Int64
    return (m.colptr[j+1]-1) - m.colptr[j] + 1
end

function get_nnz(m::SparseVector{Float64, Int64})::Int64
    return length(m.nzind)
end

function get_nnz(m::SparseMatrixCSC{Float64, Int64}, j::Int64, t::Int64)
    count = 0
    for i in m.colptr[j]:(m.colptr[j+1]-1)
        if m.rowval[i] > t
            count += 1
        end
    end
    return count
end

function dot(s::SparseMatrixCSC{Float64, Int64}, j::Int64, x::Array{Float64, 1})    
    total = 0.0
    @inbounds for k in s.colptr[j]:(s.colptr[j+1]-1)
        xv = x[s.rowval[k]]
        if xv != 0.0
            total += s.nzval[k] * xv
        end
    end
    return total
end

function dot(s::SparseVector{Float64, Int64}, x::Array{Float64, 1})::Float64
    total = 0.0
    @inbounds for i in 1:length(s.nzind)
        total += s.nzval[i] * x[s.nzind[i]]
    end
    return total
end

function ftran!(e::ETAMatrix, x::Array{Float64, 1})
    #eta-ftran (x := E^{−1}x) : xp := xp/ηp and then x := x − xp.η
    if abs(x[e.column_index]) < 1e-12
        #noop
    else
        @inbounds x[e.column_index] /= e.eta_pivot
        #@assert !isnan(x[e.column_index])
        #@assert !isinf(x[e.column_index])
        xp = x[e.column_index]
        eta = e.eta_other_vector
        @inbounds for i in 1:length(eta.nzind)
            x[eta.nzind[i]] -= xp * eta.nzval[i]
            #@assert !isnan(x[eta.nzind[i]])
            #@assert !isinf(x[eta.nzind[i]])
        end
    end
end

function btran!(e::ETAMatrix, x::Array{Float64, 1})
    #x'.E = x' (or E'.x = x)
    #eta-btran (x := E^{−T}x) : xp := (xp − x'η)/ηp 
    @inbounds x[e.column_index] -= dot(e.eta_other_vector, x) 
    @inbounds x[e.column_index] /= e.eta_pivot
    #@assert !isnan(x[e.column_index]) "$(e.eta_pivot)"
    #@assert !isinf(x[e.column_index]) "$(e.eta_pivot)"
end