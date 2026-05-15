using SparseArrays

"Add upper bounds as constraints in the main matrix"
function add_upper_bounds!(p)
    @assert length(p.b) == size(p.A)[1] "$(length(p.b)) != $(size(p.A)[1])"
    @assert length(p.c) == size(p.A)[2]

    lb = length(p.b)
    lc = length(p.c)

    bound_count = count(p.upper_bounds .< Inf64)
    
    resize!(p.b, length(p.b) + bound_count)

    (Is, Js, Vs) = findnz(p.A)

    #p.A = resize(p.A, size(p.A)[1] + bound_count, size(p.A)[2])
    j = 1
    for i in 1:length(p.upper_bounds)
        if p.upper_bounds[i] < Inf64
            p.b[lb + j] = p.upper_bounds[i]
            push!(Is, lb + j)
            push!(Js, i)
            push!(Vs, 1.0)            
            #p.A[lb + j, i] = 1.0
            j += 1
        end
    end
    p.A = sparse(Is, Js, Vs, size(p.A)[1] + bound_count, size(p.A)[2])

    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]
end

"Add lower bounds as constraints in the main matrix"
function add_lower_bounds!(p)
    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]

    lb = length(p.b)
    lc = length(p.c)    

    bound_count = count(p.lower_bounds .> 0.0)
    
    resize!(p.b, length(p.b) + bound_count)

    (Is, Js, Vs) = findnz(p.A)
    #p.A = resize(p.A, size(p.A)[1] + bound_count, size(p.A)[2])
    j = 1
    for i in 1:length(p.lower_bounds)
        if p.lower_bounds[i] > 0.0
            p.b[lb + j] = -p.lower_bounds[i]
            push!(Is, lb + j)
            push!(Js, i)
            push!(Vs, -1.0)            
            #p.A[lb + j, i] = -1.0
            j += 1
        end
    end
    p.A = sparse(Is, Js, Vs, size(p.A)[1] + bound_count, size(p.A)[2])

    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]
end

"Splits variables into a variable for the positive part and a variable for the negative part.
Returns (flipped, split_pairs) where flipped is the set of indices negated by Case 1, and
split_pairs maps each Case 2 index to the index of its newly created negative-part variable."
function handle_negative_lowerbound_variables!(p)
    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]

    lc = length(p.c)

    flipped = Set{Int}()
    for i in 1:lc
        if p.lower_bounds[i] < 0 && p.upper_bounds[i] <= 0
            p.c[i] = -p.c[i]
            (Is, Vs) = get_nz(p.A, i)
            for k in 1:length(Is)
                p.A[Is[k], i] = -Vs[k]
            end
            l = p.lower_bounds[i]
            p.lower_bounds[i] = -p.upper_bounds[i]
            p.upper_bounds[i] = -l
            push!(flipped, i)
        end
    end

    bound_count = count((p.lower_bounds .< 0) .& (p.upper_bounds .> 0))

    resize!(p.c, length(p.c) + bound_count)
    resize!(p.lower_bounds, length(p.lower_bounds) + bound_count)
    resize!(p.upper_bounds, length(p.upper_bounds) + bound_count)
    (Is, Js, Vs) = findnz(p.A)
    (m, n) = size(p.A)

    split_pairs = Dict{Int, Int}()
    j = 1
    for i in 1:lc
        if p.lower_bounds[i] < 0 && p.upper_bounds[i] > 0
            new_column = -p.A[:, i]
            append!(Is, new_column.nzind)
            append!(Js, repeat([n + j], length(new_column.nzind)))
            append!(Vs, new_column.nzval)

            p.c[lc + j] = -p.c[i]
            p.lower_bounds[lc + j] = 0.0
            p.upper_bounds[lc + j] = -p.lower_bounds[i]

            p.lower_bounds[i] = 0.0

            split_pairs[i] = lc + j
            j += 1
        end
    end

    p.A = sparse(Is, Js, Vs, m, n + bound_count)
    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]

    return (flipped, split_pairs)
end

function add_slack_variables!(p)
    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]

    lb = length(p.b)
    lc = length(p.c)

    resize!(p.c, length(p.c) + lb)
    #p.A = resize(p.A, size(p.A)[1], size(p.A)[2] + lb)
    (Is, Js, Vs) = findnz(p.A)
    for i in 1:lb
        p.c[lc + i] = 0.0
        push!(Is, i)
        push!(Js, lc + i)
        push!(Vs, 1.0)
        #p.A[i, lc + i] = 1.0
    end
    p.A = sparse(Is, Js, Vs, size(p.A)[1], size(p.A)[2] + lb)
end

function resize(a::SparseMatrixCSC{Float64, Int64}, m, n)
    (I, J, V) = findnz(a)
    return sparse(I, J, V, m, n)
end

function presolve!(p, presolution::Dict{String, Float64})
    # p.A * x <= p.b
    active = repeat([true], length(p.lower_bounds))
    for i in 1:length(p.lower_bounds)
        if p.lower_bounds[i] == p.upper_bounds[i]
            active[i] = false            
            if p.lower_bounds[i] != 0.0
                (Is, Vs) = get_nz(p.A, i)
                for k in 1:length(Is)
                    #println("p.b[$(Is[k])] $(p.b[Is[k]]) => $(p.b[Is[k]] - p.lower_bounds[i] * Vs[k])")
                    p.b[Is[k]] -= p.lower_bounds[i] * Vs[k]
                end
            end
            push!(presolution, p.c_names[i] => p.lower_bounds[i])
        end
    end

    p.A = p.A[:,active]
    p.c = p.c[active]
    p.upper_bounds = p.upper_bounds[active]
    p.lower_bounds = p.lower_bounds[active]
    p.c_names = p.c_names[active]

    active = repeat([true], length(p.b))
    (Is, Js, Vs) = findnz(p.A)
    tA = sparse(Js, Is, Vs, size(p.A)[2], size(p.A)[1])
    row_scaling = repeat([1.0], size(p.A)[1])
    for i in 1:size(tA)[2]
        (Is, Vs) = get_nz(tA, i)
        if length(Vs) > 0
            maxV = maximum(abs.(Vs))
            if maxV > 1.0
                row_scaling[i] = 1.0 / maxV
            end
        else
            if p.b[i] < 0.0
                throw(ErrorException("Infeasible: $i $(p.b[i])"))
            end
            active[i] = false
        end
    end
    tA = tA[:,active]
    p.b = p.b[active]
    p.equality_rows = p.equality_rows[active]
    row_scaling = row_scaling[active]

    (Is, Js, Vs) = findnz(tA)
    p.A = sparse(Js, Is, Vs, size(tA)[2], size(tA)[1])

    d = Diagonal(row_scaling)
    p.A = d * p.A
    p.b = d * p.b

    col_scaling = ones(Float64, size(p.A, 2))
    for j in 1:size(p.A, 2)
        col_range = p.A.colptr[j]:(p.A.colptr[j+1]-1)
        isempty(col_range) && continue
        max_val = maximum(abs, view(p.A.nzval, col_range))
        if max_val > 0.0
            s = 1.0 / max_val
            p.A.nzval[col_range] .*= s
            p.c[j] *= s
            p.lower_bounds[j] /= s
            p.upper_bounds[j] /= s
            col_scaling[j] = s
        end
    end
    return col_scaling
end

# For each equality row i, its slack (at column n_before_slacks+i) must equal zero.
# Enforce this by adding a trivial row [s_eq ≤ 0] plus its own slack, which is far
# cheaper than the previous approach of duplicating the full row with negated signs.
function add_equality_upper_bound_rows!(p, equality_rows, n_before_slacks)
    eq_indices = findall(equality_rows)
    isempty(eq_indices) && return

    m   = length(p.b)
    n   = size(p.A, 2)
    lc  = length(p.c)
    k   = length(eq_indices)

    (Is, Js, Vs) = findnz(p.A)
    resize!(p.b, m + k)
    resize!(p.c, lc + k)

    for (t, i) in enumerate(eq_indices)
        new_row       = m + t
        slack_col     = n_before_slacks + i   # the equality row's slack
        new_slack_col = lc + t                # slack for this new upper-bound row

        push!(Is, new_row);  push!(Js, slack_col);     push!(Vs,  1.0)
        push!(Is, new_row);  push!(Js, new_slack_col); push!(Vs,  1.0)

        p.b[new_row]        = 0.0
        p.c[new_slack_col]  = 0.0
    end

    p.A = sparse(Is, Js, Vs, m + k, n + k)
end
