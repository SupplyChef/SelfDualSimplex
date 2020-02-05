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

"Splits variables into a variable for the positive part and a variable for the negative part"
function handle_negative_lowerbound_variables!(p)
    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]

    lc = length(p.c)

    for i in 1:length(p.lower_bounds)
        if p.lower_bounds[i] < 0 && p.upper_bounds[i] <= 0
            p.c[i] = -p.c[i]
            (Is, Vs) = get_nz(p.A, i)
            for k in 1:length(Is)
                p.A[Is[k], i] = -Vs[k]
            end
            l = p.lower_bounds[i]
            p.lower_bounds[i] = -p.upper_bounds[i]
            p.upper_bounds[i] = -l
        end
    end

    bound_count = count((p.lower_bounds .< 0) .& (p.upper_bounds .> 0))
    
    resize!(p.c, length(p.c) + bound_count)
    resize!(p.lower_bounds, length(p.lower_bounds) + bound_count)
    resize!(p.upper_bounds, length(p.upper_bounds) + bound_count)
    (Is, Js, Vs) = findnz(p.A)
    (m, n) = size(p.A)
    #p.A = resize(p.A, size(p.A)[1], size(p.A)[2] + bound_count)
    j = 1
    for i in 1:length(p.lower_bounds)
        if p.lower_bounds[i] < 0 && p.upper_bounds[i] > 0
            #p.A[:, lc + j] = -p.A[:, i]
            new_column = -p.A[:, i]
            append!(Is, new_column.nzind)
            append!(Js, repeat([n + j], length(new_column.nzind)))
            append!(Vs, new_column.nzval)
    
            p.c[lc + j] = -p.c[i]
            p.lower_bounds[lc + j] = 0.0
            p.upper_bounds[lc + j] = -p.lower_bounds[i]

            p.lower_bounds[i] = 0.0
            
            j += 1
        end
    end

    p.A = sparse(Is, Js, Vs, m, n + bound_count)
    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]
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
            maxV = maximum(sqrt.(abs.(Vs)))
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
    row_scaling = row_scaling[active]

    (Is, Js, Vs) = findnz(tA)
    p.A = sparse(Js, Is, Vs, size(tA)[2], size(tA)[1])
    
    d = Diagonal(row_scaling)
    p.A = d * p.A
    p.b = d * p.b
end
