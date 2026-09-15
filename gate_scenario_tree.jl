#    stage 1  DAM gate        13:00 D-1   1 node        nothing known yet
#    stage 2  IDA1 gate       15:00 D-1   K2 nodes      DAM results out at 14:00
#    stage 3  ISP deadline    16:45 D-1   K2*K3 nodes   IDA1 results out at 16:30
#    stage 4  IDA2 gate       22:00 D-1   K2*K3*K4      ISP1 indicative at 17:30
#    stage 5+ delivery                                  the day is now determined

using Statistics, Random, LinearAlgebra

function kmeans_cols(X::AbstractMatrix{Float64}, k::Int; iters::Int = 200, seed::Int = 1)
    d, n = size(X)
    k = min(k, n)
    rng = MersenneTwister(seed)
    centres = zeros(d, k)
    centres[:, 1] = X[:, rand(rng, 1:n)]
    for j in 2:k
        d2 = [minimum(sum(abs2, X[:, i] .- centres[:, l]) for l in 1:j-1) for i in 1:n]
        s = sum(d2)
        centres[:, j] = X[:, s <= 0 ? rand(rng, 1:n) : findfirst(>=(rand(rng) * s), cumsum(d2))]
    end
    assign = zeros(Int, n)
    for _ in 1:iters
        changed = false
        for i in 1:n
            best = argmin([sum(abs2, X[:, i] .- centres[:, j]) for j in 1:k])
            best == assign[i] || (assign[i] = best; changed = true)
        end
        for j in 1:k
            mem = findall(==(j), assign)
            isempty(mem) || (centres[:, j] = mean(X[:, mem], dims = 2))
        end
        changed || break
    end
    return assign, centres
end

struct Scaler; mu::Matrix{Float64}; sd::Matrix{Float64}; end
Scaler(X::AbstractMatrix) = Scaler(mean(X, dims = 2), max.(std(X, dims = 2), 1e-9))
apply(sc::Scaler, X) = (X .- sc.mu) ./ sc.sd

feat_dam(d::DayScenario)  = d.λ_DAM                                  
feat_ida1(d::DayScenario) = d.λ_IDA1                                 
feat_isp1(d::DayScenario) = vec(d.λ_last)                            

struct GateTree
    leaves::Vector{DayScenario}        
    members::Vector{Vector{Int}}       
    transitions::Vector{Matrix{Float64}}
    assign::Vector{Int}                
    centres::NTuple{3,Any}             
    scalers::NTuple{3,Scaler}        
end

const LEAF_MODE = Symbol(get(ENV, "BESS_LEAF", "medoid"))

function centroid(days::Vector{DayScenario}, idx::Vector{Int}, label::String)
    sub = days[idx]
    if LEAF_MODE === :medoid && !isempty(sub)
        μ = mean(d.λ_DAM for d in sub)
        return sub[argmin([sum(abs2, d.λ_DAM .- μ) for d in sub])]
    end
    avg(f) = mean(getfield(d, f) for d in sub)
    DayScenario(label,
        avg(:λ_DAM), avg(:λ_IDA1), avg(:λ_IDA2), avg(:λ_IDA3),
        avg(:λ_XB_buy), avg(:λ_XB_sell),
        avg(:λ_cap), avg(:λ_last),
        avg(:λ_RT_aFRR_up), avg(:λ_RT_aFRR_dn),
        avg(:λ_RT_mFRR_up), avg(:λ_RT_mFRR_dn),
        avg(:λ_Imb), avg(:λ_UA),
        mean(d.a for d in sub),
        mean(d.accept for d in sub))
end

function build_gate_tree(days::Vector{DayScenario}; K = (4, 3, 2), T::Int, seed::Int = 1)
    K2, K3, K4 = K
    K2 = min(K2, length(days))
    n = length(days)
    S2 = Scaler(reduce(hcat, feat_dam.(days)))
    S3 = Scaler(reduce(hcat, feat_ida1.(days)))
    S4 = Scaler(reduce(hcat, feat_isp1.(days)))
    X2 = apply(S2, reduce(hcat, feat_dam.(days)))
    if K2 >= n
        a2 = collect(1:n); c2 = X2; K2 = n
    else
        a2, c2 = kmeans_cols(X2, K2; seed = seed)
    end
    a3 = zeros(Int, n); c3 = Dict{Int,Matrix{Float64}}()
    for i in 1:K2
        mem = findall(==(i), a2); isempty(mem) && continue
        sub, c = kmeans_cols(apply(S3, reduce(hcat, feat_ida1.(days[mem]))), K3; seed = seed + i)
        a3[mem] = sub; c3[i] = c
    end
    a4 = zeros(Int, n); c4 = Dict{Int,Matrix{Float64}}()
    node3(d) = (a2[d] - 1) * K3 + a3[d]
    for i3 in 1:(K2 * K3)
        mem = findall(d -> node3(d) == i3, 1:n); isempty(mem) && continue
        sub, c = kmeans_cols(apply(S4, reduce(hcat, feat_isp1.(days[mem]))), K4; seed = seed + 100 + i3)
        a4[mem] = sub; c4[i3] = c
    end
    node4(d) = (node3(d) - 1) * K4 + a4[d]
    n2, n3, n4 = K2, K2 * K3, K2 * K3 * K4
    leafof = [node4(d) for d in 1:n]
    members = [findall(==(l), leafof) for l in 1:n4]

    P = Matrix{Float64}[]
    push!(P, ones(1, 1))

    p2 = zeros(1, n2)
    for d in 1:n; p2[1, a2[d]] += 1 / n; end
    any(iszero, p2) && (p2 .= max.(p2, 1e-6); p2 ./= sum(p2))
    push!(P, p2)

    p3 = zeros(n2, n3)
    for d in 1:n; p3[a2[d], node3(d)] += 1; end
    for i in 1:n2
        s = sum(p3[i, :])
        s > 0 ? (p3[i, :] ./= s) : (p3[i, (i-1)*K3+1:i*K3] .= 1 / K3)
    end
    push!(P, p3)

    p4 = zeros(n3, n4)
    for d in 1:n; p4[node3(d), node4(d)] += 1; end
    for i in 1:n3
        s = sum(p4[i, :])
        s > 0 ? (p4[i, :] ./= s) : (p4[i, (i-1)*K4+1:i*K4] .= 1 / K4)
    end
    push!(P, p4)

    for _ in 5:T
        push!(P, Matrix{Float64}(I, n4, n4))     # the day is determined; hold
    end

    leaves = [isempty(members[l]) ?
              centroid(days, collect(1:n), "leaf$l-EMPTY") :
              centroid(days, members[l], "leaf$l") for l in 1:n4]

    @info "Gate tree: $n days -> $n2 / $n3 / $n4 nodes at the IDA1 / ISP / IDA2 gates" *
          "; leaf sizes $(length.(members))"
    return GateTree(leaves, members, P, leafof, (c2, c3, c4), (S2, S3, S4))
end

function leaf_of(tree::GateTree, d::DayScenario; K = (4, 3, 2))
    K2, K3, K4 = K
    c2, c3, c4 = tree.centres
    S2, S3, S4 = tree.scalers
    near(x, C) = argmin([sum(abs2, x .- C[:, j]) for j in 1:size(C, 2)])
    i2 = near(vec(apply(S2, reshape(feat_dam(d),  :, 1))), c2)
    i3 = (i2 - 1) * K3 + near(vec(apply(S3, reshape(feat_ida1(d), :, 1))), c3[i2])
    return (i3 - 1) * K4 + near(vec(apply(S4, reshape(feat_isp1(d), :, 1))), c4[i3])
end

