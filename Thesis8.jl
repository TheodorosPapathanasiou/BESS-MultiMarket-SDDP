#    Layer A — BIDDING stages, no delivery:
#      t=1  DAM gate closure        13:00 D-1   -> qDA   for MTU  1..96
#      t=2  IDA1 gate closure       15:00 D-1   -> qI1   for MTU  1..96
#      t=3  ISP bid deadline        16:45 D-1   -> reserve offers
#      t=4  IDA2 gate closure       22:00 D-1   -> qI2   for MTU  1..96
#
#    Layer B — DELIVERY stages, t = 4+m, m = 1..96:
#      m=41  10:00 D  IDA3 gate closure         -> qI3   for MTU 49..96
#      XBID continuous, closes T-60'            -> at MTU m, trade for MTU m+4
#      RTBM balancing,  closes T-45'            -> at MTU m, offer for MTU m+3
#
#    Award binding (NOT the same as offer submission):
#      ISP1 (16:45 D-1) non-binding — indicative only, no money
#      ISP2 (00:00 D)   binds DP  1..48 = MTU  1..48  (00:00-12:00)
#      ISP3 (12:00 D)   binds DP 49..96 = MTU 49..96  (12:00-24:00)
import Pkg
Pkg.add("HiGHS")
using JuMP, SDDP, HiGHS , CSV, DataFrames, LinearAlgebra, Statistics, Dates, Printf, Random



# -----------------------------------------------------------------------------
# 1. TIME AXIS AND GATE CALENDAR
# -----------------------------------------------------------------------------
const MTU = 96
const Dt  = 0.25

const S_DAM, S_IDA1, S_ISP, S_IDA2 = 1, 2, 3, 4
const N_BID = 4
const T     = N_BID + MTU

mtu_of(t)   = t <= N_BID ? 0 : t - N_BID
stage_of(m) = N_BID + m

const M_IDA3       = 41   # 10:00 D — IDA3 gate closure
const M_IDA3_FIRST = 49   # 12:00 D — IDA3 has a price for MTU 49..96 and none for 1..48
const M_ISP2       = 1    # 00:00 D — ISP2 executed, binds MTU 1..48
const M_ISP3       = 49   # 12:00 D — ISP3 executed, binds MTU 49..96

const LAG_XBID = 4        # T-60'
const LAG_RTBM = 3        # T-45'
@assert LAG_XBID == LAG_RTBM + 1   

# -----------------------------------------------------------------------------
# 2. COMMITMENT GRANULARITY
# -----------------------------------------------------------------------------
const DA_BLK  = 4
const IDA_BLK = 4
const RES_BLK = 4

nblk(w) = cld(MTU, w) #how many blocks are required
blk(m, w) = cld(m, w) #which block does MTU m belong to

const N_DA   = nblk(DA_BLK)
const N_IDA  = nblk(IDA_BLK)
const N_RES  = nblk(RES_BLK)
const N_IDA3 = cld(MTU - M_IDA3_FIRST + 1, IDA_BLK)
blk3(m) = cld(m - M_IDA3_FIRST + 1, IDA_BLK)  

# -----------------------------------------------------------------------------
# 3. BESS PARAMETERS
# -----------------------------------------------------------------------------
const P_CH_MAX  = 5.0
const P_DCH_MAX = 5.0
const ETA_CH    = 0.9
const ETA_DCH   = 0.9
const SOC_MIN   = 1.0
const SOC_MAX   = 10.0
const SOC_INIT  = 5.0
const DEG_COST  = 10.0     

# -----------------------------------------------------------------------------
# 4. DATA LOADING
# -----------------------------------------------------------------------------
const CSV_MISSING = ["-", "", "#N/A", "NA", "N/A"]
const INSTANCE = Symbol(get(ENV, "BESS_INSTANCE", "SS"))
const RUN_ID   = get(ENV, "BESS_RUN_ID", "local")

parsenum(x::Real) = Float64(x)
parsenum(::Missing) = NaN
function parsenum(s::AbstractString)
    t = strip(s)
    (t in CSV_MISSING) && return NaN
    return parse(Float64, replace(t, ',' => '.'))
end

struct DayScenario
    date::String
    λ_DAM::Vector{Float64}
    λ_IDA1::Vector{Float64}
    λ_IDA2::Vector{Float64}
    λ_IDA3::Vector{Float64}          
    λ_XB_buy::Vector{Float64}
    λ_XB_sell::Vector{Float64}
    λ_cap::Matrix{Float64}           #the price actually paid
    λ_last::Matrix{Float64}          #the marginal price
    λ_RT_aFRR_up::Vector{Float64}; λ_RT_aFRR_dn::Vector{Float64}
    λ_RT_mFRR_up::Vector{Float64}; λ_RT_mFRR_dn::Vector{Float64}
    λ_Imb::Vector{Float64}
    λ_UA::Vector{Float64}          
    a::Matrix{Float64}              
    accept::Matrix{Float64}          
end

const PROD = (:FCRu, :FCRd, :aFRRu, :aFRRd, :mFRRu, :mFRRd)

function load_days(path::AbstractString)
    df = CSV.read(path, DataFrame; delim = ';', decimal = ',',
                  missingstring = CSV_MISSING, stringtype = String)
    ("DateTime" in names(df)) || rename!(df, names(df)[1] => :DateTime)
    nraw = nrow(df)
    df = df[.!ismissing.(df.DateTime), :]
    nraw > nrow(df) && @info "Dropped $(nraw - nrow(df)) rows with no DateTime."
    fmt    = dateformat"dd/mm/yyyy HH:MM:SS"
    starts = [strip(first(split(String(s), " - "))) for s in df.DateTime]
    dts    = DateTime[]
    for (j, s) in enumerate(starts)
        try
            push!(dts, DateTime(s, fmt))
        catch
            error("Row $j: cannot parse timestamp $(repr(s)) with format $fmt")
        end
    end
    df.date = Dates.format.(dts, dateformat"yyyy-mm-dd")
    df.mtu  = @. Dates.hour(dts) * 4 + Dates.minute(dts) ÷ 15 + 1
    col(sub, name) = parsenum.(sub[!, name])
    days, dropped = DayScenario[], String[]
    for sub in groupby(sort(df, [:date, :mtu]), :date)
        d = first(sub.date)
        if nrow(sub) != MTU || sub.mtu != collect(1:MTU)
            push!(dropped, d); continue
        end
       
        a = zeros(6, MTU)
        for (i, (acc, prc, fallback)) in enumerate((
                (:Act_FCR_UP,  :Proc_FCR_UP,  :a_FCR_UP),
                (:Act_FCR_DN,  :Proc_FCR_DN,  :a_FCR_DN),
                (:Act_aFRR_UP, :Proc_aFRR_UP, :a_aFRR_UP),
                (:Act_aFRR_DN, :Proc_aFRR_DN, :a_aFRR_DN),
                (:Act_mFRR_UP, :Proc_mFRR_UP, :a_mFRR_UP),
                (:Act_mFRR_DN, :Proc_mFRR_DN, :a_mFRR_DN)))
            recomputed = if hasproperty(sub, acc) && hasproperty(sub, prc)
                A, P = col(sub, acc), col(sub, prc)
                any(isnan, A) || any(isnan, P) || any(iszero, P) ? nothing : A ./ (P .* Dt)
            else
                nothing
            end
            a[i, :] = clamp.(something(recomputed, col(sub, fallback)), 0.0, 1.0)
        end

        vals = Dict(c => col(sub, c) for c in
            (:DAM, :IDA1, :IDA2, :IDA3, :XBID_BUY_VWAP, :XBID_SELL_VWAP,
             :RTBM_aFRR_UP, :RTBM_aFRR_DN, :RTBM_mFRR_UP, :RTBM_mFRR_DN,
             :Imbalances, :Uplift_Account_1, :Uplift_Account_2, :Uplift_Account_3))
        prodcols(prefix) = (Symbol("$(prefix)_FCR_UP"),  Symbol("$(prefix)_FCR_DN"),
                            Symbol("$(prefix)_aFRR_UP"), Symbol("$(prefix)_aFRR_DN"),
                            Symbol("$(prefix)_mFRR_UP"), Symbol("$(prefix)_mFRR_DN"))
        λ_last = reduce(vcat, (col(sub, c)' for c in prodcols("λ_last")))
        λ_wa = copy(λ_last)
        for (i, c) in enumerate(prodcols("λ_wa"))
            hasproperty(sub, c) || continue
            v = col(sub, c)
            any(isnan, v) || (λ_wa[i, :] = v)
        end
        λ_cap = INSTANCE === :PI ? λ_last : λ_wa
        incomplete = any(k -> k === :IDA3 ? any(isnan, vals[k][M_IDA3_FIRST:MTU])
                                          : any(isnan, vals[k]), keys(vals)) ||
                     any(isnan, a) || any(isnan, λ_last)
        if incomplete
            push!(dropped, d); continue
        end

        push!(days, DayScenario(d,
            vals[:DAM], vals[:IDA1], vals[:IDA2], vals[:IDA3],
            vals[:XBID_BUY_VWAP], vals[:XBID_SELL_VWAP],
            λ_cap, λ_last,
            vals[:RTBM_aFRR_UP], vals[:RTBM_aFRR_DN],
            vals[:RTBM_mFRR_UP], vals[:RTBM_mFRR_DN],
            vals[:Imbalances],
            vals[:Uplift_Account_1] .+ vals[:Uplift_Account_2] .+ vals[:Uplift_Account_3],
            a,
            ones(6, MTU),
        ))
    end
    @info "Loaded $(length(days)) complete days; dropped $(length(dropped))." dropped
    return days
end

const ALL_DAYS = load_days("Dataset.csv")

# NON-DELIVERY COST
const UNCNPBE = 3.5
const T_MONTH = parse(Float64, get(ENV, "BESS_T_MONTH", "144"))   
const A_NPBE  = 3.22 / (1 + exp(-0.0026 * (T_MONTH - 1050))) + 0.8
const TOL_BE  = 0.03
const NCAP    = P_DCH_MAX
const NC_BAND = 0.25 * TOL_BE * NCAP        

const IMB_P90 = quantile([v for d in ALL_DAYS for v in d.λ_Imb], 0.9)
const PEN_ND  = parse(Float64, get(ENV, "BESS_PEN_MULT", "1.0")) *
                (IMB_P90 + UNCNPBE * A_NPBE)
@info "Non-delivery cost from RAE 478/2022" UNCNPBE A_NPBE TOL_BE charge_only=UNCNPBE*A_NPBE imbalance_p90=IMB_P90 PEN_ND

# -----------------------------------------------------------------------------
# 4a. mFRR ACTIVATION — offer price
# -----------------------------------------------------------------------------
const MFRR_OFFER_Q = parse(Float64, get(ENV, "BESS_MFRR_Q", "0.5"))  

function apply_mfrr_offer_rule!(days::Vector{DayScenario}; q::Float64 = MFRR_OFFER_Q)
    up = [quantile([d.λ_RT_mFRR_up[m] for d in days], q)       for m in 1:MTU]
    dn = [quantile([d.λ_RT_mFRR_dn[m] for d in days], 1.0 - q) for m in 1:MTU]
    before = (mean(mean(d.a[5, :]) for d in days), mean(mean(d.a[6, :]) for d in days))
    for d in days                      
        d.a[5, :] .= Float64.(d.λ_RT_mFRR_up .>= up .- 1e-9)
        d.a[6, :] .= Float64.(d.λ_RT_mFRR_dn .<= dn .+ 1e-9)
    end
    after = (mean(mean(d.a[5, :]) for d in days), mean(mean(d.a[6, :]) for d in days))
    @info "mFRR activation rule at q=$q" clamped_ratio_up_dn=before offer_rule_up_dn=after
    return (up, dn)
end
const MFRR_OFFER = apply_mfrr_offer_rule!(ALL_DAYS)

# -----------------------------------------------------------------------------
# 4c. TRAIN / TEST AND GATE SCENARIO TREE
# -----------------------------------------------------------------------------
const N_TRAIN     = parse(Int, get(ENV, "BESS_N_TRAIN", "120"))
const TRAIN_START = parse(Int, get(ENV, "BESS_TRAIN_START", "1"))
const SPLIT_MODE = Symbol(get(ENV, "BESS_SPLIT", "interleave"))
const TRAIN, TEST = if SPLIT_MODE === :chrono
    lo, hi = TRAIN_START, min(TRAIN_START + N_TRAIN - 1, length(ALL_DAYS))
    ALL_DAYS[lo:hi], ALL_DAYS[hi+1:end]
else
    te = [d for (i, d) in enumerate(ALL_DAYS) if i % 3 == 0]
    tr = [d for (i, d) in enumerate(ALL_DAYS) if i % 3 != 0]
    tr[1:min(N_TRAIN, length(tr))], te
end
@info "Split" mode=SPLIT_MODE n_train=length(TRAIN) n_test=length(TEST)

include("gate_scenario_tree.jl")

const TREE_K = INSTANCE === :PI ? (length(TRAIN), 1, 1) :
    Tuple(parse.(Int, split(get(ENV, "BESS_TREE_K", "3,2,2"), ",")))
@info """
    ========================================================
      RUNNING THE $(INSTANCE) INSTANCE
        training days : $(length(TRAIN))
        tree K        : $(TREE_K)
        reserve price : $(INSTANCE === :PI ? "lambda_last, always accepted" :
                                             "own offer at quantile OFFER_Q")
    ========================================================
    """
const TREE   = build_gate_tree(TRAIN; K = TREE_K, T = T)
const SCEN   = TREE.leaves
const S      = length(SCEN)

# -----------------------------------------------------------------------------
# 4d. RESERVE OFFER PRICE AND ACCEPTANCE
# -----------------------------------------------------------------------------
const OFFER_Q = parse(Float64, get(ENV, "BESS_OFFER_Q", "0.5"))
const GROSS_MULT = parse(Float64, get(ENV, "BESS_GROSS_MULT", "2.0"))
const OFFER = INSTANCE === :PI ? nothing :
    [quantile([d.λ_last[k, m] for d in TRAIN], OFFER_Q) for k in 1:6, m in 1:MTU]

if INSTANCE !== :PI
    for (l, leaf) in enumerate(TREE.leaves)
        mem = isempty(TREE.members[l]) ? collect(eachindex(TRAIN)) : TREE.members[l]
        leaf.λ_cap .= OFFER
        for k in 1:6, m in 1:MTU
            leaf.accept[k, m] = OFFER[k, m] <= leaf.λ_last[k, m] + 1e-9 ? 1.0 : 0.0
        end
    end
    @info "Pay-as-bid offers at q=$OFFER_Q" mean_offer_EUR_per_MW_h=mean(OFFER) mean_acceptance=mean(mean(l.accept) for l in TREE.leaves)
end
blkavg(v, b, w) = (lo = (b-1)*w + 1; hi = min(b*w, MTU); mean(@view v[lo:hi]))

# -----------------------------------------------------------------------------
# 4b. PENALTY SCALE
# -----------------------------------------------------------------------------
const PMAX = let v = Float64[]
    for d in ALL_DAYS
        append!(v, d.λ_DAM); append!(v, d.λ_IDA1); append!(v, d.λ_IDA2)
        append!(v, d.λ_XB_sell); append!(v, view(d.λ_IDA3, M_IDA3_FIRST:MTU))
        append!(v, vec(d.λ_last)); append!(v, d.λ_RT_aFRR_up)
        append!(v, d.λ_RT_mFRR_up); append!(v, d.λ_Imb)
    end
    quantile(filter(isfinite, v), 0.99)
end
const PEN_SOC  =  20 * PMAX   # SOC bound violation
const PEN_TERM =  50 * PMAX   # terminal SOC shortfall
const PEN_GATE =  20 * PMAX   # gate feasibility drift
@assert PEN_SOC > PEN_ND "SOC violation must cost more than non-delivery"
@info "Penalty scale from PMAX = $(round(PMAX, digits=1)) EUR/MWh" PEN_ND PEN_SOC PEN_TERM

function naive_revenue_bound(d::DayScenario)
    tot = 0.0
    for m in 1:MTU
        e = max(d.λ_DAM[m], d.λ_IDA1[m], d.λ_IDA2[m], d.λ_XB_sell[m],
                m >= M_IDA3_FIRST ? d.λ_IDA3[m] : -Inf, 0.0)
        tot += (P_DCH_MAX * e
              + P_DCH_MAX * (d.λ_cap[1, m] + d.λ_cap[3, m] + d.λ_cap[5, m])
              + P_CH_MAX  * (d.λ_cap[2, m] + d.λ_cap[4, m] + d.λ_cap[6, m])
              + P_DCH_MAX * max(d.λ_RT_aFRR_up[m], d.λ_RT_mFRR_up[m], 0.0)
              + P_CH_MAX  * max(-d.λ_XB_buy[m], 0.0)) * Dt
    end
    return tot
end
const UB = 2 * maximum(naive_revenue_bound, ALL_DAYS)
@info "SDDP upper_bound set to $(round(UB, digits=0)) EUR/day"

# -----------------------------------------------------------------------------
# 5. WITHIN-DAY UNCERTAINTY: utilisation factors
# -----------------------------------------------------------------------------
struct Activation
    a::NTuple{6,Float64}
end

function build_activation_noise(days; K::Int = 3)
    probs = fill(1/K, K)
    qs    = [(2k - 1) / (2K) for k in 1:K]
    Ω = [Activation[] for _ in 1:MTU]
    for m in 1:MTU, (k, q) in enumerate(qs)
        push!(Ω[m], Activation(ntuple(6) do i
            v = quantile([d.a[i, m] for d in days], q)
            v < 1e-4 ? 0.0 : v
        end))
    end
    return Ω, probs
end
const ACT_Ω, ACT_P = build_activation_noise(ALL_DAYS; K = 3)

# -----------------------------------------------------------------------------
# 6. POLICY GRAPH
# -----------------------------------------------------------------------------
function build_transitions()
    P = Array{Float64,2}[ones(1, 1), fill(1/S, 1, S)]
    for _ in 3:T
        push!(P, Matrix{Float64}(I, S, S))
    end
    return P
end

model = SDDP.MarkovianPolicyGraph(;
    transition_matrices = TREE.transitions,
    sense = :Max,
    upper_bound = UB,
    optimizer = JuMP.optimizer_with_attributes(
        () -> HiGHS.Optimizer()),
) do sp, node

    t, i = node
    m  = mtu_of(t)
    
    # ---- STATES -------------------------------------------------------------
    @variable(sp, SOC_MIN <= SOC <= SOC_MAX, SDDP.State, initial_value = SOC_INIT)

    
    @variable(sp, -P_CH_MAX <= qDA[1:N_DA]   <= P_DCH_MAX, SDDP.State, initial_value = 0.0)
    @variable(sp, -P_CH_MAX <= qI1[1:N_IDA]  <= P_DCH_MAX, SDDP.State, initial_value = 0.0)
    @variable(sp, -P_CH_MAX <= qI2[1:N_IDA]  <= P_DCH_MAX, SDDP.State, initial_value = 0.0)
    @variable(sp, -P_CH_MAX <= qI3[1:N_IDA3] <= P_DCH_MAX, SDDP.State, initial_value = 0.0)
    @variable(sp, 0 <= r[1:6, 1:N_RES] <= max(P_CH_MAX, P_DCH_MAX), SDDP.State, initial_value = 0.0)

    @variable(sp, -P_CH_MAX <= xb[1:LAG_XBID] <= P_DCH_MAX, SDDP.State, initial_value = 0.0)
    @variable(sp, 0 <= rtu[1:LAG_RTBM] <= P_DCH_MAX, SDDP.State, initial_value = 0.0)
    @variable(sp, 0 <= rtd[1:LAG_RTBM] <= P_CH_MAX,  SDDP.State, initial_value = 0.0)

    # ---- FREEZE: free at the gate, pinned everywhere else --------------------
    t != S_DAM  && @constraint(sp, [b in 1:N_DA],  qDA[b].out == qDA[b].in)
    t != S_IDA1 && @constraint(sp, [b in 1:N_IDA], qI1[b].out == qI1[b].in)
    t != S_IDA2 && @constraint(sp, [b in 1:N_IDA], qI2[b].out == qI2[b].in)
    m != M_IDA3 && @constraint(sp, [b in 1:N_IDA3], qI3[b].out == qI3[b].in)
    t != S_ISP  && @constraint(sp, [k in 1:6, b in 1:N_RES], r[k,b].out == r[k,b].in)

    pos(m2; ida3::Bool = true) =
        qDA[blk(m2, DA_BLK)].in + qI1[blk(m2, IDA_BLK)].in + qI2[blk(m2, IDA_BLK)].in +
        ((ida3 && m2 >= M_IDA3_FIRST) ? qI3[blk3(m2)].in : zero(AffExpr))
    _abscache = Dict{Tuple{Symbol,Int},VariableRef}()
    function absv(key::Tuple{Symbol,Int}, e)
        get!(_abscache, key) do
            v = @variable(sp, lower_bound = 0.0)
            @constraint(sp, v >= e); @constraint(sp, v >= -e)
            v
        end
    end
    const_P_GROSS = GROSS_MULT * max(P_CH_MAX, P_DCH_MAX)

    r_up(m2) = r[1, blk(m2, RES_BLK)].in + r[3, blk(m2, RES_BLK)].in + r[5, blk(m2, RES_BLK)].in
    r_dn(m2) = r[2, blk(m2, RES_BLK)].in + r[4, blk(m2, RES_BLK)].in + r[6, blk(m2, RES_BLK)].in

    # =========================== BIDDING STAGES ==============================
    if m == 0
        @variable(sp, bs >= 0)        
        @variable(sp, js >= 0)          

        @constraint(sp, SOC.out == SOC.in)
        @constraint(sp, [k in 1:LAG_XBID], xb[k].out == xb[k].in)
        @constraint(sp, [k in 1:LAG_RTBM], rtu[k].out == rtu[k].in)
        @constraint(sp, [k in 1:LAG_RTBM], rtd[k].out == rtd[k].in)

        if t == S_IDA1
            @constraint(sp, [m2 in 1:MTU],
                qDA[blk(m2,DA_BLK)].in + qI1[blk(m2,IDA_BLK)].out <=  P_DCH_MAX + bs)
            @constraint(sp, [m2 in 1:MTU],
                qDA[blk(m2,DA_BLK)].in + qI1[blk(m2,IDA_BLK)].out >= -P_CH_MAX - bs)
            for m2 in 1:MTU
                @constraint(sp, absv((:qDA,  blk(m2,DA_BLK)),  qDA[blk(m2,DA_BLK)].in) +
                                absv((:qI1o, blk(m2,IDA_BLK)), qI1[blk(m2,IDA_BLK)].out) <= const_P_GROSS + bs)
            end
        elseif t == S_ISP
            for m2 in 1:MTU
                b = blk(m2, RES_BLK)
                p = qDA[blk(m2,DA_BLK)].in + qI1[blk(m2,IDA_BLK)].in
                @constraint(sp, r[1,b].out + r[3,b].out + r[5,b].out <= P_DCH_MAX - p + bs)
                @constraint(sp, r[2,b].out + r[4,b].out + r[6,b].out <= P_CH_MAX  + p + bs)
                @constraint(sp,
                    (r[1,b].out + r[3,b].out + r[5,b].out) / P_DCH_MAX +
                    (r[2,b].out + r[4,b].out + r[6,b].out) / P_CH_MAX <= 1 + js)
            end
        elseif t == S_IDA2
            for m2 in 1:MTU
                p = qDA[blk(m2,DA_BLK)].in + qI1[blk(m2,IDA_BLK)].in + qI2[blk(m2,IDA_BLK)].out
                @constraint(sp, p <=  P_DCH_MAX - r_up(m2) + bs)
                @constraint(sp, p >= -P_CH_MAX + r_dn(m2) - bs)
                @constraint(sp, absv((:qDA,  blk(m2,DA_BLK)),  qDA[blk(m2,DA_BLK)].in) + absv((:qI1,  blk(m2,IDA_BLK)), qI1[blk(m2,IDA_BLK)].in) +
                                absv((:qI2o, blk(m2,IDA_BLK)), qI2[blk(m2,IDA_BLK)].out) <= const_P_GROSS + bs)
            end
        end

        @stageobjective(sp, -PEN_GATE * bs - PEN_ND * P_DCH_MAX * js)
        return
    end

    # =========================== DELIVERY STAGES =============================
    sc = SCEN[i]                    
    bD, bI, bR = blk(m, DA_BLK), blk(m, IDA_BLK), blk(m, RES_BLK)

    @constraint(sp, [k in 1:LAG_XBID-1], xb[k].out == xb[k+1].in)
    @constraint(sp, [k in 1:LAG_RTBM-1], rtu[k].out == rtu[k+1].in)
    @constraint(sp, [k in 1:LAG_RTBM-1], rtd[k].out == rtd[k+1].in)

    @variable(sp, gs[1:6] >= 0)

    # --- IDA3 gate (10:00 D), covering MTU 49..96 ---
    if m == M_IDA3
        for m2 in M_IDA3_FIRST:MTU
            p = qDA[blk(m2,DA_BLK)].in + qI1[blk(m2,IDA_BLK)].in +
                qI2[blk(m2,IDA_BLK)].in + qI3[blk3(m2)].out
            @constraint(sp, p <= P_DCH_MAX - r_up(m2) + gs[1])
            @constraint(sp, p >= -P_CH_MAX + r_dn(m2) - gs[2])
            @constraint(sp, absv((:qDA,  blk(m2,DA_BLK)),  qDA[blk(m2,DA_BLK)].in) + absv((:qI1,  blk(m2,IDA_BLK)), qI1[blk(m2,IDA_BLK)].in) +
                            absv((:qI2,  blk(m2,IDA_BLK)), qI2[blk(m2,IDA_BLK)].in) + absv((:qI3o, blk3(m2)),        qI3[blk3(m2)].out)
                            <= const_P_GROSS + gs[1])
        end
    end

    # --- XBID gate (T-60') for MTU m+LAG_XBID ---
    if m + LAG_XBID <= MTU
        mx = m + LAG_XBID
        @constraint(sp, pos(mx) + xb[LAG_XBID].out <=  P_DCH_MAX - r_up(mx) + gs[3])
        @constraint(sp, pos(mx) + xb[LAG_XBID].out >= -P_CH_MAX + r_dn(mx) - gs[4])
        @constraint(sp, absv((:qDA,  blk(mx,DA_BLK)),  qDA[blk(mx,DA_BLK)].in) + absv((:qI1,  blk(mx,IDA_BLK)), qI1[blk(mx,IDA_BLK)].in) +
                        absv((:qI2,  blk(mx,IDA_BLK)), qI2[blk(mx,IDA_BLK)].in) +
                        (mx >= M_IDA3_FIRST ? absv((:qI3,  blk3(mx)),        qI3[blk3(mx)].in) : zero(AffExpr)) +
                        absv((:xbo,  0),               xb[LAG_XBID].out) <= const_P_GROSS + gs[3])
    else
        @constraint(sp, xb[LAG_XBID].out == 0)
    end

    # --- RTBM gate (T-45') for MTU m+LAG_RTBM ---
    if m + LAG_RTBM <= MTU
        mr = m + LAG_RTBM
        pr = pos(mr) + xb[LAG_RTBM+1].in
        @constraint(sp, rtu[LAG_RTBM].out <= P_DCH_MAX - pr - r_up(mr) + gs[5])
        @constraint(sp, rtd[LAG_RTBM].out <= P_CH_MAX  + pr - r_dn(mr) + gs[6])
        @constraint(sp,
            (rtu[LAG_RTBM].out + r_up(mr)) / P_DCH_MAX +
            (rtd[LAG_RTBM].out + r_dn(mr)) / P_CH_MAX <= 1 + gs[5] / P_DCH_MAX)
    else
        @constraint(sp, rtu[LAG_RTBM].out == 0)
        @constraint(sp, rtd[LAG_RTBM].out == 0)
    end

    @expression(sp, net, pos(m) + xb[1].in)

    @variable(sp, 0 <= p_ch  <= P_CH_MAX)
    @variable(sp, 0 <= p_dch <= P_DCH_MAX)
    @variable(sp, nd_up  >= 0)     
    @variable(sp, nd_dn  >= 0)     
    @variable(sp, s_under >= 0)   
    @variable(sp, s_over  >= 0)
    @variable(sp, s_term  >= 0)

    @constraint(sp, p_ch / P_CH_MAX + p_dch / P_DCH_MAX <= 1)

    @constraint(sp, SOC.out ==
        SOC.in + (ETA_CH * p_ch - p_dch / ETA_DCH) * Dt + s_under - s_over)

    if m == MTU
        @constraint(sp, SOC.out + s_term >= SOC_INIT)
    else
        @constraint(sp, s_term == 0)
    end

    balance = @constraint(sp,
        net + rtu[1].in - rtd[1].in == p_dch - p_ch + nd_up - nd_dn)

    SDDP.parameterize(sp, ACT_Ω[m], ACT_P) do ω
        a = ω.a
        w = @view sc.accept[:, m]       

        JuMP.set_normalized_coefficient(balance, r[1,bR].in,  a[1] * w[1])
        JuMP.set_normalized_coefficient(balance, r[2,bR].in, -a[2] * w[2])
        JuMP.set_normalized_coefficient(balance, r[3,bR].in,  a[3] * w[3])
        JuMP.set_normalized_coefficient(balance, r[4,bR].in, -a[4] * w[4])
        JuMP.set_normalized_coefficient(balance, r[5,bR].in,  a[5] * w[5])
        JuMP.set_normalized_coefficient(balance, r[6,bR].in, -a[6] * w[6])
        JuMP.set_normalized_coefficient(balance, rtu[1].in,  a[5])
        JuMP.set_normalized_coefficient(balance, rtd[1].in, -a[6])

        ida3_rev = m >= M_IDA3_FIRST ? qI3[blk3(m)].in * sc.λ_IDA3[m] * Dt : 0.0

        @stageobjective(sp,
              qDA[bD].in * sc.λ_DAM[m]  * Dt
            + qI1[bI].in * sc.λ_IDA1[m] * Dt
            + qI2[bI].in * sc.λ_IDA2[m] * Dt
            + ida3_rev
            + xb[1].in * 0.5 * (sc.λ_XB_buy[m] + sc.λ_XB_sell[m]) * Dt
            + (r[1,bR].in * w[1] * sc.λ_cap[1, m]  + r[2,bR].in * w[2] * sc.λ_cap[2, m])  * Dt
            + (r[3,bR].in * w[3] * sc.λ_cap[3, m] + r[4,bR].in * w[4] * sc.λ_cap[4, m]) * Dt
            + (r[5,bR].in * w[5] * sc.λ_cap[5, m] + r[6,bR].in * w[6] * sc.λ_cap[6, m]) * Dt
            + r[3,bR].in * w[3] * a[3] * sc.λ_RT_aFRR_up[m] * Dt
            - r[4,bR].in * w[4] * a[4] * sc.λ_RT_aFRR_dn[m] * Dt
            + (r[5,bR].in * w[5] + rtu[1].in) * a[5] * sc.λ_RT_mFRR_up[m] * Dt
            - (r[6,bR].in * w[6] + rtd[1].in) * a[6] * sc.λ_RT_mFRR_dn[m] * Dt
            + (r[1,bR].in * w[1] * a[1] - r[2,bR].in * w[2] * a[2]) * sc.λ_Imb[m] * Dt
            - (p_ch + p_dch) * DEG_COST * Dt
            - (p_ch + p_dch) * sc.λ_UA[m] * Dt
            - PEN_ND   * (nd_up + nd_dn) * Dt
            - PEN_GATE * sum(gs) * Dt
            - PEN_SOC  * (s_under + s_over) - PEN_TERM * s_term
        )
    end
end

# -----------------------------------------------------------------------------
# 7. TRAIN, SIMULATE
# -----------------------------------------------------------------------------
Random.seed!(1234)
SDDP.numerical_stability_report(model)

SDDP.train(model;
    iteration_limit = parse(Int, get(ENV, "BESS_ITERS", "6000")),
    time_limit      = parse(Float64, get(ENV, "BESS_TIME_LIMIT", "86400.0")),        
    stopping_rules  = [SDDP.BoundStalling(300, 1e-2)],
    log_frequency   = 100,
    parallel_scheme = SDDP.Serial(),
    cut_deletion_minimum = parse(Int, get(ENV, "BESS_CUT_DEL", "50")))

sims = SDDP.simulate(model, 200,
    [:SOC, :qDA, :qI1, :qI2, :qI3, :r, :xb, :rtu, :rtd,
     :p_ch, :p_dch, :nd_up, :nd_dn, :s_under, :s_over, :s_term, :gs, :bs, :js];
    skip_undefined_variables = true)

isdelivery(nd) = first(nd[:node_index]) > N_BID
deliv(s) = filter(isdelivery, s)

let bad = 0
    for s in sims
        d = deliv(s)
        ref = first(d)[:qDA]
        for nd in d, b in 1:N_DA
            isapprox(nd[:qDA][b].in, ref[b].in; atol = 1e-6) || (bad += 1)
        end
    end
    println(bad == 0 ?
        "OK: DA position constant through the delivery day." :
        "FAIL: DA position moves within a replication ($bad mismatches) — a freeze constraint is missing.")
end

let ref = first(deliv(sims[1]))[:qDA]
    spread = maximum(abs(first(deliv(s))[:qDA][b].in - ref[b].in)
                     for s in sims for b in 1:N_DA)
    @printf("max DA spread across replications: %.4f MW\n", spread)
end

let nodes = [nd for s in sims for nd in deliv(s)]
    nd_mwh = sum(n[:nd_up] + n[:nd_dn] for n in nodes) * Dt
    oblig  = sum(abs(n[:p_dch] - n[:p_ch] + n[:nd_up] - n[:nd_dn]) for n in nodes) * Dt
    @printf("violation rate: %.2f%%   (non-delivered %.1f MWh over %.1f MWh)\n",
            100 * nd_mwh / max(oblig, eps()), nd_mwh, oblig)
end

let
    pen_nd(n)   = PEN_ND   * (n[:nd_up] + n[:nd_dn]) * Dt
    pen_soc(n)  = PEN_SOC  * (n[:s_under] + n[:s_over])
    pen_term(n) = PEN_TERM * n[:s_term]
    pen_gs(n)   = PEN_GATE * sum(n[:gs]) * Dt
    pen(n)      = pen_nd(n) + pen_soc(n) + pen_term(n) + pen_gs(n)

    raw  = [sum(nd[:stage_objective] for nd in s)          for s in sims]
    pens = [sum(pen(nd) for nd in deliv(s))                for s in sims]
    econ = raw .+ pens

    @printf("\neconomic profit (penalties excluded)\n")
    @printf("   mean %10.1f EUR/day = %.1f EUR/MW/day\n", mean(econ), mean(econ)/P_DCH_MAX)
    @printf("   range [%.1f, %.1f]\n", minimum(econ), maximum(econ))
    @printf("objective as optimised (penalties included): mean %.1f EUR/day\n", mean(raw))
    @printf("\npenalties charged, mean EUR/day -- diagnostics, not costs\n")
    for (name, f) in (("non-delivery", pen_nd), ("SOC violation", pen_soc),
                      ("terminal SOC", pen_term), ("gate drift", pen_gs))
        v = [sum(f(nd) for nd in deliv(s)) for s in sims]
        @printf("   %-14s %10.1f   (max %10.1f)\n", name, mean(v), maximum(v))
    end
    let nodes = [n for s in sims for n in deliv(s)],
        over  = count(n -> (n[:nd_up] + n[:nd_dn]) * Dt > NC_BAND, nodes)
        @printf("\nArt. 22.4 breaches: %d of %d ISPs (%.3f%%) exceed the %.3f MWh tolerance band\n",
                over, length(nodes), 100over/length(nodes), NC_BAND)
    end

    let nodes = [n for s in sims for n in deliv(s)],
        overlap = [min(n[:p_ch], n[:p_dch]) for n in nodes],
        both = count(>(1e-4), overlap)
        @printf("\nsimultaneous charge+discharge: %d of %d MTUs (%.3f%%)\n",
                both, length(nodes), 100both/length(nodes))
        if both > 0
            @printf("   overlap MW: max %.4f  mean over breaching MTUs %.4f  total %.2f MWh/day\n",
                    maximum(overlap), sum(overlap)/both, sum(overlap)*Dt/length(sims))
            let hot = [n for n in nodes if min(n[:p_ch], n[:p_dch]) > 1e-4]
                @printf("   breaching MTUs: SOC.in %.2f MWh (max %.1f), rtd %.2f MW, rtu %.2f MW, xb %.2f MW\n",
                        mean(n[:SOC].in for n in hot), SOC_MAX,
                        mean(n[:rtd][1].in for n in hot), mean(n[:rtu][1].in for n in hot),
                        mean(n[:xb][1].in for n in hot))
            end
            println(maximum(overlap) < 0.05 ?
                "   -> negligible, consistent with degenerate ties; report as such" :
                "   -> material; the Virtual Entity rule needs enforcing, not relying on cost")
        end
    end
    let labels = ("IDA3 up", "IDA3 dn", "XBID up", "XBID dn", "RTBM up", "RTBM dn"),
        nodes = [nd for s in sims for nd in deliv(s)]
        let jsv = [nd[:js] for s in sims for nd in s if !isdelivery(nd)]
            isempty(jsv) || @printf("   joint-reserve slack: max %.3f of converter (fires in %.2f%% of bidding nodes)\n",
                                    maximum(jsv), 100*count(>(1e-5), jsv)/length(jsv))
        end
        let bsv = [nd[:bs] for s in sims for nd in s if !isdelivery(nd)]
            isempty(bsv) || @printf("   max BIDDING-gate slack: %.2e MW  (fires in %.2f%% of bidding nodes)\n",
                                    maximum(bsv), 100*count(>(1e-5), bsv)/length(bsv))
        end
        @printf("   max gate slack in any single MTU: %.2e MW\n",
                maximum(sum(nd[:gs]) for nd in nodes))
        for k in 1:6
            v = [nd[:gs][k] for nd in nodes]
            maximum(v) > 1e-5 && @printf("      %-8s max %7.3f MW   fires in %5.2f%% of MTUs\n",
                labels[k], maximum(v), 100 * count(>(1e-5), v) / length(v))
        end
    end
    b  = SDDP.calculate_bound(model)
    se = std(raw) / sqrt(length(raw))
    @printf("\nSDDP bound %.1f vs simulated %.1f +/- %.1f  -> gap %.1f%%\n",
            b, mean(raw), 1.96se, 100 * (b - mean(raw)) / max(abs(b), eps()))
end

try
    rev = Dict{Symbol,Float64}(k => 0.0 for k in
        (:DAM, :IDA, :XBID, :FCR_cap, :aFRR_cap, :mFRR_cap, :BalEnergy, :Imbalance, :Degradation, :Uplift))
    for s in sims, n in deliv(s)
        t, i = n[:node_index]; m = t - N_BID; sc = SCEN[i]
        a = n[:noise_term].a
        w = view(sc.accept, :, m)
        bD, bI, bR = blk(m, DA_BLK), blk(m, IDA_BLK), blk(m, RES_BLK)
        r_, q1, q2 = n[:r], n[:qI1][bI].in, n[:qI2][bI].in
        q3 = m >= M_IDA3_FIRST ? n[:qI3][blk3(m)].in * sc.λ_IDA3[m] : 0.0

        rev[:DAM]       += n[:qDA][bD].in * sc.λ_DAM[m] * Dt
        rev[:IDA]       += (q1 * sc.λ_IDA1[m] + q2 * sc.λ_IDA2[m] + q3) * Dt
        rev[:XBID]      += n[:xb][1].in * 0.5*(sc.λ_XB_buy[m] + sc.λ_XB_sell[m]) * Dt
        rev[:FCR_cap]   += (r_[1,bR].in*w[1]*sc.λ_cap[1, m]  + r_[2,bR].in*w[2]*sc.λ_cap[2, m])  * Dt
        rev[:aFRR_cap]  += (r_[3,bR].in*w[3]*sc.λ_cap[3, m] + r_[4,bR].in*w[4]*sc.λ_cap[4, m]) * Dt
        rev[:mFRR_cap]  += (r_[5,bR].in*w[5]*sc.λ_cap[5, m] + r_[6,bR].in*w[6]*sc.λ_cap[6, m]) * Dt
        rev[:BalEnergy] += ( r_[3,bR].in*w[3]*a[3]*sc.λ_RT_aFRR_up[m]
                           - r_[4,bR].in*w[4]*a[4]*sc.λ_RT_aFRR_dn[m]
                           + (r_[5,bR].in*w[5] + n[:rtu][1].in) * a[5] * sc.λ_RT_mFRR_up[m]
                           - (r_[6,bR].in*w[6] + n[:rtd][1].in) * a[6] * sc.λ_RT_mFRR_dn[m]) * Dt
        rev[:Imbalance] += (r_[1,bR].in*w[1]*a[1] - r_[2,bR].in*w[2]*a[2]) * sc.λ_Imb[m] * Dt
        rev[:Degradation] -= (n[:p_ch] + n[:p_dch]) * DEG_COST * Dt
        rev[:Uplift]      -= (n[:p_ch] + n[:p_dch]) * sc.λ_UA[m] * Dt
    end
    n = length(sims)
    @printf("\nrevenue attribution, EUR/day averaged over %d replications\n", n)
    tot = sum(values(rev)) / n
    for k in (:DAM, :IDA, :XBID, :FCR_cap, :aFRR_cap, :mFRR_cap, :BalEnergy, :Imbalance, :Degradation, :Uplift)
        @printf("   %-12s %10.1f  (%5.1f%%)\n", k, rev[k]/n, 100 * rev[k]/n / tot)
    end
    @printf("   %-12s %10.1f\n", "TOTAL", tot)

    let nodes = [n for s in sims for n in deliv(s)], gross = 0.0, netv = 0.0
        for n in nodes
            t, i = n[:node_index]; m = t - N_BID
            legs = (n[:qDA][blk(m, DA_BLK)].in,
                    n[:qI1][blk(m, IDA_BLK)].in, n[:qI2][blk(m, IDA_BLK)].in,
                    m >= M_IDA3_FIRST ? n[:qI3][blk3(m)].in : 0.0,
                    n[:xb][1].in)
            gross += sum(abs, legs); netv += abs(sum(legs))
        end
        @printf("\ngross traded %.1f MW-MTU/day vs net position %.1f  -> offsetting ratio %.2fx\n",
                gross/length(sims), netv/length(sims), gross/max(netv, eps()))
    end
catch err
    @warn "revenue attribution failed; the results above are unaffected" err
end

try
    mkpath("results")
    nodes  = [nd for sim in sims for nd in deliv(sim)]
    ndmwh  = sum(n[:nd_up] + n[:nd_dn] for n in nodes) * Dt
    oblig  = sum(abs(n[:p_dch] - n[:p_ch] + n[:nd_up] - n[:nd_dn]) for n in nodes) * Dt
    penrow(n) = PEN_ND*(n[:nd_up]+n[:nd_dn])*Dt + PEN_SOC*(n[:s_under]+n[:s_over]) +
                PEN_TERM*n[:s_term] + PEN_GATE*sum(n[:gs])*Dt
    raw   = [sum(nd[:stage_objective] for nd in sim) for sim in sims]
    econ  = raw .+ [sum(penrow(nd) for nd in deliv(sim)) for sim in sims]
    bound = SDDP.calculate_bound(model)
    open("results/run_$(RUN_ID).csv", "w") do io
        println(io, "run_id,instance,offer_q,mfrr_q,n_train,tree_k,iters,",
                    "econ_profit_eur_mw_day,objective,bound,gap_pct,",
                    "violation_pct,overlap_mwh_day,max_gate_slack")
        println(io, join((RUN_ID, INSTANCE, OFFER_Q, MFRR_OFFER_Q, N_TRAIN,
            "\"$(join(TREE_K,'-'))\"", parse(Int, get(ENV,"BESS_ITERS","6000")),
            round(mean(econ)/P_DCH_MAX, digits=2), round(mean(raw), digits=2),
            round(bound, digits=2), round(100*(bound-mean(raw))/max(abs(bound),eps()), digits=2),
            round(100*ndmwh/max(oblig,eps()), digits=3),
            round(sum(min(n[:p_ch], n[:p_dch]) for n in nodes)*Dt/length(sims), digits=3),
            round(maximum(sum(nd[:gs]) for nd in nodes), digits=6)), ","))
    end
    @info "wrote results/run_$(RUN_ID).csv"
catch err
    @warn "result row failed; the printed output above is unaffected" err
end

include("thesis_outputs.jl")
