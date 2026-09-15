using Printf, Statistics

function allocation_profile(sims; path = "results/allocation_$(RUN_ID).csv")
    mkpath(dirname(path))
    nodes_by_mtu = [NamedTuple[] for _ in 1:MTU]
    for s in sims, nd in deliv(s)
        m = first(nd[:node_index]) - N_BID
        bD, bI, bR = blk(m, DA_BLK), blk(m, IDA_BLK), blk(m, RES_BLK)
        push!(nodes_by_mtu[m], (
            dam  = nd[:qDA][bD].in,
            ida  = nd[:qI1][bI].in + nd[:qI2][bI].in +
                   (m >= M_IDA3_FIRST ? nd[:qI3][blk3(m)].in : 0.0),
            xbid = nd[:xb][1].in,
            fcru = nd[:r][1,bR].in, fcrd = nd[:r][2,bR].in,
            afru = nd[:r][3,bR].in, afrd = nd[:r][4,bR].in,
            mfru = nd[:r][5,bR].in, mfrd = nd[:r][6,bR].in,
            rtu  = nd[:rtu][1].in,  rtd  = nd[:rtd][1].in,
            soc  = nd[:SOC].out,
        ))
    end
    avg(m, f) = mean(getfield(x, f) for x in nodes_by_mtu[m])
    open(path, "w") do io
        println(io, "mtu,hour,dam_mw,ida_mw,xbid_mw,net_energy_mw,",
                    "fcr_up_mw,fcr_dn_mw,afrr_up_mw,afrr_dn_mw,mfrr_up_mw,mfrr_dn_mw,",
                    "rtbm_up_mw,rtbm_dn_mw,reserve_up_total_mw,reserve_dn_total_mw,soc_mwh")
        for m in 1:MTU
            isempty(nodes_by_mtu[m]) && continue
            d, i, x = avg(m,:dam), avg(m,:ida), avg(m,:xbid)
            up = avg(m,:fcru) + avg(m,:afru) + avg(m,:mfru)
            dn = avg(m,:fcrd) + avg(m,:afrd) + avg(m,:mfrd)
            println(io, join((m, round((m-1)/4, digits=2),
                round.((d, i, x, d+i+x,
                        avg(m,:fcru), avg(m,:fcrd), avg(m,:afru), avg(m,:afrd),
                        avg(m,:mfru), avg(m,:mfrd), avg(m,:rtu), avg(m,:rtd),
                        up, dn, avg(m,:soc)), digits=4)...), ","))
        end
    end

    println("\n" * "="^62)
    println("  OPTIMAL CAPACITY ALLOCATION  (day average, $(INSTANCE) instance)")
    println("="^62)
    tot(f) = mean(avg(m, f) for m in 1:MTU if !isempty(nodes_by_mtu[m]))
    @printf("  energy position   net %+6.2f MW   (DAM %+.2f  IDA %+.2f  XBID %+.2f)\n",
            tot(:dam)+tot(:ida)+tot(:xbid), tot(:dam), tot(:ida), tot(:xbid))
    println("  balancing capacity committed, MW and % of the 5 MW rating:")
    for (lbl, f) in (("FCR up",:fcru), ("FCR down",:fcrd), ("aFRR up",:afru),
                     ("aFRR down",:afrd), ("mFRR up",:mfru), ("mFRR down",:mfrd))
        v = tot(f)
        @printf("     %-10s %6.3f MW  (%5.1f%%)%s\n", lbl, v, 100v/P_DCH_MAX,
                v < 0.05 ? "   <- effectively unallocated" : "")
    end
    @printf("  mean SOC %.2f MWh of %.1f\n", tot(:soc), SOC_MAX)
    println("  full 96-MTU profile written to $path")
    return path
end

function out_of_sample(test_days::Vector{DayScenario})
    isempty(test_days) && (@warn "TEST is empty"; return nothing)
    if INSTANCE === :PI
        println("\n  out-of-sample skipped: not defined for the PI instance")
        return nothing
    end

    K2, K3, K4 = TREE_K
    vars = [:SOC, :qDA, :qI1, :qI2, :qI3, :r, :xb, :rtu, :rtd,
            :p_ch, :p_dch, :nd_up, :nd_dn, :s_under, :s_over, :s_term, :gs]

    function scenario_for(day)
        l  = leaf_of(TREE, day; K = TREE_K)
        i3 = cld(l, K4); i2 = cld(i3, K3)
        scen = Tuple{Tuple{Int,Int},Any}[((1,1),nothing), ((2,i2),nothing),
                                         ((3,i3),nothing), ((4,l),nothing)]
        for m in 1:MTU
            push!(scen, ((N_BID + m, l), Activation(ntuple(k -> day.a[k, m], 6))))
        end
        return scen
    end

    profits, viols, accs = Float64[], Float64[], Float64[]
    charges, caps = Float64[], Float64[]
    failed = 0
    for day in test_days
        sim = try
            SDDP.simulate(model, 1, vars;
                          sampling_scheme = SDDP.Historical([scenario_for(day)]),
                          skip_undefined_variables = true)[1]
        catch err
            failed += 1; continue
        end

        rev = 0.0; nd = 0.0; oblig = 0.0; acc_n = 0; acc_d = 0
        charge = 0.0; cap_rev = 0.0
        for node in filter(isdelivery, sim)
            m = first(node[:node_index]) - N_BID
            bD, bI, bR = blk(m, DA_BLK), blk(m, IDA_BLK), blk(m, RES_BLK)
            a = node[:noise_term].a
            duty = abs(node[:p_dch] - node[:p_ch] + node[:nd_up] - node[:nd_dn])
            frac = duty < 1e-9 ? 1.0 :
                   clamp(1.0 - (node[:nd_up] + node[:nd_dn]) / duty, 0.0, 1.0)
            if (node[:nd_up] + node[:nd_dn]) * Dt > NC_BAND
                charge += UNCNPBE * A_NPBE * (node[:nd_up] + node[:nd_dn]) * Dt
            end

            rev += node[:qDA][bD].in * day.λ_DAM[m]  * Dt
            rev += node[:qI1][bI].in * day.λ_IDA1[m] * Dt
            rev += node[:qI2][bI].in * day.λ_IDA2[m] * Dt
            m >= M_IDA3_FIRST && (rev += node[:qI3][blk3(m)].in * day.λ_IDA3[m] * Dt)
            rev += node[:xb][1].in * 0.5*(day.λ_XB_buy[m] + day.λ_XB_sell[m]) * Dt

            for k in 1:6
                q = node[:r][k, bR].in
                q < 1e-9 && continue
                price = INSTANCE === :PI ? day.λ_last[k, m] : OFFER[k, m]
                ok = INSTANCE === :PI || price <= day.λ_last[k, m] + 1e-9
                acc_d += 1; acc_n += ok
                ok || continue
                rev += q*price*Dt; cap_rev += q*price*Dt     
                if k in (3, 4)                           
                    rev += frac * (k == 3 ? 1 : -1) * q * a[k] *
                           (k == 3 ? day.λ_RT_aFRR_up[m] : day.λ_RT_aFRR_dn[m]) * Dt
                elseif k in (5, 6)                          # mFRR energy
                    rev += frac * (k == 5 ? 1 : -1) * q * a[k] *
                           (k == 5 ? day.λ_RT_mFRR_up[m] : day.λ_RT_mFRR_dn[m]) * Dt
                else                                        # FCR at imbalance price
                    rev += frac * (k == 1 ? 1 : -1) * q * a[k] * day.λ_Imb[m] * Dt
                end
            end
            rev += frac * (node[:rtu][1].in * day.λ_RT_mFRR_up[m] -
                           node[:rtd][1].in * day.λ_RT_mFRR_dn[m]) * Dt
            rev -= (node[:p_ch] + node[:p_dch]) * (DEG_COST + day.λ_UA[m]) * Dt

            nd    += (node[:nd_up] + node[:nd_dn]) * Dt
            oblig += abs(node[:p_dch] - node[:p_ch] + node[:nd_up] - node[:nd_dn]) * Dt
        end
        push!(profits, rev - charge); push!(charges, charge); push!(caps, cap_rev)
        push!(viols, 100 * nd / max(oblig, eps()))
        push!(accs, acc_d == 0 ? 100.0 : 100 * acc_n / acc_d)
    end

    isempty(profits) && (@warn "every out-of-sample replay failed"; return nothing)
    ins = mean(econ_profit_in_sample)
    println("\n" * "="^62)
    println("  OUT-OF-SAMPLE VALIDATION  ($(length(profits)) unseen days, $(INSTANCE))")
    println("="^62)
    failed > 0 && @printf("  NOTE %d of %d days could not be replayed\n", failed, length(test_days))
    @printf("  realised profit   %8.1f EUR/MW/day  (in-sample %.1f)\n",
            mean(profits)/P_DCH_MAX, ins/P_DCH_MAX)
    @printf("  spread            [%.1f, %.1f]\n",
            minimum(profits)/P_DCH_MAX, maximum(profits)/P_DCH_MAX)
    @printf("  of which capacity %8.1f EUR/MW/day\n", mean(caps)/P_DCH_MAX)
    @printf("  Art. 22.4 charges %8.1f EUR/MW/day  (already deducted)\n", mean(charges)/P_DCH_MAX)
    @printf("  violation rate    %6.2f%%   (eq. 31)\n", mean(viols))
    @printf("  offers accepted   %6.1f%%   (eq. 21-24)\n", mean(accs))
    @printf("  in-sample optimism %5.1f%%\n", 100*(ins - mean(profits))/max(abs(ins), eps()))
    println("  Energy revenue credited on DELIVERED energy only; capacity is")
    println("  credited in full, since the award precedes delivery.")
    return (profit = mean(profits), violation = mean(viols), acceptance = mean(accs))
end

const econ_profit_in_sample = let
    penrow(n) = PEN_ND*(n[:nd_up]+n[:nd_dn])*Dt + PEN_SOC*(n[:s_under]+n[:s_over]) +
                PEN_TERM*n[:s_term] + PEN_GATE*sum(n[:gs])*Dt
    [sum(nd[:stage_objective] for nd in s) + sum(penrow(nd) for nd in deliv(s)) for s in sims]
end

allocation_profile(sims)
out_of_sample(TEST)
