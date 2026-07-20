import JuMP, SDDP, Gurobi, CSV, DataFrames, Plots
using JuMP, SDDP, Gurobi, CSV, DataFrames, Plots

market_data = CSV.read("DATASET_2_MORNING.csv", DataFrame)

const GRB_ENV = Gurobi.Env()
const T = 96
const Dt = 0.25
const Scenarios = 7
const P = 0.1

struct MarketRealization
    λ_DA::Float64
    λ_IDM_buy::Float64
    λ_IDM_sell::Float64
    λ_FCR_up_wa::Float64
    λ_FCR_dn_wa::Float64
    λ_aFRR_up_wa::Float64
    λ_aFRR_dn_wa::Float64
    λ_mFRR_up::Float64
    λ_mFRR_dn::Float64

    a_FCR_up::Float64
    a_FCR_dn::Float64
    a_aFRR_up::Float64
    a_aFRR_dn::Float64
    a_mFRR_up::Float64
    a_mFRR_dn::Float64
end

stage_realizations = [MarketRealization[] for t in 1:T]
for t in 1:T
    for s in 1:Scenarios 
        idx = (s - 1) * T + t
        push!(stage_realizations[t], MarketRealization(
            market_data.HEnEx_11_MCP[idx],
            market_data.HEnEx_80_VWAP_Buy[idx],  
            market_data.HEnEx_81_VWAP_Sell[idx],
            market_data.ADMIE_145_FCR_Price_Up[idx],     
            market_data.ADMIE_145_FCR_Price_Down[idx],
            market_data.ADMIE_186_aFRR_Price_Up[idx],     
            market_data.ADMIE_186_aFRR_Price_Down[idx],   
            market_data.ADMIE_223_mFRR_Price_Up[idx],     
            market_data.ADMIE_223_mFRR_Price_Down[idx],
            0.0194, 
            0.0199,
            0.0500,  
            0.0500,  
            0.1000,  
            0.1000   
        ))
    end
end

probability = fill(1.0 / Scenarios, Scenarios)

p_charge_max = 5.0
p_discharge_max = 5.0
n_charge = 0.9
n_discharge = 0.9
SOC_min = 1.0
SOC_max = 10.0
SOC_initial = 5.0

model = SDDP.LinearPolicyGraph(;
    stages = T,
    sense = :Max,
    upper_bound = 10000000,
    optimizer = JuMP.optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV), 
        "OutputFlag" => 0
    ),
) do sp, t

    @variable(sp, SOC_min <= SOC <= SOC_max, SDDP.State, initial_value = SOC_initial)
    
    @variable(sp, 0 <= p_DA_sell <= p_discharge_max)
    @variable(sp, 0 <= p_DA_buy <= p_charge_max)
    @variable(sp, 0 <= p_IDM_sell <= p_discharge_max)
    @variable(sp, 0 <= p_IDM_buy <= p_charge_max)
    @variable(sp, 0 <= p_FCR_up <= p_discharge_max)
    @variable(sp, 0 <= p_FCR_dn <= p_charge_max)
    @variable(sp, 0 <= p_aFRR_up <= p_discharge_max)
    @variable(sp, 0 <= p_aFRR_dn <= p_charge_max)
    @variable(sp, 0 <= p_mFRR_up <= p_discharge_max)
    @variable(sp, 0 <= p_mFRR_dn <= p_charge_max)

    slack_terminal = @variable(sp, lower_bound = 0)
    if t == T
        @constraint(sp, SOC.out + slack_terminal >= SOC_initial)
    else
        @constraint(sp, slack_terminal == 0) 
    end

    @variable(sp, p_charge >= 0)
    @variable(sp, p_discharge >= 0)
    @variable(sp, slack_soc_under >= 0)
    @variable(sp, slack_soc_over >= 0)
    
    @constraint(sp, p_charge <= p_charge_max)
    @constraint(sp, p_discharge <= p_discharge_max)

    @constraint(sp, SOC.out == SOC.in + (n_charge * p_charge - (p_discharge / n_discharge)) * Dt + slack_soc_under - slack_soc_over)

    @constraint(sp, p_FCR_up + p_aFRR_up + p_mFRR_up <= (p_DA_buy + p_IDM_buy) + (p_discharge_max - (p_DA_sell + p_IDM_sell)))
    @constraint(sp, p_FCR_dn + p_aFRR_dn + p_mFRR_dn <= (p_DA_sell + p_IDM_sell) + (p_charge_max - (p_DA_buy + p_IDM_buy)))

    @constraint(sp, power_balance, 
        p_DA_sell + p_IDM_sell - p_DA_buy - p_IDM_buy + 
        1.0 * p_FCR_up - 1.0 * p_FCR_dn +
        1.0 * p_aFRR_up - 1.0 * p_aFRR_dn +
        1.0 * p_mFRR_up - 1.0 * p_mFRR_dn == p_discharge - p_charge
    )

    SDDP.parameterize(sp, stage_realizations[t], probability) do omega
        JuMP.set_normalized_coefficient(power_balance, p_FCR_up, omega.a_FCR_up)
        JuMP.set_normalized_coefficient(power_balance, p_FCR_dn, -omega.a_FCR_dn)
        JuMP.set_normalized_coefficient(power_balance, p_aFRR_up, omega.a_aFRR_up)
        JuMP.set_normalized_coefficient(power_balance, p_aFRR_dn, -omega.a_aFRR_dn)
        JuMP.set_normalized_coefficient(power_balance, p_mFRR_up, omega.a_mFRR_up)
        JuMP.set_normalized_coefficient(power_balance, p_mFRR_dn, -omega.a_mFRR_dn)

        @stageobjective(sp, 
            (p_DA_sell - p_DA_buy) * omega.λ_DA * Dt + 
            (p_IDM_sell * omega.λ_IDM_sell - p_IDM_buy * omega.λ_IDM_buy) * Dt +
            (p_FCR_up * omega.λ_FCR_up_wa + p_FCR_dn * omega.λ_FCR_dn_wa) * Dt +
            (p_aFRR_up * omega.λ_aFRR_up_wa + p_aFRR_dn * omega.λ_aFRR_dn_wa) * Dt +
            (p_mFRR_up * omega.λ_mFRR_up + p_mFRR_dn * omega.λ_mFRR_dn) * Dt - 
            (p_charge + p_discharge) * P * Dt - 1e6 * slack_terminal - 1e5 * (slack_soc_under + slack_soc_over)
        )
    end
end        

SDDP.train(model; iteration_limit = 100)

simulations = SDDP.simulate(
    model, 
    1, 
    [:SOC, :p_DA_sell, :p_DA_buy, :p_IDM_sell, :p_IDM_buy, :p_FCR_up, :p_FCR_dn, :p_aFRR_up, :p_aFRR_dn, :p_mFRR_up, :p_mFRR_dn]
)

hours = [t * 0.25 for t in 1:T]

soc_profile       = [stage[:SOC].in for stage in simulations[1]]
p_DA_sell_profile = [stage[:p_DA_sell] for stage in simulations[1]]
p_DA_buy_profile  = [stage[:p_DA_buy] for stage in simulations[1]]
p_IDM_sell_profile = [stage[:p_IDM_sell] for stage in simulations[1]]
p_IDM_buy_profile  = [stage[:p_IDM_buy] for stage in simulations[1]]
p_FCR_up_profile  = [stage[:p_FCR_up] for stage in simulations[1]]
p_FCR_dn_profile  = [stage[:p_FCR_dn] for stage in simulations[1]]
p_aFRR_up_profile = [stage[:p_aFRR_up] for stage in simulations[1]]
p_aFRR_dn_profile = [stage[:p_aFRR_dn] for stage in simulations[1]]
p_mFRR_up_profile = [stage[:p_mFRR_up] for stage in simulations[1]]
p_mFRR_dn_profile = [stage[:p_mFRR_dn] for stage in simulations[1]]

plt1 = plot(hours, soc_profile, 
    label="State of Energy (SOE)", 
    seriestype=:path,
    linewidth=2.5, 
    color=:blue,
    xlabel="Hours", 
    ylabel="MWh",
    title="SOE",
    titlefontsize=11, guidefontsize=10, tickfontsize=9,
    xlims=(0, 24), ylims=(0, 11),
    xticks=0:2:24, grid=:dot, size=(800, 400)
)
hline!([SOC_min], color=:red, linestyle=:dash, alpha=0.6, label="SOC Min (1.0 MWh)")
hline!([SOC_max], color=:green, linestyle=:dash, alpha=0.6, label="SOC Max (10.0 MWh)")
display(plt1) 

plt2 = plot(hours, p_DA_buy_profile, 
    label="DAM Buy Offer (p_DA_buy)", 
    seriestype=:steppost,
    linewidth=2, color=:red,
    xlabel="Hours", 
    ylabel="MW",
    title="DAM Offers",
    titlefontsize=11, guidefontsize=10, tickfontsize=9,
    xlims=(0, 25), ylims=(0, 11),
    xticks=0:2:24, grid=:dot, size=(800, 400)
)
plot!(hours, p_DA_sell_profile, label="DAM Sell Offer (p_DA_sell)", linewidth=1.5, color=:orange, linestyle=:dashdot)
display(plt2) 

plt3 = plot(hours, p_IDM_buy_profile, 
    label="IDM Buy Offer", seriestype=:steppost, linewidth=2, color=:blue,
    xlabel="Hours", ylabel="MW", title="Intraday Market (IDM) Offers",
    titlefontsize=11, guidefontsize=10, tickfontsize=9,
    xlims=(0, 25), ylims=(0, 11), xticks=0:2:24, grid=:dot, size=(800, 400)
)
plot!(hours, p_IDM_sell_profile, label="IDM Sell Offer", linewidth=1.5, color=:lightblue, linestyle=:dashdot)
display(plt3) 

# Plot 4: FCR Offers
plt4 = plot(hours, p_FCR_up_profile, 
    label="FCR Up", seriestype=:steppost, linewidth=2, color=:red,
    xlabel="Hours", ylabel="MW", title="FCR Offers",
    titlefontsize=11, guidefontsize=10, tickfontsize=9,
    xlims=(0, 25), ylims=(0, 11), xticks=0:2:24, grid=:dot, size=(800, 400)
)
plot!(hours, p_FCR_dn_profile, label="FCR Downward", linewidth=1.5, color=:purple, linestyle=:dash)
display(plt4) 

# Plot 5: aFRR Offers
plt5 = plot(hours, p_aFRR_up_profile, 
    label="aFRR Up", seriestype=:steppost, linewidth=2, color=:red,
    xlabel="Hours", ylabel="MW", title="aFRR Offers",
    titlefontsize=11, guidefontsize=10, tickfontsize=9,
    xlims=(0, 25), ylims=(0, 11), xticks=0:2:24, grid=:dot, size=(800, 400)
)
plot!(hours, p_aFRR_dn_profile, label="aFRR Downward", linewidth=1.5, color=:purple, linestyle=:dash)
display(plt5) 

# Plot 6: mFRR Offers
plt6 = plot(hours, p_mFRR_up_profile, 
    label="mFRR Up", seriestype=:steppost,linewidth=2, color=:red,
    xlabel="Hours", ylabel="MW", title="mFRR Offers",
    titlefontsize=11, guidefontsize=10, tickfontsize=9,
    xlims=(0, 25), ylims=(0, 11), xticks=0:2:24, grid=:dot, size=(800, 400)
)
plot!(hours, p_mFRR_dn_profile, label="mFRR Downward", linewidth=1.5, color=:purple, linestyle=:dash)
display(plt6)