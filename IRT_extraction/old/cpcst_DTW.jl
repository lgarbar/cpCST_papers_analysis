### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ d0f6e95e-52e0-11ed-11a1-8f1cf737d23b
begin 
	using Pkg
	Pkg.add("CSV")
	Pkg.add("PlutoUI")
	Pkg.add("DataFrames")
	Pkg.add("CairoMakie")
	Pkg.add("DynamicAxisWarping")
	Pkg.add("Distances")
	Pkg.add("Glob")
	Pkg.add("Smoothers")
	using CSV
	using PlutoUI
	using DataFrames
	using Statistics
	using CairoMakie
	# using Makie
	using Glob
	using DynamicAxisWarping
	using Distances
	using Smoothers
end


# ╔═╡ d83fe3f8-f6a7-4be3-8e78-8605b5bcd89c
Pkg.add("Query")

# ╔═╡ 35a9f0f0-96a2-4f1e-8d0c-f6e072a67fa3
# using DynamicAxisWarping
begin
	Pkg.add("SentinelArrays")
	using SentinelArrays
end

# ╔═╡ 335eeb6b-6de8-4438-a2eb-3423b4627382
using Query

# ╔═╡ 7c4b42f3-042f-457f-a3b0-74d1a26317e1
CairoMakie.activate!()

# ╔═╡ 5d56f8d8-da1e-4fbe-9b5b-1cada36308ea
function ffill!(vec)
	for k in range(1,length(vec))
		if isnan(vec[k]) && k>1
			vec[k] = vec[k-1]
		end
	end
end


# ╔═╡ ce15d81a-3000-4a8f-ab90-5884c13d3195
function load_cpCST_csv(filepath)
    """Could make string vs. DF dispatch"""
    fr = CSV.read(filepath, DataFrame)
	ffill!(fr.user_pos)
	ffill!(fr.stim_pos)
	fr[!,:user_pos] = fr.user_pos * -1
	fr[!,:time_secs] = fr.expected_time .- fr.expected_time[1]
	return fr
end


# ╔═╡ 2faceb9b-4d66-40e3-a374-4ed032c03c99
infil = "/Users/danielgarcia-barnett/Desktop/Coding/cpCST_data_analysis/data/raw_cpt_data/sub-M10942248_ses-MOBI1_task-CPT_run-1_events.csv"

# ╔═╡ e58deb3f-9bed-4eb3-a73b-6ab3e43f05f9
df = load_cpCST_csv(infil)

# ╔═╡ 62f57fc3-0583-4f55-afd2-4236e8a0a94f
tdat = DataFrame(a=1:3, b=4:6)

# ╔═╡ 84bbb3c7-0ba5-4fa1-81e2-7572c40a550b
begin
	a = df.user_pos
	b = df.stim_pos
	dist=SqEuclidean(1e-12)
	radius=500
	cost, i1, i2 = fastdtw(a,b, dist, radius)
	# cost, i1, i2 = dtw(a,b, dist; radius=500, transportcost = 1)
	# cost, i1, i2 = dtw(a, b, [SqEuclidean], radius=500)
end

# ╔═╡ 9bbad991-dd28-4ef8-b9fb-c9f61791357f
function get_dtw_vals(DF, thresh=1e-12, radius=500)
	a = DF.stim_pos
	b = DF.user_pos
	dist=SqEuclidean(thresh)
	cost, i1, i2 = fastdtw(a,b, dist, radius)
	dat = DataFrame(user=i1, stim=i2)
	return(dat)
end

# ╔═╡ 4d5357b1-30c7-4faf-9836-418747bd8c07
function get_row(dat, tgt_val)
	tmp = @from i in dat begin
            @where i.user == tgt_val
            @select {i.user, i.stim}
            @collect DataFrame
	end
	tmp[!, :user_ts] = tmp.user .* (1/60)
	tmp[!, :stim_ts] = tmp.stim .* (1/60)
	return(tmp)
end

# ╔═╡ 0a774c17-bbda-4296-b920-f5878055a484
function get_point_rt(dat)
    idx = dat.user[1]
	user_ts = dat.user_ts[1]
	mean_stim = mean(dat.stim_ts)
	inst_rt = mean_stim - user_ts
	return(inst_rt)
end

# ╔═╡ bc04e647-c779-466e-9168-75a952c9a162
begin
	A = [x for x in range(1,length(df.stim_pos))]
end


# ╔═╡ cc95c5cd-6e59-4c3b-8be3-0b03e46ccd0c
function compute_irt(DF)
	dat = get_dtw_vals(DF)
	n_vals = length(DF.user_pos)
	user_irt = zeros(n_vals)
	for u in range(1,n_vals)
		rdat = get_row(dat, u)
		irt = get_point_rt(rdat)
		print(irt)
		user_irt[u] = irt
	end
	return(user_irt)
end

# ╔═╡ 4eee0ea4-5f68-4524-a9f7-93fe982d96c2
function compute_irt!(DF)
	dat = get_dtw_vals(DF)
	n_vals = length(DF.user_pos)
	user_irt = zeros(n_vals)
	for u in range(1,n_vals)
		rdat = get_row(dat, u)
		irt = get_point_rt(rdat)
		print(irt)
		user_irt[u] = irt
	end
	df[!,:irt] = user_irt
end

# ╔═╡ 7034f321-ed2f-4be0-9536-a0639ca12ba9
flist = Glob.glob("*CPT_run*.csv", "/Users/danielgarcia-barnett/Desktop/Coding/cpCST_data_analysis/data/raw_cpt_data")

# ╔═╡ cdf10878-e163-461a-b828-bd5961b82cf3
"s_"*split(flist[1], '/')[end]

# ╔═╡ bb9901be-17df-4c9b-b448-475ec8be8e37
fil = flist[1]

# ╔═╡ ab420f9e-b733-4314-a350-5d6732203936
Df = load_cpCST_csv(fil)

# ╔═╡ e08538a1-b09d-49af-a7fc-ad53304e5dc0
irt = compute_irt(Df)

# ╔═╡ 160d9a53-3e6f-4b77-b442-f058e11085a1
plot(irt)

# ╔═╡ 285b752f-1cf8-43d6-93f9-f1ad667ce3fc
length(irt)

# ╔═╡ ec143b1f-7271-4cef-bfdb-26e670239555
pwd()

# ╔═╡ e8982984-1c89-4f15-9e0d-a50bf2c7667a
function filt_irt(irt_vec, w=100)
	# filt_irt = sma(irt_vec,w,true)
	x = irt_vec
	filt_irt = sma(x,w,true)
	return(filt_irt) 
end

# ╔═╡ 2d689c0c-f9b6-424c-b782-a9649c012492
for fil in flist
	w = 100
	filname = split(fil, '/')[end]
	outfile = "proc_"*filname
	print(filname)
	Df_temp = load_cpCST_csv(fil)
	IRT = compute_irt(Df_temp)
	Df_temp[!, :irt] = IRT
	# Df[!:, :sm_irt] = sma(IRT,w,true)
	
	CSV.write(outfile, Df_temp)
end

# ╔═╡ 5e934ed1-cbf0-4633-a18a-09e5f4b9103d
flist

# ╔═╡ 8d0a617c-5177-4c2c-8250-5b431573e2fd
function pointwise_lag(
    x::Vector{Float64},
    y::Vector{Float64},
    sampling_frequency::Int64,
    max_lag_seconds::Float64=5.0
)
    max_lag = Int(ceil(sampling_frequency * max_lag_seconds))
    m, n = length(x), length(y)
    lag = zeros(Int, m)

    for i in 1:m
        min_distance = Inf
        min_lag = 0
        search_start = max(1, i - max_lag)
        search_end = min(n, i + max_lag)

        for j in search_start:search_end
            # Slicing vectors to compare
            x_slice = x[max(1, i-1):min(m, i+1)]
            y_slice = y[max(1, j-1):min(n, j+1)]

            # Ensure we only use the distance from dtw result
            distance, _ = dtw(x_slice, y_slice)  # Extract distance, ignore additional info (e.g., path)

            if distance < min_distance
                min_distance = distance
                min_lag = j - i
            end
        end

        lag[i] = min_lag
    end

    time_lag = lag ./ sampling_frequency
    return time_lag
end

# ╔═╡ 431b761f-6aab-42ef-aa24-2a20e07c430b
# begin
# 	# Convert df.user_pos to a ChainedVector
# 	user_pos_chained = ChainedVector([df.user_pos])
	
# 	time_lag = pointwise_lag(df.stim_pos, user_pos_chained, 30)
# end

# ╔═╡ 8c7ce6a6-c4a0-4412-861c-bc38cc0c555f
begin
	stim_pos_vector = collect(df.stim_pos)  # Convert to Vector{Float64}
	user_pos_vector = collect(df.user_pos)  # Convert to Vector{Float64}

	time_lag = pointwise_lag(stim_pos_vector, user_pos_vector, 30)	
end

# ╔═╡ 6ba978d6-601c-4aa2-ad36-510742fa2e68
println(time_lag)

# ╔═╡ 2aceeca4-c978-496b-8ff1-6a0d970dcfcd
length(irt)

# ╔═╡ 44f3f7ea-fdae-4408-8fb2-c7120758ac51
hist(irt)

# ╔═╡ 68b1cc8e-3c9d-4174-b756-c3c4127f8b83
mean(irt)

# ╔═╡ cb9716e7-c941-4f9d-9c3f-65ad48dc6bfd
CSV.write("/Users/danielgarcia-barnett/Desktop/Coding/cpCST_data_analysis/data/dtw_data/testing2.csv", df)

# ╔═╡ f12bec44-4119-4739-b7cd-9b3ad0444d58
firt = filt_irt(irt)

# ╔═╡ 8479c6f7-d420-4ead-97df-1d74491abfd3
length(df[!,:flip_time])

# ╔═╡ 8dbd1177-06cd-43a9-b4b1-99fafe5bda4c
length(irt) - length(df[!,:flip_time])

# ╔═╡ 267009a4-a265-40f3-866f-a4b3869dc268
df[!, :irt] = irt

# ╔═╡ e7f42b05-9774-48dc-8463-674a3f131315
begin
	w = 150; sw = string(w)
	x = df.irt
	t = df.time_secs
	vals = sma(x,w,true)
	lines(t, vals, color=:blue, linewidth=1, linestyle=:solid) 
	#loess(t,x;q=w)(t)
	
end

# ╔═╡ 79ec4eb7-414c-4e4e-8661-1841bae0996d
begin
	sm_irt = filt_irt(df.irt, 100)
end

# ╔═╡ f2598621-868a-42b5-b1b0-6fb0e08e1b1c
hist(sm_irt)

# ╔═╡ a51fce8c-2981-49c0-9e88-c91441a0996f
hist(df.irt)

# ╔═╡ 2b0bdda1-bc3c-4fb2-bdd3-b5d856b75e38
md"""Python DTW Function:
```python
def dtw_series(user, stim, timeseries):
    distance, path = fastdtw.fastdtw(user, stim, radius=500) #, dist=euclidean)
    p1 = np.array([p[0] for p in path])
    p2 = np.array([p[1] for p in path])

    pdat = pd.DataFrame(columns=["p1",'p2'])
    pdat.p1 = p1
    pdat.p2 = p2

    shift = []
    for k in range(len(user)):
        t = pdat[pdat.p2==k].p1.values
        t -= k
        shift.append(t.mean())

    shift = np.array(shift) * np.diff(timeseries).mean()
    shift_stats = [distance, shift.mean(), shift.std()]
    return(shift, shift_stats, pdat)
```
"""

# ╔═╡ 6b24851a-f616-4ba2-8251-576abc4c1d96
begin 
	lines(1:10, (1:10).^2; color=:black, linewidth=2, linestyle=:dash,
    figure=(; figure_padding=5, resolution=(1200, 400), font="sans",
        backgroundcolor=:grey90, fontsize=16),
    axis=(; xlabel="x", ylabel="x²", title="title",
        xgridstyle=:dash, ygridstyle=:dash))
    current_figure()
end

# ╔═╡ 15f2f21f-7edb-48d9-a9df-f0915c736052
# begin
# 	# f2 = dtwplot(df,zz,lc=:green, lw=1, transportcost=2)
# 	a =  AbstractArray([d for d in df.stim_pos])
# 	b = AbstractArray([d for d in df.user_pos])
# 	dist=SqEuclidean
# 	radius=15
# 	# cost, i1, i2 = dtw(a,b, dist=SqEuclidean; transportcost = 2)
#     cost = fastdtw(a, b, dist, radius)
# 	# f3 = matchplot(a,b,ds=3,separation=1)


# end


# ╔═╡ Cell order:
# ╠═d0f6e95e-52e0-11ed-11a1-8f1cf737d23b
# ╠═7c4b42f3-042f-457f-a3b0-74d1a26317e1
# ╠═5d56f8d8-da1e-4fbe-9b5b-1cada36308ea
# ╠═ce15d81a-3000-4a8f-ab90-5884c13d3195
# ╠═2faceb9b-4d66-40e3-a374-4ed032c03c99
# ╠═e58deb3f-9bed-4eb3-a73b-6ab3e43f05f9
# ╠═62f57fc3-0583-4f55-afd2-4236e8a0a94f
# ╠═d83fe3f8-f6a7-4be3-8e78-8605b5bcd89c
# ╠═335eeb6b-6de8-4438-a2eb-3423b4627382
# ╠═84bbb3c7-0ba5-4fa1-81e2-7572c40a550b
# ╠═9bbad991-dd28-4ef8-b9fb-c9f61791357f
# ╠═4d5357b1-30c7-4faf-9836-418747bd8c07
# ╠═0a774c17-bbda-4296-b920-f5878055a484
# ╠═bc04e647-c779-466e-9168-75a952c9a162
# ╠═cc95c5cd-6e59-4c3b-8be3-0b03e46ccd0c
# ╠═4eee0ea4-5f68-4524-a9f7-93fe982d96c2
# ╠═7034f321-ed2f-4be0-9536-a0639ca12ba9
# ╠═cdf10878-e163-461a-b828-bd5961b82cf3
# ╠═bb9901be-17df-4c9b-b448-475ec8be8e37
# ╠═ab420f9e-b733-4314-a350-5d6732203936
# ╠═e08538a1-b09d-49af-a7fc-ad53304e5dc0
# ╠═160d9a53-3e6f-4b77-b442-f058e11085a1
# ╠═285b752f-1cf8-43d6-93f9-f1ad667ce3fc
# ╠═ec143b1f-7271-4cef-bfdb-26e670239555
# ╠═e8982984-1c89-4f15-9e0d-a50bf2c7667a
# ╠═2d689c0c-f9b6-424c-b782-a9649c012492
# ╠═5e934ed1-cbf0-4633-a18a-09e5f4b9103d
# ╠═35a9f0f0-96a2-4f1e-8d0c-f6e072a67fa3
# ╠═8d0a617c-5177-4c2c-8250-5b431573e2fd
# ╠═431b761f-6aab-42ef-aa24-2a20e07c430b
# ╠═8c7ce6a6-c4a0-4412-861c-bc38cc0c555f
# ╠═6ba978d6-601c-4aa2-ad36-510742fa2e68
# ╠═2aceeca4-c978-496b-8ff1-6a0d970dcfcd
# ╠═44f3f7ea-fdae-4408-8fb2-c7120758ac51
# ╠═68b1cc8e-3c9d-4174-b756-c3c4127f8b83
# ╠═cb9716e7-c941-4f9d-9c3f-65ad48dc6bfd
# ╠═f12bec44-4119-4739-b7cd-9b3ad0444d58
# ╠═8479c6f7-d420-4ead-97df-1d74491abfd3
# ╠═8dbd1177-06cd-43a9-b4b1-99fafe5bda4c
# ╠═267009a4-a265-40f3-866f-a4b3869dc268
# ╠═e7f42b05-9774-48dc-8463-674a3f131315
# ╠═79ec4eb7-414c-4e4e-8661-1841bae0996d
# ╠═f2598621-868a-42b5-b1b0-6fb0e08e1b1c
# ╠═a51fce8c-2981-49c0-9e88-c91441a0996f
# ╟─2b0bdda1-bc3c-4fb2-bdd3-b5d856b75e38
# ╠═6b24851a-f616-4ba2-8251-576abc4c1d96
# ╠═15f2f21f-7edb-48d9-a9df-f0915c736052
