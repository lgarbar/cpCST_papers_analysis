### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ c09085f4-ab76-4b05-b88c-cc6fa0986caf
begin
	# Import necessary packages
	using Pkg
	Pkg.add("ArgParse")
end

# ╔═╡ 66b12300-d698-11ef-3c53-3550e6a70748
begin
	using CSV
	using DataFrames
	using CairoMakie
	using DynamicAxisWarping
	using Distances
	using Glob
	using Smoothers
	using Query
	using SentinelArrays
	using Statistics
end

# ╔═╡ 99a4dd93-0adf-4836-a47b-4b11de0c6342
# Activate CairoMakie
CairoMakie.activate!()

# ╔═╡ a8fb423d-4bd4-4954-a7ad-647a1eb87841
function ffill!(vec)
    last_value = missing
    for i in 1:length(vec)
        if ismissing(vec[i])
            vec[i] = last_value
        else
            last_value = vec[i]
        end
    end
end

# ╔═╡ d325e2cf-d745-4007-b702-31d801de8307
function load_cpCST_csv(filepath)
    fr = CSV.read(filepath, DataFrame)
    ffill!(fr.user_pos)
    ffill!(fr.stim_pos)
    fr[!, :user_pos] = fr.user_pos .* -1
    fr[!, :time_secs] = fr.flip_time .- fr.flip_time[1]
    return fr
end

# ╔═╡ 3ac77082-ed74-44d0-89be-69d437493297
function get_dtw_vals(DF, thresh=1e-12, radius=120)
    a = skipmissing(DF.stim_pos) |> collect
    b = skipmissing(DF.user_pos) |> collect
    dist = SqEuclidean(thresh)
    cost, i1, i2 = fastdtw(a, b, dist, radius)
    dat = DataFrame(user=i1, stim=i2)
    return dat
end

# ╔═╡ efd75f06-bd91-476f-89db-f216f1332c88
# Function to get a specific row based on target value
function get_row(dat, tgt_val)
	tmp = @from i in dat begin
		@where i.user == tgt_val
		@select {i.user, i.stim}
		@collect DataFrame
	end
	tmp[!, :user_ts] = tmp.user .* (1/60)
	tmp[!, :stim_ts] = tmp.stim .* (1/60)
	return tmp
end

# ╔═╡ d420f5e0-ffc9-4e27-876d-6d9820605780
# Function to compute pointwise reaction time
function get_point_rt(dat)
	idx = dat.user[1]
	user_ts = dat.user_ts[1]
	mean_stim = mean(dat.stim_ts)
	inst_rt = mean_stim - user_ts
	return inst_rt
end

# ╔═╡ 85159ded-52a9-494d-ac85-661d8bb74746
# Function to compute inter-response time and update DataFrame
function compute_irt!(DF)
	dat = get_dtw_vals(DF)
	n_vals = length(DF.user_pos)
	user_irt = zeros(n_vals)
	for u in 1:n_vals
		rdat = get_row(dat, u)
		irt = get_point_rt(rdat)
		user_irt[u] = irt
	end
	DF[!, :irt] = user_irt
end

# ╔═╡ cc61bde5-9c0e-4c4d-8ab6-59401ad5b32c
function process_files(source_folder, destination_folder; participants=[])
    csv_files = glob("*.csv", source_folder)
	if length(participants) > 0
    	csv_files = [file for file in csv_files if any(occursin(participant, file) for participant in participants)]
	end
    for file in csv_files
        println("Processing file: $(basename(file))")
        df = load_cpCST_csv(file)
        compute_irt!(df)
        dest_file = joinpath(destination_folder, basename(file))
        CSV.write(dest_file, df)
    end
end

# ╔═╡ 46263c52-4482-47b5-99d2-126f52df03bf
begin
	# Example usage
	using ArgParse
	
	# Create a parser object
	parser = ArgParseSettings()
	
	@add_arg_table parser begin
		"source_folder"
		help = "Path to the source folder containing CSV files"
		
		"destination_folder"
		help = "Path to the destination folder to save processed CSV files"
	end
	
	# Parse command line arguments
	parsed_args = parse_args(parser)
	
	destination_folder = "/Users/danielgarcia-barnett/Desktop/Coding/cpCST_data_analysis/IRT_extraction/test_data/jan_28_2024/test_1/IRT_data/cleaned_data_new_dtw/"
	source_folder = "/Users/danielgarcia-barnett/Desktop/Coding/cpCST_data_analysis/IRT_extraction/test_data/jan_28_2024/test_1/cpCST_data/cleaned_data/"
	process_files(source_folder, destination_folder)
end

# ╔═╡ b3e654e7-8f41-4899-86f6-a914c2aed9c1
crash_ptps = ["M10902833", "M10904765", "M10905290", "M10905520", "M10905521",
 "M10908665", "M10909540", "M10910591", "M10913260", "M10916065",
 "M10917604", "M10918927", "M10919134", "M10928541", "M10929065",
 "M10929927", "M10931165", "M10936289", "M10938265", "M10942248",
 "M10942846", "M10943691", "M10945411", "M10945575", "M10948593",
 "M10948896", "M10950715", "M10952753", "M10957570", "M10957731",
 "M10958986", "M10959209", "M10959572", "M10960573", "M10960879",
 "M10961312", "M10964357", "M10966987", "M10968423", "M10970400",
 "M10970636", "M10978174", "M10981315", "M10982585", "M10984244",
 "M10984509", "M10988214", "M10988888", "M10992053", "M10997698"]

# ╔═╡ Cell order:
# ╠═c09085f4-ab76-4b05-b88c-cc6fa0986caf
# ╠═66b12300-d698-11ef-3c53-3550e6a70748
# ╠═99a4dd93-0adf-4836-a47b-4b11de0c6342
# ╠═a8fb423d-4bd4-4954-a7ad-647a1eb87841
# ╠═d325e2cf-d745-4007-b702-31d801de8307
# ╠═3ac77082-ed74-44d0-89be-69d437493297
# ╠═efd75f06-bd91-476f-89db-f216f1332c88
# ╠═d420f5e0-ffc9-4e27-876d-6d9820605780
# ╠═85159ded-52a9-494d-ac85-661d8bb74746
# ╠═cc61bde5-9c0e-4c4d-8ab6-59401ad5b32c
# ╠═b3e654e7-8f41-4899-86f6-a914c2aed9c1
# ╠═46263c52-4482-47b5-99d2-126f52df03bf
