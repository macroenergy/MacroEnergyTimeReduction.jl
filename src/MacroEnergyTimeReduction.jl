module MacroEnergyTimeReduction

using DataFrames
using StatsBase
using Clustering
using Distances

using Random
using LinearAlgebra
using Flux
using Flux: glorot_uniform, leakyrelu, Dense, Conv, flatten, params, update!, gradient
import Zygote
import Flux.Optimise: update!
Zygote.@nograd fill!
Zygote.@nograd Flux.create_bias

using Distances: Euclidean, pairwise

src_dir = @__DIR__

include(joinpath(src_dir,"algorithms", "cluster.jl"))
include(joinpath(src_dir,"algorithms", "cluster_kmeans.jl"))
include(joinpath(src_dir,"algorithms", "cluster_kmedoids.jl"))
include(joinpath(src_dir,"algorithms", "cluster_sequential.jl"))
include(joinpath(src_dir,"algorithms", "cluster_simultaneous.jl"))

export cluster, 
       cluster_kmeans, 
       cluster_kmedoids, 
       cluster_sequential, 
       cluster_simultaneous


end