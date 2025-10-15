module MacroEnergyTimeReduction

using DataFrames
using StatsBase
using Clustering
using Distances
using CSV
using Random
using LinearAlgebra
using Flux
using Zygote

src_dir = @__DIR__

include(joinpath(src_dir,"algorithms", "cluster.jl"))
include(joinpath(src_dir,"algorithms", "cluster_kmeans.jl"))
include(joinpath(src_dir,"algorithms", "cluster_kmedoids.jl"))
include(joinpath(src_dir,"algorithms", "cluster_autoencoder_sequential.jl"))
include(joinpath(src_dir,"algorithms", "cluster_autoencoder_simultaneous.jl"))

export cluster, 
       cluster_kmeans, 
       cluster_kmedoids, 
       cluster_autoencoder_sequential, 
       cluster_autoencoder_simultaneous
end