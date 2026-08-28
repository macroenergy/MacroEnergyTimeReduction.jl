using Test
using DataFrames
using LinearAlgebra
using MacroEnergyTimeReduction

const CLUSTERING_INPUT = DataFrame([
    0.0  0.1  10.0  10.1  20.0  20.1
    0.0 -0.1  10.0   9.9  20.0  20.1
], :auto)

function assert_clustering_result(result, nclusters, nperiods)
    R, assignments, counts, representatives, distances, clustering_time = result

    @test length(assignments) == nperiods
    @test length(counts) == nclusters
    @test sum(counts) == nperiods
    @test length(representatives) == nclusters
    @test all(x -> 1 <= x <= nclusters, assignments)
    @test all(x -> 1 <= x <= nperiods, representatives)
    @test size(distances) == (nperiods, nperiods)
    @test distances ≈ distances'
    @test all(iszero, diag(distances))
    @test R.assignments == assignments
    @test R.counts == counts
    @test clustering_time >= 0
end

@testset "clustering algorithms" begin
    nclusters = 3
    nperiods = ncol(CLUSTERING_INPUT)

    @testset "k-means" begin
        assert_clustering_result(
            cluster_kmeans(CLUSTERING_INPUT, nclusters, 2), nclusters, nperiods,
        )
    end

    @testset "k-medoids" begin
        # Regression test for Clustering.jl's current kmedoids keyword API.
        assert_clustering_result(
            cluster_kmedoids(CLUSTERING_INPUT, nclusters, 2), nclusters, nperiods,
        )

        # A zero-restart request still returns the initial clustering result.
        assert_clustering_result(
            cluster_kmedoids(CLUSTERING_INPUT, nclusters, 0), nclusters, nperiods,
        )
    end
end
