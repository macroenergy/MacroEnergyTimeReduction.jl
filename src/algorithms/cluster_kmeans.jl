@doc raw"""
    cluster_kmeans(ClusteringInputDF, NClusters, nIters)

Get representative periods using cluster centers from kmeans
"""
function cluster_kmeans(ClusteringInputDF::DataFrame, NClusters::Int, nIters::Int; v::Bool=false)
    X = Matrix(ClusteringInputDF)
    DistMatrix = pairwise(Euclidean(), X, dims=2)

    rng = MersenneTwister(42)

    clustering_time = @elapsed begin
        R = kmeans(X, NClusters; rng=rng, init=:kmpp)

        best = R
        best_cost = R.totalcost
        no_improve = 0
        patience = 50

        for i in 1:nIters
            rng_i = MersenneTwister(42 + i)
            R_i = kmeans(X, NClusters; rng=rng_i, init=:kmpp)
        
            if R_i.totalcost < best_cost - 1e-6
                best = R_i
                best_cost = R_i.totalcost
                no_improve = 0
            else
                no_improve += 1
            end
        
            if v && (i % max(1, nIters ÷ 10) == 0)
                println("Iter $i : cost=$(round(R_i.totalcost, digits=3))  best=$(round(best_cost, digits=3))")
            end
        
            if v && no_improve ≥ patience
                println("Stopping early at iteration $i (no improvement for $patience restarts), best=$(round(best_cost, digits=3))")
                break
            end
        end
        
        R = best

        # Ensure non-empty clusters
        A = copy(R.assignments)
        W = copy(R.counts)
        Centers = copy(R.centers)

        empty_clusters = findall(==(0), W)

        for c in empty_clusters
            donor_candidates = findall(w -> w > 1, W)
            isempty(donor_candidates) && error("No cluster with count > 1 to donate from.")
            donor = donor_candidates[argmax(W[donor_candidates])]

            members = findall(a -> a == donor, A)
            length(members) <= 1 && continue

            cent_d = Centers[:, donor]
            dists = [euclidean(cent_d, X[:, j]) for j in members]
            j_move = members[argmax(dists)]

            A[j_move] = c
            W[donor] -= 1
            W[c]     += 1

            donor_members = findall(a -> a == donor, A)
            Centers[:, donor] .= mean(X[:, donor_members]; dims=2)[:]

            c_members = findall(a -> a == c, A)
            Centers[:, c] .= mean(X[:, c_members]; dims=2)[:]
        end

        @assert all(W .>= 1) "Some clusters are still empty after repair."

        M = Int[]
        chosen = Set{Int}()
        
        for i in 1:NClusters
            dists = [euclidean(Centers[:, i], X[:, j]) for j in 1:size(X, 2)]
            candidates = sortperm(dists)
        
            rep = findfirst(idx -> !(idx in chosen), candidates)
            if rep === nothing
                error("Could not find a unique representative for cluster $i")
            end
        
            push!(M, candidates[rep])
            push!(chosen, candidates[rep])
        end
    end

    if v
        println("Kmeans approach completed successfully.")
        println("A:", A)
        println("W:", W)
        println("M:", M)
    end

    return R, A, W, M, DistMatrix, clustering_time
end