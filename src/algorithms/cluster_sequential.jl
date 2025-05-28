@doc raw"""
    cluster_sequential(ClusteringInputDF, NClusters, nIters)

Get representative periods using cluster centers from autoencoder sequential method
"""

function cluster_sequential(ClusteringInputDF::DataFrame, NClusters::Int, nIters::Int, v::Bool=false)
    function get_batches_seq(data, batch_size)
        N = size(data, 1)
        idxs = randperm(N)
        batches = []
        for i in 1:batch_size:N
            upper = min(i + batch_size - 1, N)
            push!(batches, data[idxs[i:upper], :, :])
        end
        batches
    end
    # Transpose input data to align with correct time series clustering
    ClusteringInputDF_T = Matrix(ClusteringInputDF)' 

    if v
        println("Shape of ClusteringInputDF (before transpose): ", size(ClusteringInputDF))  
        println("Shape of ClusteringInputDF_T (after transpose): ", size(ClusteringInputDF_T))
    end

    # Define model hyperparameters
    input_dim = 1
    timesteps = size(ClusteringInputDF_T, 2)
    n_series = size(ClusteringInputDF_T, 1)
    n_filters = 30
    kernel_size = 5
    stride = 3
    latent_dim = 40
    lambda_param = 0.3
    padding = 1
    epochs = 10
    batch_size = 100

    # Encoder and Decoder definition
    encoder_net = Chain(
        x -> begin
            x_cwn = permutedims(x, (3, 2, 1))
            y = Conv((kernel_size,), input_dim=>n_filters; stride=(stride,), pad=(padding,))(x_cwn)
            y_ncw = permutedims(y, (3, 2, 1))
            z = leakyrelu.(y_ncw)
            flatten_y = flatten(permutedims(z, (3, 2, 1)))

            if v
                println("Shape of flatten_y: ", size(flatten_y))
            end

            return Dense(size(flatten_y, 1), latent_dim)(flatten_y)
        end
    )

    decoder_net = Chain(
        Dense(latent_dim, timesteps * input_dim),
        relu,
        x -> begin
            total_size = length(x)  # Total number of elements in the array
            new_batch_size = total_size ÷ (input_dim * timesteps)
            reshape(x, (new_batch_size, input_dim, timesteps))  # Ensure correct reshaping
        end
    )

    # Autoencoder function
    function autoencoder_seq(x)
        z = encoder_net(x)
        decoder_net(z)
    end

    # Reshape input for the model
    data_array = Float32.(ClusteringInputDF_T)
    data_ncw = reshape(data_array, :, 1, size(data_array, 2))  # (N, 1, T)

    if v
        println("Shape of data_ncw: ", size(data_ncw))
    end

    

    # Autoencoder Training setup
    opt = ADAM()
    training_loss = Float32[]
    opt_state = Flux.setup(ADAM(), (encoder_net, decoder_net))

    # Train Autoencoder
    for epoch in 1:epochs
        epoch_loss_acc = 0.0f0
        batches = get_batches_seq(data_ncw, batch_size)
        nbatches = length(batches)

        for batch_data in batches
            if v
                println("Shape of batch_data: ", size(batch_data))
            end

            encoded_data = encoder_net(batch_data)
            decoded_data = decoder_net(encoded_data)

            if v
                println("Shape of decoded_data: ", size(decoded_data))
            end

            # Compute autoencoder loss (MSE reconstruction loss)
            function loss_fn_seq()
                mean((batch_data .- decoded_data).^2)
            end

            # Compute gradients and update parameters
            gs = gradient(() -> loss_fn_seq(), params(encoder_net, decoder_net))
            Flux.update!(opt_state, (encoder_net, decoder_net), gs)
            epoch_loss_acc += loss_fn_seq()
        end

        epoch_loss = epoch_loss_acc / nbatches
        push!(training_loss, epoch_loss)

        if v
            println("Epoch $epoch/$epochs, Autoencoder Loss: $epoch_loss")
        end
    end

    println("Autoencoder Training Completed.")

    # Encode entire dataset after training
    encoded_data_all = encoder_net(data_ncw)

    # Perform KMeans clustering on encoded data
    R = kmeans(Matrix(encoded_data_all), NClusters, init=:kmcen)
    centroids_final = R.centers
    labels_final = R.assignments

    M = []
    ClusteringInputDF_indices = collect(1:size(ClusteringInputDF, 2))

    for i in 1:NClusters
        cluster_idxs = findall(x -> x == i, labels_final)  # indices of weeks in cluster i
        if !isempty(cluster_idxs)
            cluster_points = encoded_data_all[:, cluster_idxs]  # shape: (latent_dim, num_points)
            centroid = centroids_final[:, i]  # shape: (latent_dim,)
            dists = [euclidean(cluster_points[:, j], centroid) for j in 1:length(cluster_idxs)]
            closest_local_idx = argmin(dists)
            closest_global_idx = cluster_idxs[closest_local_idx]
            push!(M, ClusteringInputDF_indices[closest_global_idx])  # guaranteed: M[i] ∈ cluster i
        else
            push!(M, -1)  # fallback if a cluster is empty (shouldn’t happen)
        end
    end

    # Compute outputs
    A = labels_final
    W = [count(==(i), A) for i in 1:NClusters]

    # Correct distance matrix computation
    DistMatrix = pairwise(Euclidean(), Matrix(ClusteringInputDF_T), dims=2)

    println("Sequential autoencoder approach completed successfully.")

    return R, A, W, M, DistMatrix
end