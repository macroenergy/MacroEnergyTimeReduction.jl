@doc raw"""
    cluster_sequential(ClusteringInputDF, NClusters, nIters)

Get representative periods using cluster centers from k means on autoencoder latent space
"""
function cluster_sequential(myTDRsetup::Dict, ClusteringInputDF::DataFrame, NClusters::Int, nIters::Int, v::Bool=false)

    # Convert to matrix and transpose so each row = one time series
    data_matrix = Matrix(ClusteringInputDF)'
    println("Size after transpose: ", size(data_matrix))

    n_series, timesteps = size(data_matrix)

    # Reshape into (N, C, T) → Flux expects batch first
    data_ncw = reshape(Float32.(data_matrix), n_series, 1, timesteps)
    println("Size after reshape to (N, C, T): ", size(data_ncw))

    # Autoencoder hyperparameters from settings
    AE_params = myTDRsetup["AutoEncoder"]
    input_dim = AE_params["input_dim"]
    n_filters = AE_params["n_filters"]
    kernel_size = AE_params["kernel_size"]
    stride = AE_params["stride"]
    latent_dim = AE_params["latent_dim"]
    padding = AE_params["padding"]
    epochs = AE_params["epochs"]
    scaling_method = myTDRsetup["ScalingMethod"]

    decoder_activation = if scaling_method == "N"
        sigmoid
    elseif scaling_method == "S"
        identity
    else
        error("Unsupported ScalingMethod. Use 'N' for normalization or 'S' for standardization.")
    end

    println("Autoencoder parameters:")
    println("input_dim:", input_dim, ", n_filters:", n_filters, ", kernel_size:", kernel_size, ", stride:", stride, ", latent_dim:", latent_dim, ", padding:", padding, ", epochs:", epochs)

    conv_output_length = div(timesteps + 2 * padding - kernel_size, stride) + 1
    println("Conv output length: ", conv_output_length)
    flattened_dim = n_filters * conv_output_length

    # Define encoder and decoder as separate chains
    encoder_net = Chain(
        x -> permutedims(x, (3, 2, 1)),
        Conv((kernel_size,), input_dim => n_filters; stride=stride, pad=padding),
        x -> permutedims(x, (3, 2, 1)),
        leakyrelu,
        x -> reshape(x, size(x, 1), :),
        x -> x',
        Dense(flattened_dim, latent_dim)
    )

    decoder_net = Chain(
        Dense(latent_dim, input_dim * timesteps),
        decoder_activation,
        x -> begin
            bsz = size(x, 2)
            reshape(x, (bsz, input_dim, timesteps))
        end
    )

    # Combine encoder and decoder into a single autoencoder model
    autoencoder = Chain(encoder_net, decoder_net)

    # Set up the optimizer for the unified model
    opt = ADAM()
    opt_state = Flux.setup(opt, autoencoder)

    # Train autoencoder
    losses = Float32[]
    threshold = 1e-5
    patience = 5
    wait = 0

    println("\nStarting Autoencoder Training...")
    for epoch in 1:epochs
        # 4. Compute loss and gradients using the unified model
        loss, grads = Flux.withgradient(autoencoder) do m
            # The forward pass `m(data_ncw)` runs data through encoder then decoder
            decoded = m(data_ncw)
            return mean((decoded .- data_ncw).^2)
        end

        # 5. Update the unified model's parameters
        Flux.update!(opt_state, autoencoder, grads[1])

        # Track loss
        push!(losses, loss)
        if v || epoch % 10 == 0 # Print loss every 10 epochs or if verbose
            println("Epoch $epoch/$epochs, Loss: $loss")
        end

        # Early stopping
        if epoch > 1 && abs(loss - losses[end-1]) < threshold
            wait += 1
            if wait >= patience
                println("Early stopping triggered at epoch $epoch.")
                break
            end
        else
            wait = 0
        end
    end

    println("Autoencoder Training Completed.")

    # Convert to matrix and ensure shape is (latent_dim, n_series)
    encoded_data_all = encoder_net(data_ncw)

    # To perform K Means clustering on encoded data
    R = kmeans(Matrix(encoded_data_all), NClusters, init=:kmcen)
    centroids_final = R.centers;
    labels_final = R.assignments;

    M = []
    ClusteringInputDF_indices = collect(1:size(ClusteringInputDF, 2));

    for i in 1:NClusters
        cluster_idxs = findall(x -> x == i, labels_final)
        if !isempty(cluster_idxs)
            cluster_points = encoded_data_all[:, cluster_idxs]
            centroid = centroids_final[:, i]
            dists = [euclidean(cluster_points[:, j], centroid) for j in 1:length(cluster_idxs)]
            closest_local_idx = argmin(dists)
            closest_global_idx = cluster_idxs[closest_local_idx]
            push!(M, ClusteringInputDF_indices[closest_global_idx])
        else
            push!(M, -1)
        end
    end

    # Compute outputs
    ClusteringInputDF_T = Matrix(ClusteringInputDF)'; 

    A = labels_final;
    W = [count(==(i), A) for i in 1:NClusters];

    # Correct distance matrix computation
    DistMatrix = pairwise(Euclidean(), Matrix(ClusteringInputDF_T), dims=2);

    println("Sequential autoencoder approach completed successfully.")

    return R, A, W, M, DistMatrix
end
