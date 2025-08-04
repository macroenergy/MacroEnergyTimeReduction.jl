@doc raw"""
    cluster_sequential(ClusteringInputDF, NClusters, nIters)

Get representative periods using cluster centers from k means on autoencoder latent space
"""

function cluster_sequential(myTDRsetup::Dict, ClusteringInputDF::DataFrame, NClusters::Int, nIters::Int, v::Bool=false)

    # Convert to matrix and transpose so each row = one time series
    data_matrix = Matrix(ClusteringInputDF)'
    println("Size after transpose: ", size(data_matrix))  # should be (n_series, timesteps)

    # Sanity check
    n_series, timesteps = size(data_matrix)

    # Reshape into (N, C, T) → Flux expects batch first
    # We'll use 1 channel (C = 1) since this is univariate
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

    if scaling_method == "N"
        decoder_activation = sigmoid  # sigmoid for normalized (0–1)
    elseif scaling_method == "S"
        decoder_activation = identity  # linear (no activation) for standardized (mean 0, std 1)
    else
        error("Unsupported ScalingMethod. Use 'N' for normalization or 'S' for standardization.")
    end

    println("Autoencoder parameters:")
    println("input_dim:", input_dim)
    println("n_filters:", n_filters)
    println("kernel_size:", kernel_size)
    println("stride:", stride)
    println("latent_dim:", latent_dim)
    println("padding:", padding)
    println("epochs:", epochs)

    timesteps = size(data_ncw, 3)
    conv_output_length = div(timesteps + 2 * padding - kernel_size, stride) + 1
    println("Conv output length: ", conv_output_length)

    flattened_dim = n_filters * conv_output_length

    #Define encoder and decoder function
    encoder_net = Chain(
        x -> permutedims(x, (3, 2, 1)),  # (N, 1, T) → (T, 1, N)
        Conv((kernel_size,), input_dim => n_filters; stride=stride, pad=padding),
        x -> permutedims(x, (3, 2, 1)),  # back to (N, F, T')
        x -> leakyrelu.(x),
        x -> reshape(x, size(x, 1), :),  # flatten: (data_size, flatten_dim)
        x -> x',                         # transpose to (flatten_dim)
        Dense(flattened_dim, latent_dim)
    )

    decoder_net = Chain(
        Dense(latent_dim, input_dim * timesteps),
        decoder_activation,
        x -> begin
            bsz = size(x, 2)  # number of samples
            reshape(x, (bsz, input_dim, timesteps))  # reshape to (data_size, 1, T)
        end
    )

    #Train autoencoder
    opt = ADAM()
    losses = Float32[]

    threshold = 1e-5    # Stop if loss change < threshold
    patience = 5        # Allow some tolerance before stopping
    wait = 0            # How many epochs we've waited

    ps = Flux.params(encoder_net, decoder_net)
    st = Flux.setup(opt, ps)

    for epoch in 1:epochs
        # Define dynamic loss function (fresh forward pass each time)
        function loss_fn()
            z = encoder_net(data_ncw)          # Encode
            decoded = decoder_net(z)           # Decode
            return mean((decoded .- data_ncw).^2)
        end

        # Compute gradients and update using setup state
        grads = gradient(() -> loss_fn(), ps)
        Flux.update!(st, ps, grads)

        # Compute and record current loss
        current_loss = loss_fn()
        push!(losses, current_loss)
        println("Epoch $epoch/$epochs, Loss: $current_loss")

        # Check stopping condition
        if epoch > 1 && abs(current_loss - losses[end-1]) < threshold
            wait += 1
            if wait >= patience
                println("Early stopping: loss change below threshold $threshold for $patience epochs.")
                break
            end
        else
            wait = 0  # reset wait counter if improvement found
        end
    end

    println("Autoencoder Training Completed.")


    # Convert to matrix and ensure shape is (weeks, latent_dim)
    encoded_data_all = encoder_net(data_ncw)

    # To perform K Means clustering on encoded data
    R = kmeans(Matrix(encoded_data_all), NClusters, init=:kmcen)
    centroids_final = R.centers;
    labels_final = R.assignments;

    M = []
    ClusteringInputDF_indices = collect(1:size(ClusteringInputDF, 2));

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
    ClusteringInputDF_T = Matrix(ClusteringInputDF)'; 

    A = labels_final;
    W = [count(==(i), A) for i in 1:NClusters];

    # Correct distance matrix computation
    DistMatrix = pairwise(Euclidean(), Matrix(ClusteringInputDF_T), dims=2);

    println("Sequential autoencoder approach completed successfully.")

    return R, A, W, M, DistMatrix
end