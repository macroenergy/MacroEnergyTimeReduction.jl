# MacroEnergyTimeReduction.jl

Time-domain reduction methods for JuMP-based models.

This package is primarily used with `MacroEnergy.jl`, which provides upstream tools to prepare demand, price, and availability data in the clustering format, and downstream tools to map clustered representative periods back into an energy system model.

## Installation

In Julia package mode:

```julia
] add MacroEnergyTimeReduction
```

or from GitHub:

```julia
] add https://github.com/macroenergy/MacroEnergyTimeReduction.jl
```

## Basic workflow

1. Build a clustering input `DataFrame` from time-series model inputs.
    - Each column is one candidate period (e.g., one week).
    - Inside each column, stack the time series of each model feature (e.g., demand, price, availability).
    - For example, 52 candidate weeks at hourly resolution (`168` hours/week) and `8` features give a `(168 × 8)-by-52` `DataFrame`.
2. Choose a clustering method and number of representative periods.
3. Run `cluster(...)` (or a specific algorithm directly).
4. Use the returned assignments/representative indices in `MacroEnergy.jl` to reconstruct reduced-time inputs and map results back to the full model horizon.

## Available clustering algorithms

The package supports the following methods (passed as a string to `cluster`):

- `"kmeans"`: This method clusters candidate periods directly on the raw input data, and representatives are chosen as the nearest real periods to each centroid.
- `"kmedoids"`: This method clusters candidate periods directly on the raw input data (via pairwise Euclidean distances), and representatives are the selected medoid periods.
- `"autoencoder_sequential"`: This method trains an autoencoder on raw input data and then runs k-means on the learned latent space.
- `"autoencoder_simultaneous"`: This method trains an autoencoder with reconstruction + clustering-aware loss and then runs k-means on the learned latent space.

## Calling the algorithms

### Main entry point

```julia
using MacroEnergyTimeReduction

result = cluster(
    inpath,
    myTDRsetup,
    "kmeans",             # or "kmedoids", "autoencoder_sequential", "autoencoder_simultaneous"
    ClusteringInputDF,
    NClusters,
    nIters;
    period_idx = 1,
    v = false,
)

R, A, W, M, DistMatrix, autoencoder_training_time, clustering_time = result
```

### Direct calls (optional)

```julia
R, A, W, M, DistMatrix, clustering_time =
    cluster_kmeans(ClusteringInputDF, NClusters, nIters; v=false)

R, A, W, M, DistMatrix, clustering_time =
    cluster_kmedoids(ClusteringInputDF, NClusters, nIters; v=false)

R, A, W, M, DistMatrix, autoencoder_training_time, clustering_time =
    cluster_autoencoder_sequential(inpath, myTDRsetup, ClusteringInputDF, NClusters, nIters; period_idx=1, v=false)

R, A, W, M, DistMatrix, autoencoder_training_time, clustering_time =
    cluster_autoencoder_simultaneous(inpath, myTDRsetup, ClusteringInputDF, NClusters, nIters; period_idx=1, v=false)
```

## Input data format

`ClusteringInputDF` is expected to be a `DataFrame` with:

- **Columns** = candidate periods (for example, weeks)
- **Rows** = `TimestepsPerRepPeriod × n_features`

Interpretation:

- Each column is one full candidate representative period.
- Within each column, feature time series are stacked one after another across the representative-period timesteps.

Example:

- 52 candidate weeks,
- hourly resolution (`168` timesteps/week),
- 8 model features,

This setup results in a `(168 × 8)-by-52` `DataFrame`.

For autoencoder methods, `myTDRsetup` should include at least:

- `"TimestepsPerRepPeriod"`
- `"ScalingMethod"` (`"N"` for normalization, `"S"` for standardization)
- `"AutoEncoder"` dictionary keys:
  - `"kernel_size"`, `"stride"`, `"epochs"`, `"min_err_diff"`, `"patience"`, `"warmup"`, `"n_filters"`, `"latent_dim"`
  - and for simultaneous mode: `"lambda"`

Optional:

- `"ForceAutoencoderTraining" => 1` to force retraining even if a cached latent-space CSV exists in `inpath`.

## Output data format

`cluster(...)` returns:

```julia
[R, A, W, M, DistMatrix, autoencoder_training_time, clustering_time]
```

where:

- `R`: clustering result object (algorithm-specific, from `Clustering.jl`)
- `A`: assignments vector, length = number of candidate periods (cluster index for each column)
- `W`: cluster counts/weights, length = `NClusters`
- `M`: representative period indices (column indices of medoids/nearest representatives)
- `DistMatrix`: pairwise Euclidean distance matrix between candidate periods
- `autoencoder_training_time`: elapsed autoencoder training time or `"NA"` / reuse message for non-AE methods
- `clustering_time`: elapsed clustering time

In `MacroEnergy.jl`, `A`, `W`, and `M` are typically the key objects used to construct reduced-time inputs and remap model outputs.

## Small end-to-end example

```julia
using MacroEnergyTimeReduction
using DataFrames, Random

Random.seed!(123)

# Example setup: 24 hourly timesteps per representative period, 3 features,
# and 52 candidate periods (e.g., weeks)
timesteps = 24
n_features = 3
n_periods = 52

# Build synthetic clustering input with shape:
# rows = timesteps * n_features, columns = candidate periods
X = rand(timesteps * n_features, n_periods)
ClusteringInputDF = DataFrame(X, :auto)

# Minimal settings (needed by the generic cluster API)
myTDRsetup = Dict(
    "TimestepsPerRepPeriod" => timesteps,
    "ScalingMethod" => "N",
    "AutoEncoder" => Dict(
        "kernel_size" => 3,
        "stride" => 1,
        "epochs" => 50,
        "min_err_diff" => 1e-4,
        "patience" => 10,
        "warmup" => 5,
        "n_filters" => 8,
        "latent_dim" => 4,
        "lambda" => 0.1,
    ),
)

# Use kmeans (does not require training an autoencoder)
NClusters = 8
nIters = 20

R, A, W, M, DistMatrix, autoencoder_training_time, clustering_time = cluster(
    nothing,              # inpath (no latent-space caching)
    myTDRsetup,
    "kmeans",
    ClusteringInputDF,
    NClusters,
    nIters;
    v = false,
)

println("Assignments length: ", length(A))   # should be n_periods
println("Cluster weights: ", W)              # periods per cluster
println("Representative indices: ", M)       # selected period columns
println("Clustering time (s): ", clustering_time)
```

To map reduced-time data back into a full-year model, a typical approach is:

- Use `M` to extract representative period profiles.
- Use `A` to map each original period to its representative period.
- Use `W` as cluster weights in aggregated objective terms.
