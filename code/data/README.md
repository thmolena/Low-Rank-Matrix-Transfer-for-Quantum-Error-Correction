# Data provenance

`source/syndromes_dataset.zip` is the version 1.0.1 archive from
[Zenodo record 11166209](https://doi.org/10.5281/zenodo.11166209), created by
Victor Yon and Gebremedhin A. Dagnew and released under CC BY 4.0.

The archive contains seven distance-three CSV files, one per physical fault
rate. Each represents 10,000,000 Stim-simulated shots aggregated by binary
label and 16-bit syndrome. The `quantity` column is the shot multiplicity and
must not be replaced by a unit row weight.

Expected SHA-256:

```text
bdfce36a71f04295ac78fb372d9c2e381801c05e3be119f919750ef59026d072
```

The reproduction command verifies this digest before reading a row. These are
published simulation data, not measurements from a quantum processor.
