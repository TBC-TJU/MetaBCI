# Reversibility and equivalence conditions for the MultiCSP OVO adapter

This note proves the transport mapping and its interaction with trial selection.
It makes no claim about unmeasured experimental results or complete API compatibility.

## Data and encoding

Let `X` have shape `(N, C, T)`: trials, channels, and samples per trial.
Assume `C > 0`, `T > 0`, a known channel count, and a fixed channel order.
Fitting also requires the original algorithm's sample and class-count conditions.
Channel names, sampling rate, units, reference, and epoch windows require separate metadata.
Use C-order throughout, with time varying fastest within each channel:

$$
E_C^{(N)}:\mathbb{R}^{N\times C\times T}\to\mathbb{R}^{N\times(CT)},
\qquad E_C^{(N)}(X)[i,cT+t]=X[i,c,t].
$$

Thus `(80, 8, 250)` becomes `(80, 2000)` and still has 80 trial rows.
No scaling, interpolation, channel permutation, or merging of trial rows is involved.

## Decoding and both inverse identities

For `Z` of shape `(N, M)`, require `M > 0` and divisibility by the known `C`.
Set `T = M/C` and define `D_C^(N)(Z)[i,c,t] = Z[i,cT+t]`.
Elementwise,

$$
D_C^{(N)}(E_C^{(N)}(X))[i,c,t]
=E_C^{(N)}(X)[i,cT+t]=X[i,c,t].
$$

Hence `D_C E_C = id`. Conversely, every `0 ≤ q < M` has a unique decomposition
`q = cT+t`, with `c = floor(q/T)` and `t = q mod T`, so

$$
E_C^{(N)}(D_C^{(N)}(Z))[i,q]=D_C^{(N)}(Z)[i,c,t]=Z[i,q].
$$

Thus `E_C D_C = id` on the stated domain. Divisibility alone cannot identify
the true channel count; an incorrect `C` gives incorrect channel/time semantics.

## Commutation with ordered trial selection

For an ordered index sequence `I = (i₀, …, iₖ₋₁)` of length `K`, let `S_I`
select only trial rows, with output row `j` taken from input row `i_j`.
Then, for every valid `j`, `c`, and `t`,

$$
\begin{aligned}
D_C^{(K)}(S_I(E_C^{(N)}(X)))[j,c,t]
&=S_I(E_C^{(N)}(X))[j,cT+t]\\
&=E_C^{(N)}(X)[i_j,cT+t]=X[i_j,c,t].
\end{aligned}
$$

Therefore `D_C^(K) S_I E_C^(N) = S_I`.
Each OVO pair must also use the same class order, indices, and binary coding,
such as `u → 0`, `v → 1`. Its reconstructed 3D data and targets then match.
With the same fitting algorithm, parameters, CV splits, and random choices,
the mathematical training computation, ordered CSP features, and outer predictions match.
For automatic component selection these conditions apply to every pair's inner search;
one global seed need not control parallel consumption of random choices.

## Validation, numerical, and interface boundaries

The real chain contains sklearn validation `V`: `D_C S_I V E_C`.
Validation must preserve values, dtype, row order, and target semantics, or its
conversions must separately be shown equivalent. Reshape alone proves no such API guarantee.
Copies or layout changes can affect rounding. Different libraries, versions, or
basis choices at repeated eigenvalues need not yield bitwise-identical fitted matrices.
Compare features within tolerances and predictions separately; equal accuracy is weaker.
Public `MultiCSP.transform` remains 3D and can use another trial count or sample length `T′`.
It must retain channel order/count and meaningful sampling metadata; shape flexibility
does not establish performance after changing sampling rate or epoch windows.
Internal OVO `estimator_.predict` takes flat rows and may require the training width `CT`.
This proof does not imply improved accuracy, complete package integration, or compatibility
with all third-party API versions. Those claims require their own measured evidence.
