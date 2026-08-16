# Shared-face thin-surface continuity prior — v0 proposal

## Status
Design only; do not enable in reconstruction experiments yet.

## Aim
Encourage two neighboring split-cell implicit surfaces to represent one continuous surface **only when both surfaces actually intersect their shared Voronoi face**. This is not a global normal-smoothing prior: it must not force unrelated tissue boundaries, inactive planes, or cells with negligible side contrast to align.

## Geometry
For Delaunay-neighbor sites `i,j`, their Voronoi shared face lies in the bisector plane
`B_ij = {x | (x - m_ij)·e_ij = 0}`, with `m_ij=(p_i+p_j)/2` and unit `e_ij=(p_j-p_i)/||p_j-p_i||`.

With frozen geometry, precompute each bounded interior shared face once from Delaunay tetrahedra:
1. Find tetrahedra incident to Delaunay edge `(i,j)`.
2. Compute their circumcenters; these are the Voronoi-face vertices.
3. Project vertices into a 2-D basis of `B_ij`, angular-sort, reject degenerate/unbounded/boundary-clipped faces, then cache face quadrature samples, area, and 2-D coordinates.
4. Rebuild the cache only after a triangulation/point update. The first experiment must use fixed cells with points hard-frozen.

The existing CSR adjacency alone is insufficient for exact face polygons; tetrahedra are already exposed by the triangulation bindings. A C++ face accessor is optional, not required for v0.

## Differentiable gated residual
Let the local learned implicit fields be `s_i(x)` and `s_j(x)` as used by the renderer. Restrict each to face samples `x_q`.

1. Compute a differentiable soft intersection score per cell from the face-restricted field, e.g. a temperature-controlled soft-min of `|s_i(x_q)|`, together with a soft sign-span score. Do not use a hard Python intersection branch in the loss.
2. Gate a pair only if both surfaces are near/crossing the shared face, both side contrasts are meaningful, and the face has adequate area. Use detached or slowly ramped gates initially to avoid a model escaping the prior merely by moving a plane away from the face.
3. At the shared face, compare the *zero-set* rather than raw normals alone. A practical first residual is the robust, gated difference of normalized signed-distance restrictions:
   `Huber((s_i/||∇_B s_i||) - sigma_ij*(s_j/||∇_B s_j||))`, averaged over quadrature samples near either zero set.
   Here `∇_B` is the local field gradient projected into the face; `sigma_ij∈{-1,+1}` resolves the arbitrary quaternion/sign convention. Choose it from a detached normal-dot sign or minimize softly over both signs.
4. Add a weaker direction residual for the two projected face gradients, sign-invariant under `n -> -n`. This stabilizes line direction but must not be the only term.

For a locally flat surface, the face restriction is a line. Equal normalized restrictions align both its orientation and its intercept; plain `1-(n_i·n_j)^2` would align directions but allow parallel displaced lines and is therefore inadequate.

## Cost
There are too many Delaunay edges to use all faces per iteration. Cache static faces and randomly sample roughly 8k–32k valid neighbor faces per optimization step. Use a small, fixed number (for example 8–16) of face quadrature samples. This needs only local thin-surface evaluations, not CT rays or the renderer backward path.

## Safety / ablation order
1. Unit test face reconstruction from synthetic point sets against known Voronoi faces.
2. Gradient-check the loss for quaternion and height parameters, including the sign-choice path.
3. Synthetic realization test: two manually continuous planes across a face should give near-zero loss; parallel offset planes and crossing/noncrossing pairs should separate correctly.
4. Frozen-geometry local teacher ablation with a known continuous surface. Sweep weight and gate temperature.
5. Measured-CT ablation only after the synthetic test: matched scalar/split, no geometry changes, fixed seeds, hard-side volume metrics and strict-air metrics.

## Risks
- CT data already showed weak/non-unique identification of independently fitted local orientations. This prior may make images look smoother while imposing a wrong surface.
- Unbounded/boundary faces and sliver Delaunay tetrahedra need explicit rejection.
- Curved K-height fields require face-restricted gradients; finite differences are acceptable for validation but autograd/analytic derivatives are required for production cost.
- Do not use this prior before point freezing, and do not interpret an improvement in projection loss alone as evidence of improved geometry.
