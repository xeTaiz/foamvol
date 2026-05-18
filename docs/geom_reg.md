# Boundary & Geometric Regularizers for the Voronoi Foam

Working document explaining the existing `boundary_alignment_regularization`
(BA) and four proposed alternatives (A, B, C, D).
LaTeX renders in VS Code preview / GitHub / pandoc. Inline SVGs render
locally; GitHub strips them for security, so view via VS Code preview or
open as HTML.

---

## 1. Setup and notation

The Voronoi foam is a set of points $\{p_i\}_{i=1}^N \subset \mathbb{R}^3$
with a Delaunay neighbor graph. For each cell $i$:

- $p_i$ — position
- $\mu_i$ — density (after softplus activation)
- $r_i$ — cell radius (farthest-neighbor distance, proxy for cell size)
- $N(i)$ — Voronoi neighbors

For each directed edge $(i, j)$ with $j \in N(i)$:

- **Direction:** $\quad n_{ij} = \dfrac{p_j - p_i}{\|p_j - p_i\|}$
- **Density jump:** $\quad \Delta\mu_{ij} = \mu_j - \mu_i$
- **Importance weight:** $\quad w_{ij} = (\Delta\mu_{ij})^2 \cdot r_i\, r_j$

The only signal we have about local density structure is the discrete edge
difference $\Delta\mu_{ij}$. There is no continuous gradient field —
everything operates on the neighborhood graph.

A **same-density gating weight** (used by losses that compare neighbors):

$$
s_{ij} = \exp\!\left(-\frac{(\Delta\mu_{ij})^2}{\sigma_v^2}\right)
$$

Note $w_{ij}$ favors **high-jump** edges (these define the boundary structure
inside one cell), while $s_{ij}$ favors **low-jump** edges (used to gate
*comparisons between neighbors*: we only ask same-density cells to agree).

---

## 2. The cell-boundary tensor $M_i$

The shared building block for BA, A, B, and C is a $3 \times 3$ symmetric
positive-semidefinite matrix per cell:

$$
M_i \;=\; \sum_{j \in N(i)} w_{ij}\; n_{ij}\, n_{ij}^{\!\top}
$$

This is a weighted scatter of unit edge directions, where high-density-jump
edges dominate. Its eigendecomposition

$$
M_i \;=\; \sum_{k=1}^{3} \lambda_k\, v_k\, v_k^{\!\top}, \qquad
\lambda_1 \geq \lambda_2 \geq \lambda_3 \geq 0
$$

reveals the **shape of the local boundary structure**:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 270" width="100%" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12">
  <style>
    .cell      { fill: #1a1a1a; }
    .nbr_lo    { fill: #f8fafc; stroke: #1a1a1a; stroke-width: 1; }
    .nbr_hi    { fill: #f8fafc; stroke: #1a1a1a; stroke-width: 1; }
    .edge_lo   { stroke: #b0b8c4; stroke-width: 1; }
    .edge_hi   { stroke: #dc2626; stroke-width: 2; }
    .bg_hi     { fill: #efe6df; }
    .bg_lo     { fill: #f0f4f8; }
    .label     { fill: #1a1a1a; }
    .ellipse   { fill: #dc2626; fill-opacity: 0.18; stroke: #dc2626; stroke-width: 1.5; }
    .bline     { stroke: #1a1a1a; stroke-width: 0.5; }
    @media (prefers-color-scheme: dark) {
      .cell    { fill: #e4e4e7; }
      .nbr_lo  { fill: #27272a; stroke: #a1a1aa; }
      .nbr_hi  { fill: #27272a; stroke: #a1a1aa; }
      .edge_lo { stroke: #52525b; }
      .edge_hi { stroke: #f87171; }
      .bg_hi   { fill: #3b1d10; }
      .bg_lo   { fill: #0d1f30; }
      .label   { fill: #e4e4e7; }
      .ellipse { fill: #f87171; stroke: #f87171; }
      .bline   { stroke: #a1a1aa; }
    }
  </style>

  <!-- Panel A: planar boundary -->
  <g transform="translate(20,10)">
    <text x="100" y="0" text-anchor="middle" class="label" font-weight="600">Planar boundary</text>
    <rect class="bg_hi" x="0" y="20" width="200" height="100"/>
    <rect class="bg_lo" x="0" y="120" width="200" height="105"/>
    <line class="bline" x1="0" y1="120" x2="200" y2="120"/>
    <line class="edge_lo" x1="100" y1="100" x2="50"  y2="70"/>
    <line class="edge_lo" x1="100" y1="100" x2="150" y2="70"/>
    <line class="edge_lo" x1="100" y1="100" x2="80"  y2="50"/>
    <line class="edge_hi" x1="100" y1="100" x2="60"  y2="160"/>
    <line class="edge_hi" x1="100" y1="100" x2="135" y2="170"/>
    <line class="edge_hi" x1="100" y1="100" x2="100" y2="195"/>
    <circle class="nbr_lo" cx="50"  cy="70"  r="4"/>
    <circle class="nbr_lo" cx="150" cy="70"  r="4"/>
    <circle class="nbr_lo" cx="80"  cy="50"  r="4"/>
    <circle class="nbr_hi" cx="60"  cy="160" r="4"/>
    <circle class="nbr_hi" cx="135" cy="170" r="4"/>
    <circle class="nbr_hi" cx="100" cy="195" r="4"/>
    <circle class="cell"   cx="100" cy="100" r="5"/>
    <ellipse class="ellipse" cx="100" cy="250" rx="6" ry="22"/>
    <text x="100" y="262" text-anchor="middle" class="label" font-size="11">stick: λ₁ ≫ λ₂≈λ₃≈0</text>
  </g>

  <!-- Panel B: ridge -->
  <g transform="translate(260,10)">
    <text x="100" y="0" text-anchor="middle" class="label" font-weight="600">Ridge / corner</text>
    <path d="M0,40 L100,120 L200,40 L200,225 L0,225 Z" class="bg_lo"/>
    <path d="M0,40 L100,120 L200,40 L200,0 L0,0 Z"    class="bg_hi"/>
    <path d="M0,40 L100,120 L200,40" class="bline" fill="none"/>
    <line class="edge_hi" x1="100" y1="140" x2="40"  y2="80"/>
    <line class="edge_hi" x1="100" y1="140" x2="160" y2="80"/>
    <line class="edge_hi" x1="100" y1="140" x2="100" y2="70"/>
    <line class="edge_lo" x1="100" y1="140" x2="60"  y2="200"/>
    <line class="edge_lo" x1="100" y1="140" x2="140" y2="200"/>
    <line class="edge_lo" x1="100" y1="140" x2="100" y2="210"/>
    <circle class="nbr_hi" cx="40"  cy="80"  r="4"/>
    <circle class="nbr_hi" cx="160" cy="80"  r="4"/>
    <circle class="nbr_hi" cx="100" cy="70"  r="4"/>
    <circle class="nbr_lo" cx="60"  cy="200" r="4"/>
    <circle class="nbr_lo" cx="140" cy="200" r="4"/>
    <circle class="nbr_lo" cx="100" cy="210" r="4"/>
    <circle class="cell"   cx="100" cy="140" r="5"/>
    <ellipse class="ellipse" cx="100" cy="250" rx="22" ry="6"/>
    <text x="100" y="262" text-anchor="middle" class="label" font-size="11">disk: λ₁ ≈ λ₂ ≫ λ₃</text>
  </g>

  <!-- Panel C: interior, no structure -->
  <g transform="translate(500,10)">
    <text x="100" y="0" text-anchor="middle" class="label" font-weight="600">Interior — no structure</text>
    <rect class="bg_hi" x="0" y="20" width="200" height="205"/>
    <line class="edge_lo" x1="100" y1="120" x2="40"  y2="70"/>
    <line class="edge_lo" x1="100" y1="120" x2="160" y2="70"/>
    <line class="edge_lo" x1="100" y1="120" x2="60"  y2="180"/>
    <line class="edge_lo" x1="100" y1="120" x2="150" y2="190"/>
    <line class="edge_lo" x1="100" y1="120" x2="40"  y2="140"/>
    <line class="edge_lo" x1="100" y1="120" x2="170" y2="130"/>
    <circle class="nbr_lo" cx="40"  cy="70"  r="4"/>
    <circle class="nbr_lo" cx="160" cy="70"  r="4"/>
    <circle class="nbr_lo" cx="60"  cy="180" r="4"/>
    <circle class="nbr_lo" cx="150" cy="190" r="4"/>
    <circle class="nbr_lo" cx="40"  cy="140" r="4"/>
    <circle class="nbr_lo" cx="170" cy="130" r="4"/>
    <circle class="cell"   cx="100" cy="120" r="5"/>
    <ellipse class="ellipse" cx="100" cy="250" rx="14" ry="14"/>
    <text x="100" y="262" text-anchor="middle" class="label" font-size="11">sphere: λ₁≈λ₂≈λ₃ (small)</text>
  </g>
</svg>

**Reading the diagrams.** Red edges are high-$\Delta\mu$ neighbors (the
"boundary edges"); gray edges are same-density. The ellipse below each cell
is the 2D analog of $M_i$ with semi-axes proportional to $\sqrt{\lambda_k}$
along $v_k$.

- **Planar boundary** — all high-jump edges point along the surface
  normal. $M_i$ is rank-1: $\lambda_1$ dominates, $v_1$ ≈ surface normal.
- **Ridge / corner** — high-jump edges span a plane. $M_i$ is rank-2:
  $\lambda_1 \approx \lambda_2 \gg \lambda_3$. $v_3$ is the ridge tangent.
- **Interior** — no high-jump edges. $\operatorname{tr}(M_i) \approx 0$ and
  $M_i$ is effectively unstructured. These cells are masked out.

Define the normalized version $\hat M_i = M_i / \operatorname{tr}(M_i)$
(removes magnitude, keeps shape) and the validity mask

$$
m_i \;=\; \mathbb{1}\!\big[\operatorname{tr}(M_i) > \tau\big]
\quad\text{with}\quad
\tau = 0.01 \cdot \operatorname{median}_k \operatorname{tr}(M_k).
$$

The **top eigenvector** $v_i := v_1(M_i)$ is the local *surface-normal
proxy*. All four direction-based losses (BA, A, C) center on this vector.

---

## 3. Existing BA — Frobenius distance on $\hat M$

Implemented at `radfoam_model/scene.py:990`.

$$
\boxed{\;
L_{\text{BA}} \;=\; \frac{1}{|E|}\sum_{(i,j)\in E}
m_{ij}\; s_{ij}\; \big\|\hat M_i - \hat M_j\big\|_F^2
\;}
$$

where $m_{ij} = m_i \wedge m_j$ and the sum runs over the (directed) edge
list $E$ of same-density neighbors.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 220" width="100%" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12">
  <style>
    .cell  { fill: #1a1a1a; }
    .edge  { stroke: #1a1a1a; stroke-width: 1.5; }
    .ell_a { fill: #dc2626; fill-opacity: 0.2; stroke: #dc2626; stroke-width: 1.5; }
    .ell_b { fill: #1d4ed8; fill-opacity: 0.2; stroke: #1d4ed8; stroke-width: 1.5; }
    .label { fill: #1a1a1a; }
    .warn  { fill: #dc2626; }
    @media (prefers-color-scheme: dark) {
      .cell  { fill: #e4e4e7; }
      .edge  { stroke: #a1a1aa; }
      .ell_a { fill: #f87171; stroke: #f87171; }
      .ell_b { fill: #60a5fa; stroke: #60a5fa; }
      .label { fill: #e4e4e7; }
      .warn  { fill: #fca5a5; }
    }
  </style>
  <g transform="translate(40,30)">
    <text x="0" y="0" class="label" font-weight="600">Two same-density neighbors:</text>
    <circle class="cell" cx="80"  cy="80" r="6"/>
    <circle class="cell" cx="280" cy="80" r="6"/>
    <line class="edge" x1="80" y1="80" x2="280" y2="80" stroke-dasharray="4,3"/>
    <text x="180" y="74" text-anchor="middle" class="label" font-size="11">edge (i,j)</text>
    <ellipse class="ell_a" cx="80"  cy="150" rx="38" ry="8"  transform="rotate(20 80 150)"/>
    <text x="80"  y="180" text-anchor="middle" font-size="11" fill="#dc2626">M̂ᵢ</text>
    <ellipse class="ell_b" cx="280" cy="150" rx="32" ry="14" transform="rotate(40 280 150)"/>
    <text x="280" y="180" text-anchor="middle" font-size="11" fill="#1d4ed8">M̂ⱼ</text>
  </g>
  <g transform="translate(420,30)">
    <text x="0" y="0"   class="label" font-weight="600">L_BA penalizes the full tensor difference:</text>
    <text x="0" y="35"  class="label" font-size="14">‖M̂ᵢ − M̂ⱼ‖²<tspan baseline-shift="sub" font-size="9">F</tspan></text>
    <text x="0" y="80"  class="label">= disagreement in direction (vᵢ vs vⱼ)</text>
    <text x="0" y="100" class="label">+ disagreement in anisotropy spectrum</text>
    <text x="0" y="120" class="label">+ disagreement in 2nd / 3rd eigenvecs</text>
    <text x="0" y="160" class="warn" font-size="11">⚠ Two cells at the same flat edge may legitimately have</text>
    <text x="0" y="175" class="warn" font-size="11">different anisotropy magnitudes (triangulation combinatorics),</text>
    <text x="0" y="190" class="warn" font-size="11">and BA penalizes that anyway.</text>
  </g>
</svg>

**What it really penalizes.** The Frobenius distance lumps three signals:
(1) disagreement in the surface-normal direction $v_1$, (2) disagreement in
the *amount* of anisotropy (eigenvalue spectrum), (3) disagreement in the
orientation of $v_2, v_3$ (often noisy when $\lambda_2 \approx \lambda_3$).
For jaggedness reduction we care almost entirely about (1).

---

## 4. Proposal A — top-eigenvector only

Drop everything except the surface-normal direction.

$$
\boxed{\;
L_A \;=\; \frac{1}{|E|}\sum_{(i,j)\in E}
m_{ij}\; s_{ij}\; \big(1 - (v_i \cdot v_j)^2\big)
\;}
$$

The $(\cdot)^2$ handles the sign ambiguity of eigenvectors ($v_i$ and $-v_i$
represent the same axis). The term is $0$ when $v_i \parallel \pm v_j$ and
$1$ when $v_i \perp v_j$.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 190" width="100%" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12">
  <style>
    .cell    { fill: #1a1a1a; }
    .arr_red { stroke: #dc2626; stroke-width: 2.5; fill: none; marker-end: url(#arr_r); }
    .arr_blu { stroke: #1d4ed8; stroke-width: 2.5; fill: none; marker-end: url(#arr_b); }
    .label   { fill: #1a1a1a; }
    .note_g  { fill: #15803d; }
    @media (prefers-color-scheme: dark) {
      .cell   { fill: #e4e4e7; }
      .label  { fill: #e4e4e7; }
      .note_g { fill: #4ade80; }
      .arr_red { stroke: #f87171; }
      .arr_blu { stroke: #60a5fa; }
    }
  </style>
  <defs>
    <marker id="arr_r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 Z" fill="#dc2626"/>
    </marker>
    <marker id="arr_b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 Z" fill="#1d4ed8"/>
    </marker>
  </defs>
  <g transform="translate(40,30)">
    <text x="0" y="0" class="label" font-weight="600">Only the top eigenvector direction is compared:</text>
    <circle class="cell" cx="80"  cy="90" r="5"/>
    <circle class="cell" cx="280" cy="90" r="5"/>
    <line class="arr_red" x1="80"  y1="90" x2="80"  y2="38"/>
    <line class="arr_blu" x1="280" y1="90" x2="261" y2="40"/>
    <text x="80"  y="118" text-anchor="middle" font-size="11" fill="#dc2626">vᵢ</text>
    <text x="280" y="118" text-anchor="middle" font-size="11" fill="#1d4ed8">vⱼ</text>
  </g>
  <g transform="translate(420,30)">
    <text x="0" y="0"   class="label" font-weight="600">1 − (vᵢ · vⱼ)²  =  sin²θ</text>
    <text x="0" y="50"  class="label">→ 0 when axes align (parallel or anti-parallel)</text>
    <text x="0" y="72"  class="label">→ 1 when axes are perpendicular</text>
    <text x="0" y="110" class="note_g" font-size="11">Ignores anisotropy magnitude entirely.</text>
    <text x="0" y="128" class="note_g" font-size="11">Still pairwise → still penalizes smooth curvature.</text>
  </g>
</svg>

**Compared to BA:** identical scaffolding (same $M_i$, same gating), but
ignores eigenvalue magnitudes and lower eigenvectors. Cheaper, cleaner
signal — but still penalizes smooth curvature (it's pairwise).

**Numerical caveat:** $v_1$ is differentiable via `torch.linalg.eigh` but
its gradient blows up when $\lambda_1$ and $\lambda_2$ get close. For
boundary cells that's fine; for ridge cells ($\lambda_1 \approx \lambda_2$)
the gradient is noisy. The mask $m_i$ partially shields this.

---

## 5. Proposal B — intrinsic planarity

Not pairwise. Penalize how *non-planar* each cell's own boundary structure
is.

$$
\boxed{\;
L_B \;=\; \frac{1}{|V|}\sum_i m_i \cdot \frac{\lambda_3(M_i)}{\operatorname{tr}(M_i)}
\;}
$$

The ratio lies in $[0, 1/3]$. It is $0$ exactly when $M_i$ has rank
$\leq 2$, i.e. all high-jump edges from cell $i$ lie in a plane. Driving
$L_B \to 0$ forces every cell's local boundary to look locally planar.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 235" width="100%" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12">
  <style>
    .cell    { fill: #1a1a1a; }
    .nbr     { fill: #f8fafc; stroke: #1a1a1a; stroke-width: 1; }
    .edge_hi { stroke: #dc2626; stroke-width: 2; }
    .edge_lo { stroke: #b0b8c4; stroke-width: 1; }
    .plane   { stroke: #15803d; stroke-width: 1.25; stroke-dasharray: 5,3; fill: none; }
    .label   { fill: #1a1a1a; }
    .good    { fill: #15803d; }
    .bad     { fill: #dc2626; }
    @media (prefers-color-scheme: dark) {
      .cell    { fill: #e4e4e7; }
      .nbr     { fill: #27272a; stroke: #a1a1aa; }
      .edge_hi { stroke: #f87171; }
      .edge_lo { stroke: #52525b; }
      .plane   { stroke: #4ade80; }
      .label   { fill: #e4e4e7; }
      .good    { fill: #4ade80; }
      .bad     { fill: #f87171; }
    }
  </style>
  <!-- L_B small: planar boundary -->
  <g transform="translate(30,30)">
    <text x="100" y="0" text-anchor="middle" class="label" font-weight="600">L_B small (planar)</text>
    <line class="plane" x1="0" y1="115" x2="200" y2="115"/>
    <line class="edge_hi" x1="100" y1="115" x2="50"  y2="50"/>
    <line class="edge_hi" x1="100" y1="115" x2="100" y2="40"/>
    <line class="edge_hi" x1="100" y1="115" x2="150" y2="50"/>
    <line class="edge_hi" x1="100" y1="115" x2="60"  y2="180"/>
    <line class="edge_hi" x1="100" y1="115" x2="140" y2="180"/>
    <circle class="nbr" cx="50"  cy="50"  r="4"/>
    <circle class="nbr" cx="100" cy="40"  r="4"/>
    <circle class="nbr" cx="150" cy="50"  r="4"/>
    <circle class="nbr" cx="60"  cy="180" r="4"/>
    <circle class="nbr" cx="140" cy="180" r="4"/>
    <circle class="cell" cx="100" cy="115" r="5"/>
    <text x="100" y="220" text-anchor="middle" class="good" font-size="11">λ₃ ≈ 0: edges lie in a plane</text>
  </g>
  <!-- L_B medium: ridge -->
  <g transform="translate(260,30)">
    <text x="100" y="0" text-anchor="middle" class="label" font-weight="600">L_B medium (ridge)</text>
    <line class="edge_hi" x1="100" y1="115" x2="40"  y2="60"/>
    <line class="edge_hi" x1="100" y1="115" x2="160" y2="60"/>
    <line class="edge_hi" x1="100" y1="115" x2="100" y2="40"/>
    <line class="edge_hi" x1="100" y1="115" x2="50"  y2="170"/>
    <line class="edge_hi" x1="100" y1="115" x2="150" y2="170"/>
    <line class="edge_hi" x1="100" y1="115" x2="100" y2="190"/>
    <circle class="nbr" cx="40"  cy="60"  r="4"/>
    <circle class="nbr" cx="160" cy="60"  r="4"/>
    <circle class="nbr" cx="100" cy="40"  r="4"/>
    <circle class="nbr" cx="50"  cy="170" r="4"/>
    <circle class="nbr" cx="150" cy="170" r="4"/>
    <circle class="nbr" cx="100" cy="190" r="4"/>
    <circle class="cell" cx="100" cy="115" r="5"/>
    <text x="100" y="220" text-anchor="middle" class="bad" font-size="11">λ₃ &gt; 0: edges span a 2D fan</text>
  </g>
  <!-- L_B large: isotropic -->
  <g transform="translate(490,30)">
    <text x="100" y="0" text-anchor="middle" class="label" font-weight="600">L_B large (isotropic)</text>
    <line class="edge_hi" x1="100" y1="115" x2="30"  y2="60"/>
    <line class="edge_hi" x1="100" y1="115" x2="170" y2="55"/>
    <line class="edge_hi" x1="100" y1="115" x2="50"  y2="180"/>
    <line class="edge_hi" x1="100" y1="115" x2="160" y2="190"/>
    <line class="edge_hi" x1="100" y1="115" x2="40"  y2="120"/>
    <line class="edge_hi" x1="100" y1="115" x2="180" y2="130"/>
    <line class="edge_hi" x1="100" y1="115" x2="105" y2="35"/>
    <circle class="nbr" cx="30"  cy="60"  r="4"/>
    <circle class="nbr" cx="170" cy="55"  r="4"/>
    <circle class="nbr" cx="50"  cy="180" r="4"/>
    <circle class="nbr" cx="160" cy="190" r="4"/>
    <circle class="nbr" cx="40"  cy="120" r="4"/>
    <circle class="nbr" cx="180" cy="130" r="4"/>
    <circle class="nbr" cx="105" cy="35"  r="4"/>
    <circle class="cell" cx="100" cy="115" r="5"/>
    <text x="100" y="220" text-anchor="middle" class="bad" font-size="11">λ₃ ≈ λ₂ ≈ λ₁ ≫ 0</text>
  </g>
</svg>

**Compared to BA / A / C:** not pairwise. Operates on each cell on its own.
Says nothing about whether neighbors *agree* — only whether each individual
cell has a locally planar boundary.

**Stability variant.** $\lambda_3$ has ill-conditioned gradients near
$\lambda_2 \approx \lambda_3$. A numerically friendlier surrogate uses the
determinant:

$$
L_B^{\det} \;=\; \frac{1}{|V|}\sum_i m_i \cdot
\frac{\det(M_i)}{\operatorname{tr}(M_i)^3}
\;=\; \frac{\lambda_1 \lambda_2 \lambda_3}{(\lambda_1+\lambda_2+\lambda_3)^3}
$$

This goes to zero whenever *any* eigenvalue is small — so it also rewards
"stick" cells (planar boundary), and avoids `eigh` entirely.

**Why this complements C:** $L_B$ enforces *intra-cell* flatness. $L_C$
enforces *inter-cell* smoothness of the normal. If both go to zero
simultaneously, every cell's local boundary is flat AND adjacent flat
patches align — that's a smooth iso-surface by construction.

---

## 6. Proposal C — graph Laplacian on the normal direction

Same $v_i$ as A, but compared against the *neighborhood mean*, not pairwise.

For each cell $i$, define the **sign-aligned** neighbor mean (alignment
needed because of the eigenvector sign ambiguity):

$$
\bar v_i \;=\; \frac{\displaystyle\sum_{j \in N(i)} s_{ij}\, \sigma_{ij}\, v_j}
                  {\displaystyle\sum_{j \in N(i)} s_{ij}}, \qquad
\sigma_{ij} \;=\; \operatorname{sign}(v_i \cdot v_j)
$$

Then:

$$
\boxed{\;
L_C \;=\; \frac{1}{|V|}\sum_i m_i\, \big\|v_i - \bar v_i\big\|^2
\;}
$$

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 240" width="100%" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12">
  <style>
    .cell    { fill: #1a1a1a; }
    .curve   { stroke: #15803d; stroke-width: 1.5; fill: none; stroke-dasharray: 6,3; }
    .kink    { stroke: #dc2626; stroke-width: 1.5; fill: none; stroke-dasharray: 6,3; }
    .normal  { stroke: #1d4ed8; stroke-width: 2.5; fill: none; marker-end: url(#arr_n); }
    .label   { fill: #1a1a1a; }
    .good    { fill: #15803d; }
    .bad     { fill: #dc2626; }
    @media (prefers-color-scheme: dark) {
      .cell   { fill: #e4e4e7; }
      .curve  { stroke: #4ade80; }
      .kink   { stroke: #f87171; }
      .normal { stroke: #60a5fa; marker-end: url(#arr_n_dk); }
      .label  { fill: #e4e4e7; }
      .good   { fill: #4ade80; }
      .bad    { fill: #f87171; }
    }
  </style>
  <defs>
    <marker id="arr_n" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 Z" fill="#1d4ed8"/>
    </marker>
    <marker id="arr_n_dk" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 Z" fill="#60a5fa"/>
    </marker>
  </defs>

  <!-- Left: smoothly curved — L_C ≈ 0 -->
  <g transform="translate(20,20)">
    <text x="160" y="0" text-anchor="middle" class="label" font-weight="600">Smoothly curved iso-surface</text>
    <path class="curve" d="M10,155 Q160,42 310,155"/>
    <circle class="cell" cx="30"  cy="140" r="4"/>
    <line class="normal" x1="30"  y1="140" x2="14"  y2="108"/>
    <circle class="cell" cx="90"  cy="100" r="4"/>
    <line class="normal" x1="90"  y1="100" x2="80"  y2="60"/>
    <circle class="cell" cx="160" cy="82"  r="4"/>
    <line class="normal" x1="160" y1="82"  x2="160" y2="42"/>
    <circle class="cell" cx="230" cy="100" r="4"/>
    <line class="normal" x1="230" y1="100" x2="240" y2="60"/>
    <circle class="cell" cx="290" cy="140" r="4"/>
    <line class="normal" x1="290" y1="140" x2="306" y2="108"/>
    <text x="160" y="205" text-anchor="middle" class="good" font-size="11">L_C ≈ 0: vᵢ ≈ mean of neighbors' vⱼ (smooth rotation)</text>
    <text x="160" y="222" text-anchor="middle" class="bad"  font-size="11">L_BA, L_A &gt; 0: pairwise vᵢ ≠ vⱼ fires here too (penalizes curvature)</text>
  </g>

  <!-- Right: kink — L_C fires -->
  <g transform="translate(380,20)">
    <text x="160" y="0" text-anchor="middle" class="label" font-weight="600">Kink (zig-zag boundary)</text>
    <path class="kink" d="M10,155 L80,72 L120,112 L200,52 L310,155"/>
    <circle class="cell" cx="40"  cy="122" r="4"/>
    <line class="normal" x1="40"  y1="122" x2="20"  y2="95"/>
    <circle class="cell" cx="95"  cy="80"  r="4"/>
    <line class="normal" x1="95"  y1="80"  x2="130" y2="60"/>
    <circle class="cell" cx="162" cy="88"  r="4"/>
    <line class="normal" x1="162" y1="88"  x2="136" y2="68"/>
    <circle class="cell" cx="222" cy="72"  r="4"/>
    <line class="normal" x1="222" y1="72"  x2="257" y2="57"/>
    <circle class="cell" cx="272" cy="128" r="4"/>
    <line class="normal" x1="272" y1="128" x2="292" y2="100"/>
    <text x="160" y="205" text-anchor="middle" class="bad" font-size="11">L_C &gt; 0: neighbor-mean averages across kink ≠ vᵢ</text>
    <text x="160" y="222" text-anchor="middle" class="bad" font-size="11">L_BA &gt; 0 too — but can't distinguish kink from curvature</text>
  </g>
</svg>

**The key advantage over BA and A.** On a *smoothly curved* iso-surface,
each cell's surface normal rotates slowly. The neighborhood-mean $\bar v_i$
closely tracks $v_i$, so $L_C \approx 0$. But every adjacent pair $(i,j)$
has $v_i \neq v_j$, so pairwise losses (BA, A) fire continuously — they
*penalize curvature itself*.

On a **kink**, $\bar v_i$ averages across the discontinuity, $v_i$ disagrees
sharply, $L_C$ fires. So $L_C$ separates "smooth curvature is fine" from
"abrupt direction change is bad" — exactly what we want for jaggedness
reduction.

**Sign alignment.** Eigenvectors are only defined up to sign; we orient
each $v_j$ so its dot product with $v_i$ is non-negative before averaging.
Without this, two cells with identical surface normal but opposite eigvec
sign would cancel in the mean.

---

## 7. Proposal D — midpoint coplanarity

A loss on *positions*, not directions. For each cell $i$ the locus of its
boundary faces is approximately the set of midpoints
$m_{ij} = (p_i + p_j)/2$ over high-jump neighbors. If the local iso-surface
is flat, those midpoints lie in a plane.

Weighted mean and covariance over high-jump edges (weight $w_{ij}$, the
same as in $M_i$):

$$
\bar m_i \;=\; \frac{\sum_j w_{ij}\, m_{ij}}{\sum_j w_{ij}}, \qquad
C_i \;=\; \sum_j w_{ij}\, (m_{ij} - \bar m_i)(m_{ij} - \bar m_i)^{\!\top}
$$

Loss = relative off-plane energy:

$$
\boxed{\;
L_D \;=\; \frac{1}{|V|}\sum_i m_i \cdot \frac{\lambda_3(C_i)}{\operatorname{tr}(C_i)}
\;}
$$

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 225" width="100%" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12">
  <style>
    .cell    { fill: #1a1a1a; }
    .nbr_hi  { fill: #fff0f0; stroke: #dc2626; stroke-width: 1.5; }
    .edge_hi { stroke: #dc2626; stroke-width: 1.5; stroke-dasharray: 3,2; }
    .mid     { fill: #1d4ed8; }
    .plane   { stroke: #15803d; stroke-width: 1.5; fill: none; stroke-dasharray: 5,3; }
    .resid   { stroke: #dc2626; stroke-width: 1.5; }
    .label   { fill: #1a1a1a; }
    .good    { fill: #15803d; }
    .bad     { fill: #dc2626; }
    @media (prefers-color-scheme: dark) {
      .cell   { fill: #e4e4e7; }
      .nbr_hi { fill: #2d1515; stroke: #f87171; }
      .edge_hi{ stroke: #f87171; }
      .mid    { fill: #60a5fa; }
      .plane  { stroke: #4ade80; }
      .resid  { stroke: #f87171; }
      .label  { fill: #e4e4e7; }
      .good   { fill: #4ade80; }
      .bad    { fill: #f87171; }
    }
  </style>

  <!-- Left: coplanar — low L_D -->
  <g transform="translate(40,20)">
    <text x="140" y="0" text-anchor="middle" class="label" font-weight="600">Boundary midpoints coplanar</text>
    <circle class="cell" cx="140" cy="168" r="5"/>
    <circle class="nbr_hi" cx="40"  cy="60" r="4"/>
    <circle class="nbr_hi" cx="100" cy="50" r="4"/>
    <circle class="nbr_hi" cx="160" cy="55" r="4"/>
    <circle class="nbr_hi" cx="220" cy="65" r="4"/>
    <circle class="nbr_hi" cx="265" cy="82" r="4"/>
    <line class="edge_hi" x1="140" y1="168" x2="40"  y2="60"/>
    <line class="edge_hi" x1="140" y1="168" x2="100" y2="50"/>
    <line class="edge_hi" x1="140" y1="168" x2="160" y2="55"/>
    <line class="edge_hi" x1="140" y1="168" x2="220" y2="65"/>
    <line class="edge_hi" x1="140" y1="168" x2="265" y2="82"/>
    <circle class="mid" cx="90"  cy="114" r="3"/>
    <circle class="mid" cx="120" cy="109" r="3"/>
    <circle class="mid" cx="150" cy="111" r="3"/>
    <circle class="mid" cx="180" cy="116" r="3"/>
    <circle class="mid" cx="202" cy="125" r="3"/>
    <line class="plane" x1="62" y1="117" x2="225" y2="117"/>
    <text x="140" y="200" text-anchor="middle" class="good" font-size="11">λ₃(Cᵢ) ≈ 0: small off-plane residuals</text>
  </g>

  <!-- Right: scattered — high L_D -->
  <g transform="translate(400,20)">
    <text x="140" y="0" text-anchor="middle" class="label" font-weight="600">Boundary midpoints not coplanar</text>
    <circle class="cell" cx="140" cy="168" r="5"/>
    <circle class="nbr_hi" cx="50"  cy="100" r="4"/>
    <circle class="nbr_hi" cx="110" cy="50"  r="4"/>
    <circle class="nbr_hi" cx="155" cy="35"  r="4"/>
    <circle class="nbr_hi" cx="220" cy="90"  r="4"/>
    <circle class="nbr_hi" cx="265" cy="40"  r="4"/>
    <line class="edge_hi" x1="140" y1="168" x2="50"  y2="100"/>
    <line class="edge_hi" x1="140" y1="168" x2="110" y2="50"/>
    <line class="edge_hi" x1="140" y1="168" x2="155" y2="35"/>
    <line class="edge_hi" x1="140" y1="168" x2="220" y2="90"/>
    <line class="edge_hi" x1="140" y1="168" x2="265" y2="40"/>
    <circle class="mid" cx="95"  cy="134" r="3"/>
    <circle class="mid" cx="125" cy="109" r="3"/>
    <circle class="mid" cx="147" cy="101" r="3"/>
    <circle class="mid" cx="180" cy="129" r="3"/>
    <circle class="mid" cx="202" cy="104" r="3"/>
    <line class="plane" x1="82" y1="118" x2="218" y2="118"/>
    <line class="resid" x1="95"  y1="118" x2="95"  y2="134"/>
    <line class="resid" x1="125" y1="118" x2="125" y2="109"/>
    <line class="resid" x1="147" y1="118" x2="147" y2="101"/>
    <line class="resid" x1="180" y1="118" x2="180" y2="129"/>
    <line class="resid" x1="202" y1="118" x2="202" y2="104"/>
    <text x="140" y="200" text-anchor="middle" class="bad" font-size="11">λ₃(Cᵢ) &gt; 0: midpoints don't lie on a plane</text>
  </g>
</svg>

**Compared to B.** Mathematically the same shape ($\lambda_3 / \mathrm{tr}$
of a 3×3 covariance), but $C_i$ uses *positions* of midpoints,
$M_i$ uses *unit directions*. Consequences:

- A neighbor twice as far contributes 4× more to $C_i$ but the same as
  anyone else to $M_i$. So D is sensitive to sliver geometry (one far
  neighbor dominates), B is not.
- D's gradient flows straight into $p_i$ through the midpoint
  $m_{ij} = (p_i + p_j)/2$; geometrically the loss says "the surface I'm
  carving should be locally flat."

**Compared to A / BA / C.** No reliance on eigenvectors of $M_i$ (it has
its own eigendecomp on $C_i$). Lives in position-space.

---

## 8. Side-by-side summary

| | Operates on | Pairwise / Intrinsic | Direction-only? | Allows smooth curvature? | Cost / extras |
|---|---|---|---|---|---|
| **BA (existing)** | $\hat M$ tensor | pairwise | no — also magnitude + lower eigvecs | no | one $M$ build + Frobenius |
| **A** | top eigvec $v_i$ | pairwise | yes | no | + 1 `eigh` per cell |
| **B** | $\lambda_3 / \mathrm{tr}(M_i)$ | intrinsic (per cell) | yes (planarity) | n/a | + 1 `eigh`; $\det$ variant avoids it |
| **C** | top eigvec $v_i$, Laplacian | neighborhood mean | yes | **yes** | + 1 `eigh` + sign-align + scatter mean |
| **D** | midpoint covariance $C_i$ | intrinsic (per cell) | yes (planarity) | n/a | new $C_i$ build + `eigh` |

### When each fires

| Scenario | BA | A | B | C | D |
|---|---|---|---|---|---|
| Flat edge, perfectly aligned cells | 0 | 0 | 0 | 0 | 0 |
| Flat edge, identical normals, different anisotropy magnitude | **>0** | 0 | 0 | 0 | 0 |
| Smoothly curved iso-surface | **>0** | **>0** | small | **0** | small |
| Kink between two flat patches | **>0** | **>0** | small per cell, large at corner cell | **>0** | large at corner cell |
| Sliver cell with one far neighbor | depends on $M_i$ rank | depends | small ($M_i$ still rank-1) | depends | **>0** (geometric flatness violated) |

---

## 9. CVT (separately, targeting slivers not jaggedness)

For completeness — the centroidal Voronoi pull targets cell *shape*, not
boundary structure.

$$
L_{\text{CVT}} \;=\; \frac{1}{|V|}\sum_i \frac{1}{r_i^{2}}
\left\|\, p_i - \frac{1}{|N(i)|}\sum_{j \in N(i)} p_j \right\|^2
$$

This is Lloyd's-step energy with $1/r_i^2$ scale normalization (so large
empty-region cells don't dominate). Drives each point toward the centroid
of its neighbors → eliminates slivers, regularizes the triangulation.
Orthogonal axis from BA/A/B/C/D: those operate on the *density-derived*
boundary structure; CVT operates on *geometry only*.

---

## 10. Notation cheat sheet

| Symbol | Meaning |
|---|---|
| $p_i, \mu_i, r_i$ | position, density, cell radius |
| $N(i)$ | Voronoi neighbors of cell $i$ |
| $\Delta\mu_{ij}$ | $\mu_j - \mu_i$ (only "gradient" signal we have) |
| $n_{ij}$ | unit direction $p_i \to p_j$ |
| $w_{ij}$ | boundary weight $(\Delta\mu_{ij})^2 r_i r_j$ |
| $s_{ij}$ | same-density gate $\exp(-(\Delta\mu_{ij})^2 / \sigma_v^2)$ |
| $m_i, m_{ij}$ | validity mask (cell/edge), midpoint position |
| $M_i$ | $\sum_j w_{ij} n_{ij} n_{ij}^\top$ — 3×3 boundary tensor |
| $\hat M_i$ | $M_i / \operatorname{tr}(M_i)$ — normalized |
| $v_i, \lambda_k$ | top eigvec / eigenvalues of $M_i$ |
| $C_i$ | midpoint covariance for D |
