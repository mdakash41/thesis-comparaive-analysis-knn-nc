import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import lognorm, skew, kurtosis
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── colour palette ────────────────────────────────────────────────────────────
C0, C1 = '#4C72B0', '#DD8452'   # class 0 = blue, class 1 = orange

# ═════════════════════════════════════════════════════════════════════════════
# HELPER: print measured properties for one dataset
# ═════════════════════════════════════════════════════════════════════════════
def print_properties(X, y, label):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    for c in np.unique(y):
        Xc = X[y == c]
        sk  = np.mean([abs(skew(Xc[:, f]))    for f in range(Xc.shape[1]) if Xc[:,f].std()>0])
        ku  = np.mean([abs(kurtosis(Xc[:, f])) for f in range(Xc.shape[1]) if Xc[:,f].std()>0])
        print(f"  class {c}: skewness={sk:.3f}  kurtosis={ku:.3f}  std={Xc.std():.3f}")

    # separability: centroid distance / within-class std
    Xs = StandardScaler().fit_transform(X)
    c0, c1 = Xs[y==0].mean(0), Xs[y==1].mean(0)
    sep = np.linalg.norm(c0 - c1) / np.mean([Xs[y==c].std() for c in [0,1]])
    print(f"  separability (centroid-dist / within-std) = {sep:.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — SKEWNESS VERIFICATION
# Shows 3 sigma levels: low / medium / high
# For each: histogram per class (feature 0) + PCA scatter
# ═════════════════════════════════════════════════════════════════════════════
def verify_skewness():
    configs = [
        (0.1, 'Low skewness  (σ=0.1)'),
        (1.0, 'Medium skewness  (σ=1.0)'),
        (2.5, 'High skewness  (σ=2.5)'),
    ]
    n = 400

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Skewness Experiment — Dataset Verification', fontsize=14, fontweight='bold')

    for col, (sigma, title) in enumerate(configs):
        X0 = lognorm.rvs(s=sigma, size=(n//2, 4), random_state=42)
        X1 = lognorm.rvs(s=sigma, size=(n//2, 4), random_state=43)
        X1 += 1.5 * sigma
        X  = np.vstack([X0, X1])
        y  = np.array([0]*(n//2) + [1]*(n//2))
        print_properties(X, y, title)

        # row 0: histogram of feature_0 per class
        ax = axes[0, col]
        ax.hist(X0[:, 0], bins=30, alpha=0.6, color=C0, label='class 0', density=True)
        ax.hist(X1[:, 0], bins=30, alpha=0.6, color=C1, label='class 1', density=True)
        sk0 = abs(skew(X0[:, 0]))
        sk1 = abs(skew(X1[:, 0]))
        ax.set_title(f'{title}\nskew(c0)={sk0:.2f}  skew(c1)={sk1:.2f}', fontsize=10)
        ax.set_xlabel('Feature 0 value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

        # row 1: PCA scatter (2D projection)
        ax = axes[1, col]
        Xp = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
        ax.scatter(Xp[y==0,0], Xp[y==0,1], alpha=0.35, s=15, color=C0, label='class 0')
        ax.scatter(Xp[y==1,0], Xp[y==1,1], alpha=0.35, s=15, color=C1, label='class 1')
        ax.set_title('PCA projection (2D)', fontsize=10)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('./verify_skewness.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("\n✓ verify_skewness.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — SEPARABILITY VERIFICATION
# Shows 3 distance levels: low / medium / high
# For each: PCA scatter + class overlap histogram
# ═════════════════════════════════════════════════════════════════════════════
def verify_separability():
    configs = [
        (0.5,  'Low separability  (d=0.5)'),
        (2.0,  'Medium separability  (d=2.0)'),
        (4.0,  'High separability  (d=4.0)'),
    ]
    n = 400

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Separability Experiment — Dataset Verification', fontsize=14, fontweight='bold')

    for col, (dist, title) in enumerate(configs):
        centres = [[0,0,0,0], [dist,0,0,0]]
        X, y    = make_blobs(n_samples=n, n_features=4,
                             centers=centres, cluster_std=1.0, random_state=42)
        print_properties(X, y, title)

        # row 0: PCA scatter
        ax  = axes[0, col]
        Xp  = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
        ax.scatter(Xp[y==0,0], Xp[y==0,1], alpha=0.35, s=15, color=C0, label='class 0')
        ax.scatter(Xp[y==1,0], Xp[y==1,1], alpha=0.35, s=15, color=C1, label='class 1')
        ax.set_title(f'{title}', fontsize=10)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.legend(fontsize=8)

        # row 1: distribution of PC1 scores — shows overlap directly
        ax = axes[1, col]
        ax.hist(Xp[y==0, 0], bins=25, alpha=0.6, color=C0,
                label='class 0', density=True)
        ax.hist(Xp[y==1, 0], bins=25, alpha=0.6, color=C1,
                label='class 1', density=True)
        # measure and annotate overlap
        Xs = StandardScaler().fit_transform(X)
        sep_val = np.linalg.norm(Xs[y==0].mean(0) - Xs[y==1].mean(0)) / \
                  np.mean([Xs[y==c].std() for c in [0,1]])
        ax.set_title(f'PC1 distribution\nseparability={sep_val:.2f}', fontsize=10)
        ax.set_xlabel('PC1 score')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('./verify_separability.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("✓ verify_separability.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — MODALITY VERIFICATION
# Shows 1 / 3 / 6 modes per class
# For each: PCA scatter coloured by mode + class-level density
# ═════════════════════════════════════════════════════════════════════════════
def verify_modality():
    configs = [
        (1, 'Unimodal  (1 mode)'),
        (3, 'Trimodal  (3 modes)'),
        (6, 'Hexamodal  (6 modes)'),
    ]
    n = 400

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Modality Experiment — Dataset Verification', fontsize=14, fontweight='bold')

    for col, (n_modes, title) in enumerate(configs):
        rng = np.random.default_rng(42)
        X_parts, y_parts, mode_labels = [], [], []

        for cls in range(2):
            nm       = n // 2 // n_modes
            centres  = [[cls*2.0, (m - n_modes/2)*3.5, 0, 0]
                        for m in range(n_modes)]
            for m, c in enumerate(centres):
                chunk = rng.normal(loc=c, scale=0.7, size=(nm, 4))
                X_parts.append(chunk)
                y_parts.append(np.full(nm, cls))
                mode_labels.append(np.full(nm, m + cls * n_modes))

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        m_labels = np.concatenate(mode_labels)
        print_properties(X, y, title)

        Xp = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))

        # row 0: PCA scatter by CLASS
        ax = axes[0, col]
        ax.scatter(Xp[y==0,0], Xp[y==0,1], alpha=0.35, s=15, color=C0, label='class 0')
        ax.scatter(Xp[y==1,0], Xp[y==1,1], alpha=0.35, s=15, color=C1, label='class 1')
        ax.set_title(f'{title}\n(coloured by class)', fontsize=10)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.legend(fontsize=8)

        # row 1: PCA scatter by MODE — shows sub-clusters clearly
        ax = axes[1, col]
        cmap = plt.cm.get_cmap('tab10', n_modes * 2)
        for m in range(n_modes * 2):
            mask = m_labels == m
            cls  = 0 if m < n_modes else 1
            ax.scatter(Xp[mask, 0], Xp[mask, 1], alpha=0.5, s=15,
                       color=cmap(m),
                       label=f'c{cls} mode{m % n_modes}' if n_modes <= 3 else None)
        ax.set_title('Coloured by sub-cluster (mode)', fontsize=10)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        if n_modes <= 3:
            ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig('./verify_modality.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("✓ verify_modality.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — SUMMARY HEATMAP
# Shows measured skewness / separability across all experiment levels
# ═════════════════════════════════════════════════════════════════════════════
def verify_summary_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Measured Properties Across Experiment Levels', fontsize=13, fontweight='bold')

    # ── skewness: sigma vs measured skew ─────────────────────────────────────
    sigmas, skew_vals = [], []
    for sigma in [0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 1.8, 2.1, 2.4, 2.8, 3.0]:
        X0 = lognorm.rvs(s=sigma, size=(200, 4), random_state=42)
        X1 = lognorm.rvs(s=sigma, size=(200, 4), random_state=43); X1 += 1.5*sigma
        X  = np.vstack([X0, X1]); y = np.array([0]*200 + [1]*200)
        vals = [abs(skew(X[y==c][:, f]))
                for c in [0,1] for f in range(4)
                if X[y==c][:, f].std() > 0]
        sigmas.append(sigma); skew_vals.append(np.mean(vals))

    ax = axes[0]
    ax.bar(range(len(sigmas)), skew_vals, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f'σ={s}' for s in sigmas], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Measured Mean Absolute Skewness')
    ax.set_title('Skewness: σ parameter → actual skewness measured')
    for i, v in enumerate(skew_vals):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=7)

    # ── separability: distance vs measured sep ────────────────────────────────
    dists, sep_vals = [], []
    for dist in [0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0]:
        X, y = make_blobs(400, n_features=4,
                          centers=[[0,0,0,0],[dist,0,0,0]],
                          cluster_std=1.0, random_state=42)
        Xs  = StandardScaler().fit_transform(X)
        sep = np.linalg.norm(Xs[y==0].mean(0) - Xs[y==1].mean(0)) / \
              np.mean([Xs[y==c].std() for c in [0,1]])
        dists.append(dist); sep_vals.append(round(sep, 3))

    ax = axes[1]
    ax.bar(range(len(dists)), sep_vals, color='darkorange', alpha=0.8)
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels([f'd={d}' for d in dists], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Measured Separability Score')
    ax.set_title('Separability: centre distance → actual separability measured')
    for i, v in enumerate(sep_vals):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig('./verify_summary.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("✓ verify_summary.png saved")


# ─── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    verify_skewness()
    verify_separability()
    verify_modality()
    verify_summary_heatmap()
    print("\nAll verification plots saved.")