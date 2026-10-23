"""
Runs every experiment in the thesis, in order, then builds the figures.
Each step can also be run on its own, e.g.  python -m experiments.exp1_scaling
Results (CSV) go to results/, figures to results/figures/.
"""

from experiments import (exp1_scaling, exp2_mechanism, exp3_structural_profile,
                         exp4_skewness_fixed_separation, exp5_separability_sweep,
                         make_figures)

if __name__ == "__main__":
    print("\n=== 1. Replication and scaling test (Sections 5.1-5.3) ===")
    exp1_scaling.run()
    print("\n=== 2. Mechanism: scale sweep and k sweep (Section 5.4) ===")
    exp2_mechanism.run()
    print("\n=== 3. Structural profile and KNN / NCN / NC (Table 4.1, Section 5.5) ===")
    exp3_structural_profile.run()
    print("\n=== 4. Skewness with separation held fixed (Section 5.6) ===")
    exp4_skewness_fixed_separation.run()
    print("\n=== 5. Separability sweep (Section 5.7) ===")
    exp5_separability_sweep.run()
    print("\n=== Figures ===")
    make_figures.run()
