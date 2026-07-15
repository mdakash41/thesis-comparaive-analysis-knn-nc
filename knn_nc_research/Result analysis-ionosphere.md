# Dataset Analysis Ionosphere

1. Dataset summary
2. Statistical analysis
3. NC suitability assessment
4. Overall bias score (optional)
5. Conclusion

---

## Dataset: Ionosphere
> Definition 1 (NC-favourable class). In this study, a class is termed NC-favourable if it satisfies the threshold defined for the corresponding statistical property (e.g., ≥70% unimodal features, ≥60% low-skewness features, etc.). The term does not imply that NC will necessarily outperform KNN; rather, it indicates that the class exhibits characteristics consistent with the assumptions of the Nearest Centroid classifier.

### 1. Dataset Overview

| Property | Value |
|----------|-------|
| Dataset | Ionosphere |
| Samples | 351 |
| Features | 34 |
| Classes | 2 |
| Samples per class | b: 126, g: 225 |

### 2. Statistical Analysis

#### 2.1 Multivariate Normality (Henze-Zirkler)

| Class | HZ Statistic | p-value | Normal Distribution |
|-------|--------------|---------|---------------------|
| b | 504.0 | 0.0 | ✗ No |
| g | 900.0 | 0.0 | ✗ No |

**Interpretation**

Neither class satisfies multivariate normality. Both classes show extremely high HZ statistics, indicating severe deviation from the multivariate normal distribution. The dataset does not conform to the centroid assumption used by Nearest Centroid.

#### 2.2 Unimodality (Dip Test)
> Note. Nearest Centroid (NC) assumes that each class can be reasonably represented by a single centroid. To evaluate this assumption, the percentage of unimodal features within each class is calculated using Hartigan's Dip Test (α = 0.05). A class is considered NC-favourable when 70% or more of its features are unimodal; otherwise, it is considered to exhibit multimodal structure that may reduce centroid representativeness.

| Class | Unimodal Features | Interpretation |
|-------|-------------------|----------------|
| b | 5.9% | Only 6% of features are unimodal — multimodal structure detected. |
| g | 97.1% | 97% of features are unimodal — NC centroid is structurally valid. |

**Interpretation**

The two classes show a stark contrast in unimodality. Class g strongly satisfies the NC assumption with 97.1% unimodal features, indicating excellent centroid representativeness. However, class b is severely multimodal with only 5.9% unimodal features, meaning a single centroid is unlikely to adequately represent this class. This asymmetry in class structure may create an imbalance in NC's ability to model the two classes.

#### 2.3 Feature Skewness
> Note. NC represents each class using its arithmetic mean (centroid). Highly skewed feature distributions can shift the mean away from the region containing most observations, making the centroid less representative. For each class, the percentage of features whose absolute skewness is below 0.5 is calculated. A class is considered NC-favourable when 60% or more of its features have low skewness; otherwise, asymmetric distributions are considered likely to reduce centroid reliability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| b | 94.1% | 94% of features have low skewness — data is NC-favourable. |
| g | 61.8% | 62% of features have low skewness — data is NC-favourable. |

**Interpretation**

Both classes satisfy the NC-favourable threshold for low skewness. Class b exhibits exceptionally high symmetry (94.1%), while class g meets the threshold more modestly at 61.8%. These results indicate that centroids are unlikely to be substantially distorted by asymmetric feature distributions in either class.

#### 2.4 Kurtosis
> Note. NC performs best when feature distributions are not dominated by heavy tails or extreme outliers, as such observations can distort the centroid. For each class, the percentage of features whose kurtosis falls within the predefined near-normal range is calculated. A class is considered NC-favourable when 60% or more of its features exhibit near-normal kurtosis; otherwise, heavy-tailed distributions are considered likely to reduce centroid stability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| b | 97.1% | 97% of features have near-normal kurtosis — tail behaviour is NC-favourable. |
| g | 91.2% | 91% of features have near-normal kurtosis — tail behaviour is NC-favourable. |

**Interpretation**

Both classes strongly satisfy the NC-favourable threshold for near-normal kurtosis, with exceptionally high percentages (97.1% and 91.2%). This indicates that tail behaviour is well-behaved in both classes and extreme observations are unlikely to exert disproportionate influence on centroid estimation.

#### 2.5 Class Separability
> Note. NC assigns a sample to the nearest class centroid. Therefore, its performance depends on the distance between class centroids relative to the spread of each class. The separability ratio is defined as the inter-class centroid distance divided by the average within-class dispersion. Larger ratios indicate better-separated classes and are therefore more favourable for NC. The following interpretation is used throughout this study:

- Ratio < 1.0: Low separability (NC-unfavourable)
- 1.0 ≤ Ratio < 1.5: Moderate separability
- Ratio ≥ 1.5: High separability (NC-favourable)

| Class Pair | Ratio | Separation |
|------------|-------|-------------|
| b vs g | 1.363 | MODERATE — Moderate separation; NC likely works but is not conclusive. |

**Average Separability Ratio**

1.363

**Interpretation**

The single class pair exhibits moderate separability, falling short of the high separability threshold (1.5). While the classes are not poorly separated, the moderate ratio suggests some overlap in the feature space, which may reduce the reliability of centroid-based discrimination.

### 3. Overall Dataset Assessment

| Criterion | Result | Supports NC? |
|-----------|--------|--------------|
| Multivariate normality | Not satisfied | ✗ |
| Unimodality | Mixed — Class g only | ~ |
| Low skewness | Satisfied | ✓ |
| Near-normal kurtosis | Strongly satisfied | ✓ |
| High separability | Moderate only | ~ |

### 4. Conclusion

**Assessment**

The Ionosphere dataset presents a notably asymmetric profile for centroid-based classification. Classes b and g exhibit fundamentally different structural properties: class g is overwhelmingly unimodal (97.1%), while class b is almost entirely multimodal (5.9%). Both classes perform well on skewness and kurtosis, and class separability is moderate. The severe multimodality in class b represents a significant violation of NC assumptions, as a single centroid cannot adequately capture a class with such fragmented feature distributions.

Therefore, observing similar performance between KNN and NC on this dataset should not be interpreted as evidence that both algorithms perform similarly in general. The extreme structural asymmetry between the two classes suggests that NC may perform adequately for class g but struggle with class b, creating an inherent performance ceiling not present in KNN. Any comparable overall performance would reflect dataset-specific properties rather than algorithmic equivalence.