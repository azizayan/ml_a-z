
## Machine Learning Learning Exercises

This repository documents a series of machine learning experiments conducted as part of a structured, hands-on learning process. Each submodule focuses on the "why" behind the "how," covering data preprocessing, model behavior, and identifying common pitfalls like data leakage and target imbalance.

---

## 📊 Additional Project: Pumpkin Price Prediction

*An investigation into US pumpkin market data and the limitations of linear regression in imbalanced target spaces.*

### 🛠 Methodology

We focused on **bushel carton packages** to maintain pricing consistency across the dataset.

* **Feature Set:** `Package` (Categorical), `Origin` (Categorical), `Month` (Numerical)
* **Target:** `Average Price`
* **Pipeline:** * **Preprocessing:** `One-Hot Encoding` via `ColumnTransformer`.
* **Model:** `Linear Regression` (Baseline).

### 📈 Evaluation & Results

The model was evaluated against a **mean predictor** baseline using  and Mean Absolute Error (MAE).

| Metric | Observation |
| --- | --- |
| **Score** | Relatively high, driven by data density. |
| **MAE** | Low in the $15–$20 range; high for outliers. |
| **Distribution** | Strong performance in the dominant low-price region. |

> [!IMPORTANT]
> **Stress Testing Results:** When tested against synthetic high-price scenarios, the model systematically underestimated values, revealing a clear **regression-to-the-mean** behavior.

---

## 🧠 Key Analytical Insights

* **The Metric Trap:** High global performance metrics (like ) can obscure significant weaknesses in underrepresented regions of the target space.
* **Extrapolation Limits:** Linear regression, especially with additive categorical encodings, has limited capacity to predict values outside the dense training "clusters."
* **Stress Testing:** Manually constructing synthetic test cases is essential for revealing how a model behaves under edge-case conditions.

---
