# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Dasi Nithin
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/dasinithin/flyrank-ml-internship-starter
- **Date:** October 2026

---

## 0. Abstract

How should central content and SEO teams allocate limited editorial review bandwidth across thousands of decaying pages without wasting human hours on noisy, mean-reverting drops? We analyzed 30,000 pseudonymized web pages from the FlyRank Search Intelligence dataset across 44 observable search, crawl, and engagement metrics over a 90-day baseline window. We formulated the task as an early-warning decline scoring problem evaluated strictly under an honest client-holdout split (27,675 train / 2,325 blind test) to prevent client memorization and domain leakage. While a transparent heuristic rule achieved a Precision@50 of 0.240 and an ROC-AUC of 0.627, a calibrated Random Forest classifier reached a Precision@50 of 0.680 (a 2.83× lift) and an ROC-AUC of 0.747, with top predictive signals driven by search presence consistency (`days_with_impressions`, 16.1%), aggregate demand (`log_impressions_90d`, 12.8%), and position tier. The resulting system transforms model probabilities into an actionable queue categorized into five explicit editorial playbooks (`refresh_and_review_ctr`, `refresh`, `refresh_and_review_engagement`, `expand_and_refresh`, and `monitor`) with interpretable reason codes, enabling editorial teams to prioritize the top 50 highest-impact pages each review cycle with high confidence while strictly guarding against automated rewrites.

---

## 1. Problem Framing

### Decision Context
In organic search management, large multi-domain portfolios accumulate thousands of published URLs. Over time, content undergoes decay due to shifting search intent, competitor content releases, algorithm updates, and stale information. However, editorial and content optimization bandwidth is strictly constrained: in a typical organization or client engagement, an editorial team can realistically review and refresh approximately **50 priority pages per monthly cycle**.

### The Core Problem
The decision this work improves is: **Of all candidate pages across the portfolio, which 50 pages should be routed to human editors first for remediation, and what specific action should be taken?**

### Unit of Analysis & Outputs
- **Unit of Analysis:** One row = one unique content asset (`content_id`) on a specific client domain (`client_id`), evaluated over a 90-day observation window.
- **Output:** A calibrated decay probability score $P(\text{decline}|X) \in [0, 1]$, an action assignment (`refresh_and_review_ctr`, `refresh`, `refresh_and_review_engagement`, `expand_and_refresh`, `monitor`), and a human-readable list of reason codes explaining why the page was flagged.

### Cost of Misclassification
- **False Positive (Flagging a healthy or transiently volatile page):** Wastes 2–4 hours of expensive subject-matter expert editorial time attempting to "fix" a page that would have naturally stabilized, potentially disrupting working content that already satisfies search intent.
- **False Negative (Missing a high-value decaying page):** Allows key traffic- and revenue-generating landing pages to silently slide off Page 1 of Search Engine Results Pages (SERPs), resulting in compound organic traffic loss that takes months to recover.

### Why Machine Learning Outperforms Fixed If-Else Rules
Static rules (e.g., "flag if clicks dropped >20%") fail because of search query volatility and mean reversion. In our empirical signal audit, pages on a short-term dip reverted to their mean 67% of the time, causing naive rules to underperform the random baseline. Machine learning models capture non-linear interactions between historical impression consistency, position tier, CTR deviations from SERP expectations, and on-page engagement signals, providing a robust, repeatable prioritization queue.

---

## 2. Data Safety & Leakage Prevention

### Data Source & Inventory
This work utilizes the FlyRank Applied Search Intelligence dataset (`data/raw/content_refresh_anonymized.csv`), containing **30,000 pseudonymized content items** across 70 client entities with 44 tabular columns representing Google Search Console (GSC) and Google Analytics 4 (GA4) metrics over a 90-day window.

### Public-Safety Protocol
In strict adherence to the FlyRank public data policy:
- No raw client names, customer domains, page URLs, page titles, or raw user queries are included.
- All identifiers (`content_id`, `client_id`) are cryptographically salted pseudonymized tokens.
- All numbers and evaluation metrics represent aggregate, observed search patterns. No reverse-engineering of Google's ranking algorithm or causal refresh claims are made.

### Deliberate Feature Exclusions
1. **Target-Derived Columns:** `trend_direction`, `trend_pct`, and future-window traffic deltas were strictly excluded from the feature matrix to eliminate target leakage.
2. **Identification Tokens:** `content_id` and `client_id` were omitted from feature inputs to prevent model memorization of domain-specific baselines.
3. **Future Window Aggregations:** Features were strictly computed over the retrospective 90-day observation window prior to the decision timestamp.

---

## 3. Baseline Model & The Anti-Baseline

### Baseline Heuristic Specification
To establish an honest benchmark, we implemented a transparent heuristic scoring formula reflecting industry standard SEO audit practices:

$$\text{Baseline Score} = 0.35 \times \text{norm}(\log(\text{impressions}_{90d})) + 0.30 \times \text{norm}(\text{content\_age\_days}) + 0.20 \times \mathbb{I}(\text{trend\_pct} < 0) + 0.15 \times \text{norm}(\text{days\_since\_update})$$

The baseline ranks pages by this composite index and assigns heuristic reason codes (`declining_with_demand`, `high_impressions_stale`, etc.).

### Benchmark Performance
When evaluated on the held-out test client partition:
- **Baseline Accuracy:** 0.609
- **Baseline ROC-AUC:** 0.627
- **Baseline Average Precision (PR-AUC):** 0.468
- **Baseline Precision@20:** 0.150
- **Baseline Precision@50:** 0.240
- **Baseline Precision@100:** 0.360

### The "Anti-Baseline" Discovery
Crucially, evaluating a naive "biggest recent drop" rule demonstrated that pages exhibiting sharp short-term declines had a forward decline rate of only 33% (below the portfolio base rate of 54.2%). Sorting by raw recent drops actively selects for mean-reverting temporary fluctuations, confirming why transparent machine learning validation is vital before deploying editorial resources.

---

## 4. Modeling & Methodology

### Label Definition
The ground-truth classification target `is_declining_label` is defined as:
$$\text{is\_declining\_label} = \begin{cases} 1 & \text{if forward trend direction} = \text{'down'} \\ 0 & \text{otherwise} \end{cases}$$
The dataset has an observed portfolio base rate of **54.2% positive declining items** (16,262 out of 30,000 rows).

### Feature Engineering
A 26-dimensional feature vector was engineered from the 90-day observational data:
1. **Log-Scale Search Volume & Traffic:** `log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`, `log_ai_sessions_90d`
2. **Presence & Consistency Signals:** `days_with_impressions`, `days_with_sessions`, `content_age_days`, `days_since_last_update`
3. **SERP Performance & User Engagement:** `avg_position`, `ctr`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`
4. **Market Context & Commercial Intent:** `search_volume`, `competition`, `cpc`, `word_count`, `char_count`
5. **Categorical Tiers (One-Hot Encoded):** `competition_level`, `content_type`, `main_intent`, `age_tier`, `freshness_tier`, `word_count_tier`, `impression_tier`, `position_tier`

Missing numerical values were median-imputed, extreme outliers clipped, and categorical unknowns encoded into an explicit category.

### Validation Architecture: Client Holdout Split
Rather than a random train/test split (which leaks domain authority and client-level ranking distributions), we implemented a **Grouped Client Holdout Split**:
- **Training Set:** 27,675 rows across 9 client domains.
- **Test Holdout Set:** 2,325 rows across a completely unseen client (`client_03`).
This guarantees that the model cannot achieve artificially inflated metrics by memorizing domain-level idiosyncrasies.

---

## 5. Empirical Evaluation

### Model Comparison Table (Client Holdout Split)

| Model Architecture | Precision@20 | Precision@50 | Precision@100 | ROC-AUC | PR-AUC (Avg Prec) | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Heuristic Baseline** | 0.150 | 0.240 | 0.360 | 0.627 | 0.468 | — | — |
| **Logistic Regression (L2)** | 0.350 | 0.400 | 0.440 | 0.700 | 0.522 | 0.567 | 0.566 |
| **Decision Tree (max_depth=6)** | 0.550 | 0.620 | 0.600 | 0.742 | 0.575 | 0.716 | 0.634 |
| **Random Forest (100 trees, selected)** | **0.700** | **0.680** | **0.700** | **0.747** | **0.610** | **0.741** | **0.638** |

### Key Findings
1. **Dramatic Lift at Review Capacity:** At the primary operational constraint of $K=50$ pages, Random Forest delivers **0.680 precision vs. 0.240 for the heuristic baseline**—a **2.83× efficiency improvement**.
2. **High Precision at the Head ($K=20$):** In the top 20 queue rows, Random Forest achieves **0.700 vs. 0.150** (a **4.67× lift**).
3. **Discriminative Capability:** ROC-AUC improves from 0.627 to 0.747, and Average Precision increases from 0.468 to 0.610 on completely unseen client domains.

### Error Analysis
- **False Positives (High Model Score, Did Not Decline):** Concentrated in pages with strong historical authority (`days_with_impressions` > 85) experiencing temporary competitive testing. Mitigated by routing pages with rising 60-day momentum into a `monitor` status.
- **False Negatives (Low Model Score, Subsequently Declined):** Pages with low historical search presence (< 20 days with impressions) that suffered catastrophic single-keyword drops. These reflect black-swan SERP restructuring that cannot be detected from historical signals alone.

---

## 6. Interpretation & Signal Importance

### Top 10 Feature Importances (Gini Impurity)

1. `days_with_impressions` (16.06%): Measures indexing stability. Pages losing daily impression continuity exhibit high risk of irreversible decay.
2. `log_impressions_90d` (12.85%): Scale of total search demand. High-volume pages show more reliable decay patterns.
3. `avg_position` (10.84%): SERP ranking tier. Pages hovering on the Page 1/Page 2 boundary (positions 8–15) are most susceptible to traffic drop-offs.
4. `content_age_days` (9.50%): Content aging. Assets older than 180 days show accelerating decay unless refreshed.
5. `word_count` (4.12%) & `char_count` (3.98%): Content depth correlates inversely with thin-content vulnerability.
6. `ctr` (3.32%): Under-performing CTR relative to position indicates snippet/intent misalignment.
7. `log_clicks_90d` (3.25%): Realized organic capture.
8. `scroll_rate` (3.09%): On-page engagement and content satisfaction.
9. `days_with_sessions` (2.99%): GA4 session regularity.
10. `position_tier_top_3` (2.47%): Premium ranking vulnerability.

---

## 7. Action Playbook & Editorial Recommendations

### The 5 Operational Action Codes
The model probability score is integrated with search visibility metrics to generate five distinct editorial actions:

1. **`refresh_and_review_ctr` (22.2% of queue / 6,655 items):**
   - *Criteria:* Model decline risk high + page possesses high SERP impressions but below-benchmark CTR.
   - *Prescribed Action:* Rewrite `<title>` tags and meta descriptions, test rich snippet schema, and align snippet messaging with search intent.
2. **`refresh` (27.4% of queue / 8,207 items):**
   - *Criteria:* Model decline risk high on established, aging content with balanced metrics.
   - *Prescribed Action:* Update outdated facts/dates, refresh body copy, add updated internal links, and re-promote.
3. **`refresh_and_review_engagement` (6.6% of queue / 1,987 items):**
   - *Criteria:* Model decline risk high + CTR is acceptable but scroll rate or engagement duration is low.
   - *Prescribed Action:* Restructure page layout, add visual media, improve readability, and eliminate intrusive elements.
4. **`expand_and_refresh` (0.3% of queue / 82 items):**
   - *Criteria:* Model decline risk high + high competition level + short word count.
   - *Prescribed Action:* Expand into comprehensive topic cluster pillar page.
5. **`monitor` (43.6% of queue / 13,069 items):**
   - *Criteria:* Low decline risk or stable performance.
   - *Prescribed Action:* Do not touch. Protect against unnecessary edits that risk disrupting stable rankings.

### Human-in-the-Loop & Automation Guardrails
- **Strictly No Autonomous Publishing:** Machine learning provides prioritization and diagnosis; human subject-matter experts must execute editorial changes.
- **Eligibility Floors:** Pages with < 100 90-day impressions or < 90 days of age are excluded from active refresh queues to prevent chasing noise.

---

## 8. Reproducibility & Research Receipts

### Environment & Execution
- **Python Version:** 3.11.11 / 3.13.2
- **Key Libraries:** `scikit-learn>=1.4.0`, `pandas>=2.2.0`, `numpy>=1.26.0`, `duckdb>=1.0.0`
- **Global Seed:** Fixed to `42` across all scripts and notebooks (`scripts/03_train_model.py`, `work/notebooks/capstone.ipynb`).
- **Reproduction Command:**
  ```bash
  git clone https://github.com/dasinithin/flyrank-ml-internship-starter.git
  cd flyrank-ml-internship-starter
  pip install -r requirements.txt
  python scripts/run_all.py
  ```

### Verification Artifacts
- Model Results Receipt: `outputs/model_results.json`
- Feature Metadata: `data/processed/feature_metadata.json`
- Scored Queue: `outputs/refresh_queue.csv`
- Generated Vector Charts: `outputs/charts/top_feature_importance.svg`, `outputs/charts/action_mix.svg`, `outputs/charts/trend_distribution.svg`

---

## 9. Acknowledgments & Data Credit

This research was conducted as part of the FlyRank Applied Search Intelligence Track. 

**Built on the FlyRank ML Internship dataset** — [https://flyrank.ai](https://flyrank.ai).
