[![Companion Workbook in Colab](https://img.shields.io/badge/Colab-Open_Workbook-orange?logo=googlecolab)](https://colab.research.google.com/github/Llander22/Data-Science-Portfolio/blob/main/CS_L7_M10_Hackathon/M10-02-02_covid19_hackathon.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_Pathway.md-blue?logo=github)](https://github.com/Llander22/Data-Science-Portfolio/blob/main/CS_L7_M10_Hackathon/M10-02-01_covid19_hackathon_pathway.md)

# Decision Pathway: COVID-19 Clinical Triage & Population Screening
### Operational Standard: CRISP-DM & Machine Learning Engineering Lifecycle

> **Operational Standard:** Cross-Industry Standard Process for Data Mining (CRISP-DM) & MLOps Engineering Lifecycle  
> **Technical Implementation:** `M10-02-02_covid19_hackathon.ipynb`  
> **Benchmark Reference:** Zoabi, Y., Deri-Rozov, S. & Shomron, N. *Machine learning-based prediction of COVID-19 diagnosis based on symptoms.* **npj Digital Medicine** 4, 3 (2021).  
> **Lifecycle Topology:** Full 6-Stage Architecture with **Selected Path (Green)**, **Rejected Options (Red)**, and **Alternative Branches (Slate)**.

---

### 📖 Narrative Workbook vs. Engineering Lifecycle Crosswalk

* **🔷 Stage 1: Clinical Problem Formulation & Triage**
  * **Workbook Mapping:** `Phase 1: Clinical Problem Formulation & Operational Framing`
  * **Key Decisions & Evidence:** Define binary diagnostic risk (y in {0, 1}) under acute March 2020 RT-PCR scarcity. Mandate non-invasive 30-second self-reporting. Enforce the chronological "Arrow of Time".

* **🔷 Stage 2: Data Ingestion & Quality Auditing**
  * **Workbook Mapping:** `Phase 2: Ingestion & Epidemiological EDA`
  * **Key Decisions & Evidence:** Audit 278,848 tests (51 days). Unmask 94.7% vs 5.3% class imbalance. Expose the "Accuracy Fallacy". Uncover the April 15 regulatory privacy redaction cliff.

* **🔷 Stage 3: Preprocessing & Pipeline Architecture**
  * **Workbook Mapping:** `Phase 2.3–2.4 & Phase 3: Pipeline Architecture`
  * **Key Decisions & Evidence:** Filter 3,892 inconclusive targets and 19k missing gender records (clean cohort n = 255,668). Enforce zero-leakage Scikit-Learn `Pipeline` + prospective split (train: 22 March–7 April; test: 8–14 April).

* **🔷 Stage 4: Modeling & Boosting Benchmarks**
  * **Workbook Mapping:** `Phase 3: Benchmark Replication & Phase 4: XGBoost`
  * **Key Decisions & Evidence:** Replicate *Zoabi et al.*: Decision Tree (0.8357), Random Forest (0.8412), LightGBM (0.8443, PR-AUC 0.5448), and XGBoost (0.8448, PR-AUC 0.5480). Validate via 5-fold CV within training window.

* **🔷 Stage 5: Clinical Explainability & Demographic Fairness**
  * **Workbook Mapping:** `Phase 4: Advanced Modeling & Interpretability`
  * **Key Decisions & Evidence:** Generate TreeSHAP beeswarm and bar attribution. Unsupervised 2D t-SNE symptom phenotyping. Subgroup fairness audit across gender (AUC 0.824 vs 0.854) and age.

* **🔷 Stage 6: Operational Thresholding & Clinical Impact**
  * **Workbook Mapping:** `Phase 5: Decision Thresholding & Phase 6: Synthesis`
  * **Key Decisions & Evidence:** Reject arbitrary 0.50 threshold (misses 567 patients). Calibrate clinical cutoff to tau = 0.20 (81.2% sensitivity, rescuing 137 infected patients). Synthesize operational triage findings and clinical recommendations.

---

## 1. Stage-Aligned Decision Architecture & Ledger (CRISP-DM / MLOps Lifecycle)

The sections below decompose the **CRISP-DM Master Decision Architecture** into its operational stages. Each stage pairs a visual flowchart on the left with its decision ledger on the right:

* 🟢 **Green (Selected):** Chosen conceptual / production configuration.
* 🔴 **Red (Rejected):** Evaluated and rejected candidates.
* ⚪ **Slate (Alternative):** Architectural alternatives not required.

---
<table width="100%" border="0" cellpadding="0" cellspacing="14">
  <tr>
    <td width="42%" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_stage_1_flowchart.png" width="100%" alt="Stage 1 Flowchart" style="border: 1px solid #dadce0; border-radius: 6px; padding: 6px; background: #ffffff;">
    </td>
    <td width="58%" valign="top">
      <h3 style="margin: 0 0 8px 0; color: #1a73e8;">🔷 STAGE 1: Clinical Problem Formulation & Triage Mandate</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 1: Clinical Problem Formulation & Operational Framing</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>Calibrated Binary Probability Classifier</b> — Estimating continuous infection risk <i>P(y=1|X) &isin; [0, 1]</i> from 8 self-reported indicators accessible in under 30 seconds to prioritize RT-PCR laboratory swabs.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Multiclass Categorization</b> — Unnecessary clinical overhead when the immediate operational decision during testing scarcity is strictly binary (administer PCR swab vs. isolate safely at home).</li>
        <li style="margin-bottom: 8px;">⚪ <b>Alternative Option:</b> <b>Survival / Longitudinal Analysis</b> — Unfeasible due to absence of symptom onset dates in nationwide testing registry.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Generating continuous probabilities allows hospital systems to dynamically tune the decision cutoff &tau; based on daily swab supply and hospital bed capacity.</li>
      </ul>
    </td>
  </tr>
</table>

---
<table width="100%" border="0" cellpadding="0" cellspacing="14">
  <tr>
    <td width="42%" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_stage_2_flowchart.png" width="100%" alt="Stage 2 Flowchart" style="border: 1px solid #dadce0; border-radius: 6px; padding: 6px; background: #ffffff;">
    </td>
    <td width="58%" valign="top">
      <h3 style="margin: 0 0 8px 0; color: #1a73e8;">🔷 STAGE 2: Data Ingestion & Quality Auditing</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 2: Ingestion & Epidemiological EDA (Sections 2.1–2.2)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>Strict Ground Truth Filtration (<i>n</i> = 255,668)</b> — Drop inconclusive <code>other</code> outcomes (3,892) and missing gender records (19,445, &lt;7%) to guarantee clinical label fidelity.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Synthetic Imputation of Missing Gender</b> — Distorts demographic distributions without adding predictive power; dataset size (&gt;255,000 verified cases) provides massive statistical power without synthetic risks.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Removing ambiguous outcomes preserves pristine ground-truth labels without sacrificing statistical power, retaining 91.7% of the original nationwide cohort. Unmasked the <b>April 15 privacy redaction</b>, where age was 100% redacted mid-epidemic.</li>
      </ul>
    </td>
  </tr>
</table>

---
<table width="100%" border="0" cellpadding="0" cellspacing="14">
  <tr>
    <td width="42%" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_stage_3_flowchart.png" width="100%" alt="Stage 3 Flowchart" style="border: 1px solid #dadce0; border-radius: 6px; padding: 6px; background: #ffffff;">
    </td>
    <td width="58%" valign="top">
      <h3 style="margin: 0 0 8px 0; color: #1a73e8;">🔷 STAGE 3: Preprocessing & Pipeline Architecture</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 2.3–2.4 (Cohort Filtration & Redaction Discovery) & Phase 3 (Zero-Leakage Pipeline Architecture)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>Encapsulated <code>Pipeline</code> + Prospective Temporal Partitioning</b> — Preprocessing (<code>ColumnTransformer</code> with <code>OneHotEncoder(handle_unknown='ignore')</code> and symptom passthrough) encapsulated inside a Scikit-Learn <code>Pipeline</code> fit exclusively on historical training data (March 22 – April 7, <i>n</i>=98,989). Tested strictly on prospective hold-out (April 8–14, <i>n</i>=37,305).</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Random Shuffling (<code>train_test_split(shuffle=True)</code>)</b> — Disastrous look-ahead leakage across evolving epidemic waves; inflates scores artificially.</li>
        <li style="margin-bottom: 8px;">⚪ <b>Alternative Option:</b> <b>Global In-Place Transformation</b> — Leaks future category levels to training and causes training-serving skew in production deployment.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Respects the epidemiological &ldquo;Arrow of Time&rdquo; and MLOps encapsulation standards, ensuring identical transformation during training and live inference.</li>
      </ul>
    </td>
  </tr>
</table>

---
<table width="100%" border="0" cellpadding="0" cellspacing="14">
  <tr>
    <td width="42%" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_stage_4_flowchart.png" width="100%" alt="Stage 4 Flowchart" style="border: 1px solid #dadce0; border-radius: 6px; padding: 6px; background: #ffffff;">
    </td>
    <td width="58%" valign="top">
      <h3 style="margin: 0 0 8px 0; color: #1a73e8;">🔷 STAGE 4: Model Architecture & Boosting Benchmarks</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 3 (Benchmark Replication: Zoabi et al. 2021) & Phase 4 (XGBoost Classifier Benchmark)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>LightGBM Champion (<code>num_leaves=20</code>, <code>min_child_samples=4</code>, <code>learning_rate=0.05</code>, <code>is_unbalance=True</code>) & XGBoost Benchmark</b> — Fast histogram-based gradient boosting achieving ROC-AUC <b>0.8443</b> (PR-AUC 0.5448) and XGBoost ROC-AUC <b>0.8448</b> (PR-AUC 0.5480), verified via internal 5-fold CV (0.9351 &plusmn; 0.0025). Robust <b>Symptom-Only LightGBM (Ablating Age)</b> achieves ROC-AUC <b>0.8448</b> (PR-AUC 0.5403), proving resilience to the April 15 redaction cliff.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Deep Neural Networks</b> — Excessive compute overhead, slow inference, and inferior tabular performance compared to tree gradient boosting.</li>
        <li style="margin-bottom: 8px;">⚪ <b>Alternative Option:</b> <b>Direct Test-Set Tuning (Test Snooping)</b> — Tuning against the test set leaks future outcomes; validation must occur strictly within training folds.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Tree ensembles capture non-linear symptom co-occurrences without feature scaling, while <code>is_unbalance=True</code> mathematically corrects gradient updates for the 17.6:1 imbalance.</li>
      </ul>
    </td>
  </tr>
</table>

---
<table width="100%" border="0" cellpadding="0" cellspacing="14">
  <tr>
    <td width="42%" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_stage_5_flowchart.png" width="100%" alt="Stage 5 Flowchart" style="border: 1px solid #dadce0; border-radius: 6px; padding: 6px; background: #ffffff;">
    </td>
    <td width="58%" valign="top">
      <h3 style="margin: 0 0 8px 0; color: #1a73e8;">🔷 STAGE 5: Clinical Explainability & Demographic Fairness</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 4: Advanced Modeling & Interpretability (TreeSHAP, 2D t-SNE Phenotyping & Subgroup Parity Audit)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>TreeSHAP Explainability & Subgroup Demographic Fairness Audit</b> — Exact game-theoretic Shapley values reveal patient risk log-odds (<code>Contact with confirmed</code> and <code>Fever</code> dominate). Demographic slicing audits gender (Male AUC 0.8243 vs. Female AUC 0.8540) and age (&lt;60 AUC 0.8457 vs. 60+ AUC 0.8328). Includes 2D t-SNE unsupervised phenotyping.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Aggregate-Only Metric Reporting</b> — Violates healthcare ethics by obscuring disparate clinical impact across vulnerable protected groups.</li>
        <li style="margin-bottom: 8px;">⚪ <b>Alternative Option:</b> <b>Permutation Importance</b> — Global ranking only, lacking individual-patient attribution.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Transparent attribution is essential for clinical adoption and regulatory compliance (FDA/NHS AI guidelines).</li>
      </ul>
    </td>
  </tr>
</table>

---
<table width="100%" border="0" cellpadding="0" cellspacing="14">
  <tr>
    <td width="42%" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_stage_6_flowchart.png" width="100%" alt="Stage 6 Flowchart" style="border: 1px solid #dadce0; border-radius: 6px; padding: 6px; background: #ffffff;">
    </td>
    <td width="58%" valign="top">
      <h3 style="margin: 0 0 8px 0; color: #1a73e8;">🔷 STAGE 6: Operational Decision Thresholding & Clinical Impact</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 5 (Operational Decision Thresholding) & Phase 6 (Executive Synthesis & Clinical Translation)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>Calibrated Clinical Cutoff (&tau; = 0.20)</b> — Elevates sensitivity to <b>81.2%</b> (capturing 1,836 of 2,261 infected patients), rescuing 137 individuals who would have been sent home under &tau; = 0.50.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Default &tau; = 0.50 Cutoff</b> — Implicitly assumes equal error costs; disastrous in epidemics where false negatives drive silent community transmission.</li>
        <li style="margin-bottom: 8px;">⚪ <b>Alternative Option:</b> <b>Emergency Cutoff (&tau; = 0.10)</b> — 94.3% sensitivity (missing only 128 patients), but triggers 26,892 false alarms, overwhelming lab capacity.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> In infectious disease control, the cost of a False Negative far exceeds the cost of a False Positive. Tuning &tau; = 0.20 aligns ML with asymmetric healthcare risk.</li>
      </ul>
    </td>
  </tr>
</table>

---
## 2. Quantitative Summary & Benchmark Scorecard

All models are evaluated on the identical, unseen prospective hold-out cohort (8–14 April 2020, n = 37,305; 2,261 positive cases, 6.06% prevalence, random seed = 5):

| Model Architecture & Configuration | Features | Train 5-Fold CV AUC | Hold-out ROC-AUC | Hold-out PR-AUC | Sensitivity (tau=0.20) | Specificity (tau=0.20) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Rule Dummy Baseline** | None | 0.5000 | 0.5000 | 0.0606 | 0.0% | 100.0% |
| **Logistic Regression (Linear)** | 8 | 0.9280 ± 0.0052 | 0.8336 | 0.4808 | 77.4% | 72.1% |
| **Decision Tree Baseline (`depth=5`)** | 8 | 0.9226 ± 0.0045 | 0.8357 | 0.5062 | 78.1% | 71.5% |
| **Random Forest (100 Trees, Bagging)** | 8 | 0.9348 ± 0.0052 | 0.8412 | 0.5374 | 80.5% | 68.4% |
| **LightGBM Champion (Paper Params)** | 8 | **0.9351 ± 0.0025** | **0.8443** | **0.5448** | **81.2%** | **68.0%** |
| **LightGBM Symptom-Only (No Age)** | 7 | **0.9333 ± 0.0022** | **0.8448** | **0.5403** | **81.5%** | **67.8%** |
| **XGBoost Benchmark (`max_depth=4`)** | 8 | **0.9350 ± 0.0025** | **0.8448** | **0.5480** | **81.4%** | **68.1%** |

#### 🏥 Clinical Operational Assessments:
* **Zero-Rule Dummy Baseline:** 94.71% accuracy but misses 100% of sick patients (illustrates the Accuracy Fallacy).
* **Logistic Regression (Linear):** Strong baseline, but linear decision boundary misses non-linear symptom co-occurrences.
* **Decision Tree Baseline:** Replicates *Zoabi et al.* baseline; provides interpretable clinical tree rules.
* **Random Forest:** Parallel variance reduction with stable non-linear partitioning.
* **LightGBM Champion:** Replicates *Zoabi et al.* prospective test range (0.84–0.86); fast leaf-wise boosting.
* **LightGBM Symptom-Only:** Resilient model that retains full discrimination (0.8448 AUC) after the April 15 age privacy redaction.
* **XGBoost Benchmark:** Confirms boosting engine invariance with depth-wise tree growth.

---

## 3. Engineering Analysis: Key Methodological Discoveries

This section documents key operational and methodological discoveries identified across the clinical triage engineering pipeline:

* **The Scikit-Learn Pipeline & Zero-Leakage Architecture (Stage 3 & 4):**  
  *Discovery:* Replacing ad-hoc pandas preprocessing with an encapsulated `Pipeline` + `ColumnTransformer` (`OneHotEncoder(handle_unknown='ignore')`) guarantees that all transformers are fitted strictly on `X_train`. This eliminates categorical vocabulary leakage and guarantees production serving parity when new patient records arrive at a clinical API.
* **The April 15 Regulatory Privacy Redaction (Stage 2 & 4):**  
  *Discovery:* Auditing daily testing volume revealed why `age_60_and_above` became 100% missing after April 14—health authorities intervened mid-epidemic to redact age to prevent citizen re-identification linkage attacks. By building and benchmarking an accessible **Symptom-Only model** (ROC-AUC **0.8448**), we proved that the clinical triage tool retains full diagnostic discrimination even when demographic features are legally redacted.
* **The "Arrow of Time" Prospective Partitioning (Stage 3):**  
  *Discovery:* Proved that standard cross-sectional random shuffling (`train_test_split(shuffle=True)`) creates catastrophic temporal look-ahead leakage across evolving epidemic waves. Validated that testing on future dates (April 8–14) simulates real hospital intake conditions.
* **Demographic Fairness & Subgroup Parity Audit (Stage 5):**  
  *Discovery:* Evaluated models across protected demographics to satisfy healthcare AI ethics (NHS/WHO guidelines). Verified that the model maintains equitable discrimination across gender (Male AUC 0.8243 vs. Female AUC 0.8540) and age (<60 AUC 0.8457 vs. 60+ AUC 0.8328).
* **Asymmetric Cost-Sensitive Threshold Calibration (Stage 6):**  
  *Discovery:* Proved that the default machine learning cutoff of tau = 0.50 is clinically unsafe, missing 567 infected patients (74.9% sensitivity). Calibrated the triage cutoff to tau = 0.20 (81.2% sensitivity), mathematically capturing 137 additional infected patients for PCR swab confirmation.
