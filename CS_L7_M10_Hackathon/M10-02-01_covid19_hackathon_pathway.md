[![Companion Workbook in Colab](https://img.shields.io/badge/Colab-Open_Workbook-orange?logo=googlecolab)](https://colab.research.google.com/github/Llander22/Data-Science-Portfolio/blob/main/CS_L7_M10_Hackathon/M10-02-02_covid19_hackathon.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_Pathway.md-blue?logo=github)](https://github.com/Llander22/Data-Science-Portfolio/blob/main/CS_L7_M10_Hackathon/M10-02-01_covid19_hackathon_pathway.md)
[![Shared Deliverables in Google Drive](https://img.shields.io/badge/Google_Drive-Shared_PDFs-34A853?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1PzhHH9AoM4HMXEWWJRMZm3P4V0W21a1c?usp=drive_link)

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
  * **Workbook Mapping:** `Phase 2: Ingestion & Epidemiological Realities`
  * **Key Decisions & Evidence:** Audit 278,848 tests (51 days). Unmask 94.7% vs 5.3% class imbalance. Expose the "Accuracy Fallacy". Uncover the April 15 regulatory privacy redaction cliff.

* **🔷 Stage 3: Preprocessing & Pipeline Architecture**
  * **Workbook Mapping:** `Phase 2.3–2.4 & Phase 3: Pipeline Architecture`
  * **Key Decisions & Evidence:** Filter 3,892 inconclusive targets and 19k missing gender records (clean cohort n = 255,668). Enforce zero-leakage Scikit-Learn `Pipeline` + prospective split (train: 22 March–7 April; test: 8–14 April).

* **🔷 Stage 4: Modelling & Boosting Benchmarks**
  * **Workbook Mapping:** `Phase 3: Benchmark Replication & Phase 4: XGBoost`
  * **Key Decisions & Evidence:** Replicate *Zoabi et al.*: Decision Tree (0.8357), Random Forest (0.8412), LightGBM (0.8443, PR-AUC 0.5448), and XGBoost (0.8448, PR-AUC 0.5480). Validate via 5-fold CV within training window.

* **🔷 Stage 5: Clinical Explainability & Demographic Fairness**
  * **Workbook Mapping:** `Phase 4: Advanced Modelling & Interpretability`
  * **Key Decisions & Evidence:** Generate TreeSHAP beeswarm and bar attribution. Unsupervised 2D t-SNE symptom phenotyping. Subgroup fairness audit across gender (AUC 0.824 vs 0.854) and age. Demographically blind ablation (0.8400 AUC) proving pure clinical features suffice for ethical deployment.

* **🔷 Stage 6: Operational Thresholding & Clinical Impact**
  * **Workbook Mapping:** `Phase 5: Operational Decision Thresholding & Closing Summary`
  * **Key Decisions & Evidence:** Reject arbitrary 0.50 threshold (misses 567 patients). Calibrate clinical cutoff to tau = 0.20 (81.2% sensitivity, rescuing 137 infected patients). Synthesise operational triage findings and clinical recommendations.

* **🔷 Stage 7: Post-Hackathon Model Enhancements & Cross-Cohort Synthesis**
  * **Workbook Mapping:** `Further Improvement following the Hackathon Presentations`
  * **Key Decisions & Evidence:** Probability calibration auditing (Logistic Regression Brier score 0.0401 vs LightGBM 0.1315). Balanced accuracy (74.6%–86.0%) and sensitivity at fixed 80% and 85% hospital lab capacity (73.2%–75.1% infections caught). 3-symptom ablation mathematically proving the 0.014–0.025 clinician documentation artifact. Dynamic Bayesian threshold updating for shifting community prevalence across epidemic waves.

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
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>Calibrated Binary Probability Classifier</b> — Estimating continuous infection risk <i>P(y=1|X) &isin; [0, 1]</i> from 8 self-reported indicators accessible in under 30 seconds to prioritise RT-PCR laboratory swabs.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Multiclass Categorisation</b> — Unnecessary clinical overhead when the immediate operational decision during testing scarcity is strictly binary (administer PCR swab vs. isolate safely at home).</li>
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
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>Strict Ground Truth Filtration (<i>n</i> = 255,668)</b> — Drop inconclusive <code>other</code> outcomes (3,892) and missing gender records (19,445, &lt;7%) to guarantee clinical label fidelity for benchmark replication.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Synthetic Imputation of Missing Gender</b> — Distorts demographic distributions without adding predictive power; dataset size (&gt;255,000 verified cases) provides massive statistical power without synthetic risks.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Removing ambiguous outcomes preserves pristine ground-truth labels without sacrificing statistical power, retaining 91.7% of the original nationwide cohort. Unmasked the <b>April 15 privacy redaction</b>, where age was 100% redacted mid-epidemic. <i>(Note: While missing gender records are excluded here to strictly replicate the Nature benchmark, Stage 5 evaluates a Demographically Blind model that safely triages incomplete records in live deployment.)</i></li>
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
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Random Shuffling (<code>train_test_split(shuffle=True)</code>)</b> — Disastrous look-ahead leakage across evolving epidemic waves; artificially inflates test ROC-AUC to <b>0.9157</b> (PR-AUC 0.7192) by predicting the past using future epidemic data and smoothing away real prevalence decay (8.46% down to 6.06%).</li>
        <li style="margin-bottom: 8px;">⚪ <b>Alternative Option:</b> <b>Global In-Place Transformation</b> — Leaks future category levels to training and causes training-serving skew in production deployment.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Respects the epidemiological &ldquo;Arrow of Time&rdquo; and MLOps encapsulation standards. While <code>test_date</code> is never fed to the model as an input feature, it is mandatory as the structural validation spine to ensure genuine, leak-free prospective discrimination (<b>0.8451 ROC-AUC</b>) matching the published *Nature* benchmark.</li>
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
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Tree ensembles capture non-linear symptom co-occurrences without feature scaling, while <code>is_unbalance=True</code> mathematically corrects gradient updates for the 17.6:1 imbalance. Evaluated 50% sub-sampling (<code>max_samples=0.5</code>) on the baseline Random Forest: because ~100k rows saturate 8 binary features, it delivers an 80% reduction in training latency with zero loss in cross-validation discrimination (<code>0.9339</code> vs <code>0.9340</code>).</li>
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
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 4: Advanced Modelling & Interpretability (TreeSHAP, 2D t-SNE Phenotyping, Subgroup Parity & Demographically Blind Model)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>TreeSHAP Explainability & Subgroup Demographic Fairness Audit</b> — Exact game-theoretic Shapley values reveal patient risk log-odds (<code>Contact with confirmed</code> and <code>Fever</code> dominate). Demographic slicing audits gender (Male AUC 0.8243 vs. Female AUC 0.8540) and age (&lt;60 AUC 0.8457 vs. 60+ AUC 0.8328). Includes 2D t-SNE unsupervised phenotyping.</li>
        <li style="margin-bottom: 8px;">⚪ <b>Extension Option:</b> <b>Pure Clinical Demographically Blind Pipeline (ROC-AUC = 0.8400)</b> — Evaluates model performance when omitting both age and gender. Drops only 0.0051 AUC vs. the 8-feature benchmark, proving 99.4% of discriminative power is clinical. Eliminates triage drop-out for patients with unrecorded demographics and satisfies NHS/WHO AI ethics standards.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Aggregate-Only Metric Reporting</b> — Violates healthcare ethics by obscuring disparate clinical impact across vulnerable protected groups.</li>
        <li style="margin-bottom: 8px;">⚪ <b>Alternative Option:</b> <b>Permutation Importance</b> — Global ranking only, lacking individual-patient attribution.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Transparent attribution is essential for clinical adoption and regulatory compliance (FDA/NHS AI guidelines). Demographically blind ablation demonstrates that frontline triage can operate without demographic dependency.</li>
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
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 5: Operational Decision Thresholding & Closing Summary (Clinical Triage Insights)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>Calibrated Clinical Cutoff (&tau; = 0.20)</b> — Elevates sensitivity to <b>81.2%</b> (capturing 1,835 of 2,261 infected patients), rescuing 137 individuals who would have been sent home under &tau; = 0.50, delivering a <b>98.2% Negative Predictive Value (NPV)</b> for safe home monitoring while sparing 23,829 laboratory swabs.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Default &tau; = 0.50 Cutoff</b> — Implicitly assumes equal error costs; disastrous in epidemics where false negatives drive silent community transmission.</li>
        <li style="margin-bottom: 8px;">⚪ <b>Alternative Option:</b> <b>Emergency Cutoff (&tau; = 0.10)</b> — 94.3% sensitivity (missing only 128 patients), but triggers 26,892 false alarms, overwhelming lab capacity.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> In infectious disease control, the cost of a False Negative far exceeds the cost of a False Positive. Tuning &tau; = 0.20 aligns ML with asymmetric healthcare risk.</li>
      </ul>
    </td>
  </tr>
</table>

---

---
<table width="100%" border="0" cellpadding="0" cellspacing="14">
  <tr>
    <td width="42%" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_stage_6_flowchart.png" width="100%" alt="Stage 7 Flowchart" style="border: 1px solid #dadce0; border-radius: 6px; padding: 6px; background: #ffffff;">
    </td>
    <td width="58%" valign="top">
      <h3 style="margin: 0 0 8px 0; color: #1a73e8;">🔷 STAGE 7: Post-Hackathon Synthesis, Calibration & Prevalence Drift</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Further Improvement following the Hackathon Presentations</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>Dual Calibration & Bayesian Threshold Updating</b> — Evaluates probability calibration via Brier score (Logistic Regression achieves <b>0.0401</b>, exceeding the &lt;0.05 clinical gold standard). Benchmarks sensitivity under fixed hospital lab capacities (catching 75.1% of cases at 80% specificity; 73.2% at 85% specificity). Applies Bayes' rule in practice to dynamically adjust the operational cutoff &tau; based on 14-day rolling community test positivity.</li>
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>Clinician Documentation Bias Ablation</b> — Eliminating the three rare secondary symptoms (<code>head_ache</code>, <code>sore_throat</code>, <code>shortness_of_breath</code>) causes prospective AUROC to slip by merely <b>0.0144</b> (from 0.8412 to 0.8268), mathematically proving that &gt;98% of true biological signal is driven by fever, cough, and known exposure.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Static Threshold Deployment</b> — Keeping a static cutoff across winter surges (10% prevalence) and summer lulls (1% prevalence) causes severe false-positive flooding during quiet months.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Cross-cohort benchmarking confirms that the data sets the ceiling (595 symptom states). Dynamic Bayesian updating and documentation auditing guarantee regulatory compliance (UK MHRA SaMD) for safe, long-term deployment.</li>
      </ul>
    </td>
  </tr>
</table>

## 2. Quantitative Summary & Benchmark Scorecard

All models are evaluated on the identical, unseen prospective hold-out cohort (8–14 April 2020, n = 37,305; 2,261 positive cases, 6.06% prevalence, random seed = 5):

| Model Architecture & Configuration | Features | Train 5-Fold CV AUC | Hold-out ROC-AUC | Hold-out PR-AUC | Sensitivity (tau=0.20) | Specificity (tau=0.20) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Rule Dummy Baseline** | None | 0.5000 | 0.5000 | 0.0606 | 0.0% | 100.0% |
| **Logistic Regression (Linear)** | 8 | 0.9280 ± 0.0052 | 0.8336 | 0.4808 | 77.4% | 72.1% |
| **Decision Tree Baseline (`depth=5`)** | 8 | 0.9226 ± 0.0045 | 0.8357 | 0.5062 | 78.1% | 71.5% |
| **Random Forest (100 Trees, Bagging)** | 8 | 0.9348 ± 0.0052 | 0.8412 | 0.5374 | 80.5% | 68.4% |
| **LightGBM Champion (Paper Params)** | 8 | **0.9351 ± 0.0025** | **0.8443** | **0.5448** | **81.2%** | **68.0%** |
| **Core 5-Feature Biological (Ablation)** | 5 | **0.9310 ± 0.0030** | **0.8268** | **0.5210** | **78.4%** | **66.8%** |
| **LightGBM Symptom-Only (No Age)** | 7 | **0.9333 ± 0.0022** | **0.8448** | **0.5403** | **81.5%** | **67.8%** |
| **LightGBM Pure Clinical (Demographically Blind)** | 6 | **0.9328 ± 0.0024** | **0.8400** | **0.5331** | **80.8%** | **67.6%** |
| **XGBoost Benchmark (`max_depth=4`)** | 8 | **0.9350 ± 0.0025** | **0.8448** | **0.5480** | **81.4%** | **68.1%** |

#### 🏥 Clinical Operational Assessments:
* **Zero-Rule Dummy Baseline:** 94.71% accuracy but misses 100% of sick patients (illustrates the Accuracy Fallacy).
* **Logistic Regression (Linear):** Strong baseline, but linear decision boundary misses non-linear symptom co-occurrences.
* **Decision Tree Baseline:** Replicates *Zoabi et al.* baseline; provides interpretable clinical tree rules.
* **Random Forest:** Parallel variance reduction with stable non-linear partitioning (50% sub-sampling trains 5x faster with identical discrimination).
* **LightGBM Champion:** Replicates *Zoabi et al.* prospective test range (0.84–0.86); fast leaf-wise boosting.
* **LightGBM Symptom-Only:** Resilient model that retains full discrimination (0.8448 AUC) after the April 15 age privacy redaction.
* **LightGBM Pure Clinical:** Demographically blind pipeline retaining 0.8400 AUC, proving 99.4% of signal is clinical and enabling triage of all patients with unrecorded demographics.
* **XGBoost Benchmark:** Confirms boosting engine invariance with depth-wise tree growth.

---

## 3. Engineering Analysis: Key Methodological Discoveries

This section documents key operational and methodological discoveries identified across the clinical triage engineering pipeline:

* **The Scikit-Learn Pipeline & Zero-Leakage Architecture (Stage 3 & 4):**  
  *Discovery:* Replacing ad-hoc pandas preprocessing with an encapsulated `Pipeline` + `ColumnTransformer` (`OneHotEncoder(handle_unknown='ignore')`) guarantees that all transformers are fitted strictly on `X_train`. This eliminates categorical vocabulary leakage and guarantees production serving parity when new patient records arrive at a clinical API.
* **The April 15 Regulatory Privacy Redaction (Stage 2 & 4):**  
  *Discovery:* Auditing daily testing volume revealed why `age_60_and_above` became 100% missing after April 14—health authorities intervened mid-epidemic to redact age to prevent citizen re-identification linkage attacks. By building and benchmarking an accessible **Symptom-Only model** (ROC-AUC **0.8448**), we proved that the clinical triage tool retains full diagnostic discrimination even when demographic features are legally redacted.
* **The "Arrow of Time" Prospective Partitioning (Stage 3):**  
  *Discovery:* Proved that standard cross-sectional random shuffling (`train_test_split(shuffle=True)`) creates catastrophic temporal look-ahead leakage across evolving epidemic waves, artificially inflating ROC-AUC to 0.9157. Validated that testing on future dates (April 8–14) simulates real hospital intake conditions and captures genuine prevalence decay.
* **Demographic Fairness & Demographically Blind Governance (Stage 5):**  
  *Discovery:* Audited models across protected demographics to satisfy healthcare AI ethics (NHS/WHO guidelines). Verified that the model maintains equitable discrimination across gender (Male AUC 0.8243 vs. Female AUC 0.8540) and age (<60 AUC 0.8457 vs. 60+ AUC 0.8328). Further proved that an entirely demographically blind model (omitting both age and gender) achieves 0.8400 AUC, eliminating patient drop-out for the 19,445 individuals with unrecorded demographic fields.
* **Asymmetric Cost-Sensitive Threshold Calibration (Stage 6):**  
  *Discovery:* Proved that the default machine learning cutoff of tau = 0.50 is clinically unsafe, missing 563 infected patients (75.1% sensitivity). Calibrated the triage cutoff to tau = 0.20 (81.2% sensitivity), mathematically capturing 137 additional infected patients for PCR swab confirmation while keeping laboratory testing demand within manageable capacity limits. Supported by continuous clinical operating curve analysis.

* **The 3-Symptom Clinician Documentation Artifact & Calibration (Stage 7):**  
  *Discovery:* Post-hackathon cross-cohort ablation proved that removing headache, sore throat, and dyspnoea drops prospective AUC by only 0.0144 (0.8412 to 0.8268). This mathematically confirms that secondary symptoms were recorded disproportionately for high-suspicion patients (headache had a 266:1 recording ratio in positive vs negative swabs). Furthermore, probability calibration auditing revealed that standard Logistic Regression delivers a clinically pristine Brier score of **0.0401** (<0.05 gold standard), while tree models require threshold shifting or Platt calibration due to positive-class upweighting (`is_unbalance=True`).

---

## 4. Addendum: Real-World Clinical Deployment in 2026

Deploying an epidemiological machine learning model into a live 2026 healthcare ecosystem requires transitioning from static retrospective validation to continuous, safety-critical MLOps governance:

<table width="100%" style="border-collapse: collapse; margin-top: 8px;">
<tr>
<th width="50%" style="background: #1e3a8a; color: #ffffff; padding: 6px 10px; text-align: left; font-size: 12px;">🏛️ Governance, Drift Monitoring & Architecture</th>
<th width="50%" style="background: #1e3a8a; color: #ffffff; padding: 6px 10px; text-align: left; font-size: 12px;">🏥 Clinical Operations, Delayed Feedback & Human-in-the-Loop</th>
</tr>
<tr>
<td width="50%" valign="top" style="padding: 10px; border: 1px solid #cbd5e1; font-size: 11px; line-height: 1.5; vertical-align: top;">
<h4 style="margin: 0 0 4px 0; color: #1e3a8a;">1. CRISP-DM to Mitchell et al. Model Cards (arXiv:1810.03993)</h4>
<p style="margin: 0 0 8px 0;">CRISP-DM standardises the engineering workflow (data preparation to threshold tuning). However, clinical deployment requires a standardised governance passport: the <b>Model Card</b>. Essential for UK MHRA Software as a Medical Device (SaMD) and EU AI Act high-risk compliance, the Model Card explicitly records intended use (secondary triage, not mass screening), subgroup fairness across demographics, demographic ablation resilience (0.8400 AUC without age/gender), and false-negative risk boundaries.</p>
<h4 style="margin: 0 0 4px 0; color: #1e3a8a;">2. Covariate Shift & Variant Drift Monitoring</h4>
<p style="margin: 0 0 8px 0;">Pathogens mutate and symptoms evolve across epidemic waves (e.g. anosmia in wild-type vs. upper respiratory symptoms in Omicron). The 2026 production pipeline integrates automated drift surveillance (Evidently AI / Great Expectations). It tracks the <b>Population Stability Index (PSI)</b> weekly across incoming symptoms; any shift exceeding <code>PSI &gt; 0.20</code> triggers clinical review and dataset refreshing before silent diagnostic decay occurs.</p>
<h4 style="margin: 0 0 4px 0; color: #1e3a8a;">3. Sub-5ms Inference Serving & EHR Integration</h4>
<p style="margin: 0;">Clinical triage cannot tolerate server latency. The pipeline is containerised as a lightweight microservice (FastAPI + ONNX Runtime). Scoring 6 to 8 binary indicators takes under <b>5 milliseconds</b> per patient, embedding seamlessly into hospital Electronic Health Records (Epic, Cerner) and national intake systems (NHS 111) via HL7/FHIR standards.</p>
</td>
<td width="50%" valign="top" style="padding: 10px; border: 1px solid #cbd5e1; font-size: 11px; line-height: 1.5; vertical-align: top;">
<h4 style="margin: 0 0 4px 0; color: #1e3a8a;">4. Delayed Ground-Truth Labels & Canary Deployments</h4>
<p style="margin: 0 0 8px 0;">RT-PCR swabs take 24–72 hours to return from regional laboratories, creating an unavoidable delayed feedback loop. Production MLOps reconciles intake predictions asynchronously against incoming laboratory LIMS outcomes. Candidate retrained models run on a <b>Champion-Challenger (90/10) canary split</b> and are only promoted after demonstrating superior prospective sensitivity across two weeks of verified laboratory tests.</p>
<h4 style="margin: 0 0 4px 0; color: #1e3a8a;">5. Three-Tier Escalation & Clinician Override Auditing</h4>
<p style="margin: 0 0 8px 0;">The calibrated &tau; = 0.20 cutoff anchors a safe three-tier operational policy:<br>&bull; <b>Low Risk (&lt; 0.15):</b> Home isolation advice and symptom tracking; swabs spared.<br>&bull; <b>Borderline Zone (0.15 &le; p &le; 0.25):</b> Mandatory escalation to a triage nurse or clinician for physical assessment.<br>&bull; <b>High Risk (&gt; 0.25):</b> Priority RT-PCR swabbing and rapid isolation.<br>All <b>clinician overrides</b> are logged and audited weekly to detect emerging symptom co-occurrences and mitigate alert fatigue.</p>
<h4 style="margin: 0 0 4px 0; color: #1e3a8a;">6. Working Approach & Shared Deliverables</h4>
<p style="margin: 0;">The companion technical workbook details the working methodology (extending to real-world deployment, dual deliverables, and human-AI pair programming). Both project deliverables are published and accessible on Google Drive:<br>&bull; 📄 <b>Decision Pathway (PDF):</b> <code>M10-02-01_covid19_hackathon_pathway.pdf</code> (11-page landscape architecture).<br>&bull; 📓 <b>Master Technical Workbook (PDF):</b> <code>M10-02-02_covid19_hackathon.pdf</code> (34-page engineering notebook).<br>🔗 <b><a href="https://drive.google.com/drive/folders/1PzhHH9AoM4HMXEWWJRMZm3P4V0W21a1c?usp=drive_link" target="_blank" style="color: #2563eb; font-weight: 700;">Open Shared Deliverables in Google Drive / Google Docs</a></b></p>
</td>
</tr>
</table>

