[![Open Pathway in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Llander22/Data-Science-Portfolio/blob/main/CS_L7_M10_Hackathon/M10-02-01_covid19_hackathon_pathway.ipynb)
[![Open Workbook in Colab](https://img.shields.io/badge/Colab-Open_Workbook-orange?logo=googlecolab)](https://colab.research.google.com/github/Llander22/Data-Science-Portfolio/blob/main/CS_L7_M10_Hackathon/M10-02-02_covid19_hackathon.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_Pathway.md-blue?logo=github)](https://github.com/Llander22/Data-Science-Portfolio/blob/main/CS_L7_M10_Hackathon/M10-02-01_covid19_hackathon_pathway.md)

# Decision Pathway: COVID-19 Clinical Triage & Population Screening
### Operational Standard: CRISP-DM & Machine Learning Engineering Lifecycle

> **Operational Standard:** Cross-Industry Standard Process for Data Mining (CRISP-DM) & MLOps Engineering Lifecycle  
> **Technical Implementation:** `M10-02-02_covid19_hackathon.ipynb`  
> **Benchmark Reference:** Zoabi, Y., Deri-Rozov, S. & Shomron, N. *Machine learning-based prediction of COVID-19 diagnosis based on symptoms.* **npj Digital Medicine** 4, 3 (2021).  
> **Lifecycle Topology:** Full 6-Stage Architecture with **Selected Path (Green)**, **Rejected Options (Red)**, and **Alternative Branches (Slate)**.

---

### 📖 Narrative Workbook vs. Engineering Lifecycle Crosswalk

<div style="margin: 18px 12px 28px 4px; max-width: 96%; overflow-x: auto; padding-right: 12px;">
<table style="width: 100%; border-collapse: collapse; table-layout: fixed; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13px; line-height: 1.5; color: #1e293b; border: 1px solid #cbd5e1; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
  <colgroup>
    <col style="width: 28%;">
    <col style="width: 26%;">
    <col style="width: 46%;">
  </colgroup>
  <thead>
    <tr style="background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); color: #ffffff;">
      <th style="padding: 10px 14px; text-align: left; font-weight: 600; border-bottom: 2px solid #0369a1;">Engineering Lifecycle Stage</th>
      <th style="padding: 10px 14px; text-align: left; font-weight: 600; border-bottom: 2px solid #0369a1;">Workbook Section Mapping</th>
      <th style="padding: 10px 14px; text-align: left; font-weight: 600; border-bottom: 2px solid #0369a1;">Key Decisions & Evidence</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #ffffff; border-bottom: 1px solid #e2e8f0;">
      <td style="padding: 10px 14px; font-weight: 600; color: #0369a1; vertical-align: top; word-wrap: break-word; white-space: normal;">🔷 Stage 1: Problem Formulation & Clinical Triage</td>
      <td style="padding: 10px 14px; font-weight: 600; color: #334155; vertical-align: top; word-wrap: break-word; white-space: normal;">Phase 1: Clinical Problem Formulation</td>
      <td style="padding: 10px 14px; color: #475569; vertical-align: top; word-wrap: break-word; white-space: normal;">Define binary diagnostic risk (<em>y</em> &isin; {0, 1}) under acute March 2020 RT-PCR scarcity. Mandate non-invasive 30-second self-reporting. Enforce the chronological &ldquo;Arrow of Time&rdquo;.</td>
    </tr>
    <tr style="background-color: #f8fafc; border-bottom: 1px solid #e2e8f0;">
      <td style="padding: 10px 14px; font-weight: 600; color: #0369a1; vertical-align: top; word-wrap: break-word; white-space: normal;">🔷 Stage 2: Data Ingestion & Quality Auditing</td>
      <td style="padding: 10px 14px; font-weight: 600; color: #334155; vertical-align: top; word-wrap: break-word; white-space: normal;">Phase 2: Ingestion & Epidemiological EDA</td>
      <td style="padding: 10px 14px; color: #475569; vertical-align: top; word-wrap: break-word; white-space: normal;">Audit 278,848 tests (51 days). Unmask 94.7% vs 5.3% class imbalance. Expose the &ldquo;Accuracy Fallacy&rdquo;. Uncover the April 15 regulatory privacy redaction cliff.</td>
    </tr>
    <tr style="background-color: #ffffff; border-bottom: 1px solid #e2e8f0;">
      <td style="padding: 10px 14px; font-weight: 600; color: #0369a1; vertical-align: top; word-wrap: break-word; white-space: normal;">🔷 Stage 3: Preprocessing & Pipeline Architecture</td>
      <td style="padding: 10px 14px; font-weight: 600; color: #334155; vertical-align: top; word-wrap: break-word; white-space: normal;">Phase 2.3–2.4 & Phase 3: Pipeline Architecture</td>
      <td style="padding: 10px 14px; color: #475569; vertical-align: top; word-wrap: break-word; white-space: normal;">Filter 3,892 inconclusive targets and 19k missing gender records (clean cohort <em>n</em> = 255,668). Enforce zero-leakage Scikit-Learn <code>Pipeline</code> + prospective split (train: 22 March–7 April; test: 8–14 April).</td>
    </tr>
    <tr style="background-color: #f8fafc; border-bottom: 1px solid #e2e8f0;">
      <td style="padding: 10px 14px; font-weight: 600; color: #0369a1; vertical-align: top; word-wrap: break-word; white-space: normal;">🔷 Stage 4: Modeling & Boosting Benchmarks</td>
      <td style="padding: 10px 14px; font-weight: 600; color: #334155; vertical-align: top; word-wrap: break-word; white-space: normal;">Phase 3: Benchmark Replication & Phase 4: XGBoost</td>
      <td style="padding: 10px 14px; color: #475569; vertical-align: top; word-wrap: break-word; white-space: normal;">Replicate <em>Zoabi et al.</em>: Decision Tree (0.8357), Random Forest (0.8412), LightGBM (0.8443, PR-AUC 0.5448), and XGBoost (0.8448, PR-AUC 0.5480). Validate via 5-fold CV within training window.</td>
    </tr>
    <tr style="background-color: #ffffff; border-bottom: 1px solid #e2e8f0;">
      <td style="padding: 10px 14px; font-weight: 600; color: #0369a1; vertical-align: top; word-wrap: break-word; white-space: normal;">🔷 Stage 5: Explainability & Demographic Fairness</td>
      <td style="padding: 10px 14px; font-weight: 600; color: #334155; vertical-align: top; word-wrap: break-word; white-space: normal;">Phase 4: Advanced Modeling & Interpretability</td>
      <td style="padding: 10px 14px; color: #475569; vertical-align: top; word-wrap: break-word; white-space: normal;">Generate TreeSHAP beeswarm and bar attribution. Unsupervised 2D t-SNE symptom phenotyping. Subgroup fairness audit across gender (AUC 0.824 vs 0.854) and age.</td>
    </tr>
    <tr style="background-color: #f8fafc;">
      <td style="padding: 10px 14px; font-weight: 600; color: #0369a1; vertical-align: top; word-wrap: break-word; white-space: normal;">🔷 Stage 6: Thresholding & Clinical Impact</td>
      <td style="padding: 10px 14px; font-weight: 600; color: #334155; vertical-align: top; word-wrap: break-word; white-space: normal;">Phase 5: Decision Thresholding & Phase 6: Synthesis</td>
      <td style="padding: 10px 14px; color: #475569; vertical-align: top; word-wrap: break-word; white-space: normal;">Reject arbitrary 0.50 threshold (misses 567 patients). Calibrate clinical cutoff to &tau; = 0.20 (81.2% sensitivity, rescuing 137 infected patients). Synthesize operational triage findings.</td>
    </tr>
  </tbody>
</table>
</div>

---

## 1. Stage-Aligned Decision Architecture & Ledger (CRISP-DM / MLOps Lifecycle)

The cards below decompose the **CRISP-DM Master Decision Architecture** into its operational stages. Each stage features a high-resolution visual flowchart on the left paired with its corresponding decision ledger card on the right:

* 🟢 **Green (Selected):** Chosen conceptual / production configuration.
* 🔴 **Red (Rejected):** Evaluated and rejected candidates.
* ⚪ **Slate (Alternative):** Architectural alternatives not required.

---
<!-- STAGE 1 ROW -->
<div style="display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; margin: 18px 12px 28px 4px; padding-bottom: 22px; border-bottom: 1px solid #e2e8f0; max-width: 96%;">

  <div style="flex: 1 1 360px; max-width: 440px; text-align: center; margin: 0 auto;">
    <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_stage_1_flowchart.png" alt="Stage 1 Flowchart" style="width: 100%; max-width: 420px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); background: #ffffff; padding: 12px; box-sizing: border-box;">
  </div>

  <div style="flex: 1 1 420px; min-width: 320px; box-sizing: border-box; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 20px 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); word-wrap: break-word; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13.5px; line-height: 1.55; color: #1e293b;">

    <h4 style="margin-top: 0; margin-bottom: 8px; color: #0369a1; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">🔷 STAGE 1: Clinical Problem Formulation & Triage Mandate</h4>
    <p style="margin: 6px 0 12px 0; font-size: 12.5px; color: #64748b; background: #f8fafc; padding: 6px 10px; border-radius: 4px; border-left: 3px solid #0284c7;">
      📖 <strong>Workbook Mapping:</strong> Phase 1: Clinical Problem Formulation & Operational Framing
    </p>

    <ul style="margin: 8px 0; padding-left: 18px;">
      <li style="margin-bottom: 8px;">🟢 <strong>Selected Option:</strong> <strong>Calibrated Binary Probability Classifier</strong> — Estimating continuous infection risk <em>P</em>(<em>y</em>=1|<em>X</em>) &isin; [0, 1] from 8 self-reported indicators accessible in under 30 seconds to prioritize RT-PCR laboratory swabs.</li>
      <li style="margin-bottom: 8px;">🔴 <strong>Rejected Option:</strong> <strong>Multiclass Categorization</strong> — Unnecessary clinical overhead when the immediate operational decision during testing scarcity is strictly binary (administer PCR swab vs. isolate safely at home).</li>
      <li style="margin-bottom: 8px;">⚪ <strong>Alternative Option:</strong> <strong>Survival / Longitudinal Analysis</strong> — Unfeasible due to absence of symptom onset dates in nationwide testing registry.</li>
      <li style="margin-bottom: 0;">🛡️ <strong>Defensibility Rationale:</strong> Generating continuous probabilities allows hospital systems to dynamically tune the decision cutoff &tau; based on daily swab supply and hospital bed capacity.</li>
    </ul>

  </div>

</div>
<!-- STAGE 2 ROW -->
<div style="display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; margin: 18px 12px 28px 4px; padding-bottom: 22px; border-bottom: 1px solid #e2e8f0; max-width: 96%;">

  <div style="flex: 1 1 360px; max-width: 440px; text-align: center; margin: 0 auto;">
    <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_stage_2_flowchart.png" alt="Stage 2 Flowchart" style="width: 100%; max-width: 420px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); background: #ffffff; padding: 12px; box-sizing: border-box;">
  </div>

  <div style="flex: 1 1 420px; min-width: 320px; box-sizing: border-box; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 20px 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); word-wrap: break-word; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13.5px; line-height: 1.55; color: #1e293b;">

    <h4 style="margin-top: 0; margin-bottom: 8px; color: #0369a1; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">🔷 STAGE 2: Data Ingestion & Quality Auditing</h4>
    <p style="margin: 6px 0 12px 0; font-size: 12.5px; color: #64748b; background: #f8fafc; padding: 6px 10px; border-radius: 4px; border-left: 3px solid #0284c7;">
      📖 <strong>Workbook Mapping:</strong> Phase 2: Ingestion & Epidemiological EDA (Sections 2.1–2.2)
    </p>

    <ul style="margin: 8px 0; padding-left: 18px;">
      <li style="margin-bottom: 8px;">🟢 <strong>Selected Option:</strong> <strong>Strict Ground Truth Filtration (<em>n</em> = 255,668)</strong> — Drop inconclusive <code>other</code> outcomes (3,892) and missing gender records (19,445, &lt;7%) to guarantee clinical label fidelity.</li>
      <li style="margin-bottom: 8px;">🔴 <strong>Rejected Option:</strong> <strong>Synthetic Imputation of Missing Gender</strong> — Distorts demographic distributions without adding predictive power; dataset size (&gt;255,000 verified cases) provides massive statistical power without synthetic risks.</li>
      
      <li style="margin-bottom: 0;">🛡️ <strong>Defensibility Rationale:</strong> Removing ambiguous outcomes preserves pristine ground-truth labels without sacrificing statistical power, retaining 91.7% of the original nationwide cohort. Unmasked the <strong>April 15 privacy redaction</strong>, where age was 100% redacted mid-epidemic.</li>
    </ul>

  </div>

</div>
<!-- STAGE 3 ROW -->
<div style="display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; margin: 18px 12px 28px 4px; padding-bottom: 22px; border-bottom: 1px solid #e2e8f0; max-width: 96%;">

  <div style="flex: 1 1 360px; max-width: 440px; text-align: center; margin: 0 auto;">
    <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_stage_3_flowchart.png" alt="Stage 3 Flowchart" style="width: 100%; max-width: 420px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); background: #ffffff; padding: 12px; box-sizing: border-box;">
  </div>

  <div style="flex: 1 1 420px; min-width: 320px; box-sizing: border-box; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 20px 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); word-wrap: break-word; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13.5px; line-height: 1.55; color: #1e293b;">

    <h4 style="margin-top: 0; margin-bottom: 8px; color: #0369a1; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">🔷 STAGE 3: Preprocessing & Pipeline Architecture</h4>
    <p style="margin: 6px 0 12px 0; font-size: 12.5px; color: #64748b; background: #f8fafc; padding: 6px 10px; border-radius: 4px; border-left: 3px solid #0284c7;">
      📖 <strong>Workbook Mapping:</strong> Phase 2.3–2.4 (Cohort Filtration & Redaction Discovery) & Phase 3 (Zero-Leakage Pipeline Architecture)
    </p>

    <ul style="margin: 8px 0; padding-left: 18px;">
      <li style="margin-bottom: 8px;">🟢 <strong>Selected Option:</strong> <strong>Encapsulated <code>Pipeline</code> + Prospective Temporal Partitioning</strong> — Preprocessing (<code>ColumnTransformer</code> with <code>OneHotEncoder(handle_unknown='ignore')</code> and symptom passthrough) encapsulated inside a Scikit-Learn <code>Pipeline</code> fit exclusively on historical training data (March 22 – April 7, <em>n</em>=98,989). Tested strictly on prospective hold-out (April 8–14, <em>n</em>=37,305).</li>
      <li style="margin-bottom: 8px;">🔴 <strong>Rejected Option:</strong> <strong>Random Shuffling (<code>train_test_split(shuffle=True)</code>)</strong> — Disastrous look-ahead leakage across evolving epidemic waves; inflates scores artificially.</li>
      <li style="margin-bottom: 8px;">⚪ <strong>Alternative Option:</strong> <strong>Global In-Place Transformation</strong> — Leaks future category levels to training and causes training-serving skew in production deployment.</li>
      <li style="margin-bottom: 0;">🛡️ <strong>Defensibility Rationale:</strong> Respects the epidemiological &ldquo;Arrow of Time&rdquo; and MLOps encapsulation standards, ensuring identical transformation during training and live inference.</li>
    </ul>

  </div>

</div>
<!-- STAGE 4 ROW -->
<div style="display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; margin: 18px 12px 28px 4px; padding-bottom: 22px; border-bottom: 1px solid #e2e8f0; max-width: 96%;">

  <div style="flex: 1 1 360px; max-width: 440px; text-align: center; margin: 0 auto;">
    <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_stage_4_flowchart.png" alt="Stage 4 Flowchart" style="width: 100%; max-width: 420px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); background: #ffffff; padding: 12px; box-sizing: border-box;">
  </div>

  <div style="flex: 1 1 420px; min-width: 320px; box-sizing: border-box; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 20px 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); word-wrap: break-word; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13.5px; line-height: 1.55; color: #1e293b;">

    <h4 style="margin-top: 0; margin-bottom: 8px; color: #0369a1; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">🔷 STAGE 4: Model Architecture & Boosting Benchmarks</h4>
    <p style="margin: 6px 0 12px 0; font-size: 12.5px; color: #64748b; background: #f8fafc; padding: 6px 10px; border-radius: 4px; border-left: 3px solid #0284c7;">
      📖 <strong>Workbook Mapping:</strong> Phase 3 (Benchmark Replication: Zoabi et al. 2021) & Phase 4 (XGBoost Classifier Benchmark)
    </p>

    <ul style="margin: 8px 0; padding-left: 18px;">
      <li style="margin-bottom: 8px;">🟢 <strong>Selected Option:</strong> <strong>LightGBM Champion (<code>num_leaves=20</code>, <code>min_child_samples=4</code>, <code>learning_rate=0.05</code>, <code>is_unbalance=True</code>) & XGBoost Benchmark</strong> — Fast histogram-based gradient boosting achieving ROC-AUC <strong>0.8443</strong> (PR-AUC 0.5448) and XGBoost ROC-AUC <strong>0.8448</strong> (PR-AUC 0.5480), verified via internal 5-fold CV (0.9351 &plusmn; 0.0025). Robust <strong>Symptom-Only LightGBM (Ablating Age)</strong> achieves ROC-AUC <strong>0.8448</strong> (PR-AUC 0.5403), proving resilience to the April 15 redaction cliff.</li>
      <li style="margin-bottom: 8px;">🔴 <strong>Rejected Option:</strong> <strong>Deep Neural Networks</strong> — Excessive compute overhead, slow inference, and inferior tabular performance compared to tree gradient boosting.</li>
      <li style="margin-bottom: 8px;">⚪ <strong>Alternative Option:</strong> <strong>Direct Test-Set Tuning (Test Snooping)</strong> — Tuning against the test set leaks future outcomes; validation must occur strictly within training folds.</li>
      <li style="margin-bottom: 0;">🛡️ <strong>Defensibility Rationale:</strong> Tree ensembles capture non-linear symptom co-occurrences without feature scaling, while <code>is_unbalance=True</code> mathematically corrects gradient updates for the 17.6:1 imbalance.</li>
    </ul>

  </div>

</div>
<!-- STAGE 5 ROW -->
<div style="display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; margin: 18px 12px 28px 4px; padding-bottom: 22px; border-bottom: 1px solid #e2e8f0; max-width: 96%;">

  <div style="flex: 1 1 360px; max-width: 440px; text-align: center; margin: 0 auto;">
    <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_stage_5_flowchart.png" alt="Stage 5 Flowchart" style="width: 100%; max-width: 420px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); background: #ffffff; padding: 12px; box-sizing: border-box;">
  </div>

  <div style="flex: 1 1 420px; min-width: 320px; box-sizing: border-box; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 20px 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); word-wrap: break-word; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13.5px; line-height: 1.55; color: #1e293b;">

    <h4 style="margin-top: 0; margin-bottom: 8px; color: #0369a1; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">🔷 STAGE 5: Clinical Explainability & Demographic Fairness</h4>
    <p style="margin: 6px 0 12px 0; font-size: 12.5px; color: #64748b; background: #f8fafc; padding: 6px 10px; border-radius: 4px; border-left: 3px solid #0284c7;">
      📖 <strong>Workbook Mapping:</strong> Phase 4: Advanced Modeling & Interpretability (TreeSHAP, 2D t-SNE Phenotyping & Subgroup Parity Audit)
    </p>

    <ul style="margin: 8px 0; padding-left: 18px;">
      <li style="margin-bottom: 8px;">🟢 <strong>Selected Option:</strong> <strong>TreeSHAP Explainability & Subgroup Demographic Fairness Audit</strong> — Exact game-theoretic Shapley values reveal patient risk log-odds (<code>Contact with confirmed</code> and <code>Fever</code> dominate). Demographic slicing audits gender (Male AUC 0.8243 vs. Female AUC 0.8540) and age (&lt;60 AUC 0.8457 vs. 60+ AUC 0.8328). Includes 2D t-SNE unsupervised phenotyping.</li>
      <li style="margin-bottom: 8px;">🔴 <strong>Rejected Option:</strong> <strong>Aggregate-Only Metric Reporting</strong> — Violates healthcare ethics by obscuring disparate clinical impact across vulnerable protected groups.</li>
      <li style="margin-bottom: 8px;">⚪ <strong>Alternative Option:</strong> <strong>Permutation Importance</strong> — Global ranking only, lacking individual-patient attribution.</li>
      <li style="margin-bottom: 0;">🛡️ <strong>Defensibility Rationale:</strong> Transparent attribution is essential for clinical adoption and regulatory compliance (FDA/NHS AI guidelines).</li>
    </ul>

  </div>

</div>
<!-- STAGE 6 ROW -->
<div style="display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; margin: 18px 12px 28px 4px; padding-bottom: 22px; border-bottom: 1px solid #e2e8f0; max-width: 96%;">

  <div style="flex: 1 1 360px; max-width: 440px; text-align: center; margin: 0 auto;">
    <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_stage_6_flowchart.png" alt="Stage 6 Flowchart" style="width: 100%; max-width: 420px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); background: #ffffff; padding: 12px; box-sizing: border-box;">
  </div>

  <div style="flex: 1 1 420px; min-width: 320px; box-sizing: border-box; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 20px 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); word-wrap: break-word; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13.5px; line-height: 1.55; color: #1e293b;">

    <h4 style="margin-top: 0; margin-bottom: 8px; color: #0369a1; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">🔷 STAGE 6: Operational Decision Thresholding & Clinical Impact</h4>
    <p style="margin: 6px 0 12px 0; font-size: 12.5px; color: #64748b; background: #f8fafc; padding: 6px 10px; border-radius: 4px; border-left: 3px solid #0284c7;">
      📖 <strong>Workbook Mapping:</strong> Phase 5 (Operational Decision Thresholding) & Phase 6 (Executive Synthesis & Clinical Translation)
    </p>

    <ul style="margin: 8px 0; padding-left: 18px;">
      <li style="margin-bottom: 8px;">🟢 <strong>Selected Option:</strong> <strong>Calibrated Clinical Cutoff (&tau; = 0.20)</strong> — Elevates sensitivity to <strong>81.2%</strong> (capturing 1,836 of 2,261 infected patients), rescuing 137 individuals who would have been sent home under &tau; = 0.50.</li>
      <li style="margin-bottom: 8px;">🔴 <strong>Rejected Option:</strong> <strong>Default &tau; = 0.50 Cutoff</strong> — Implicitly assumes equal error costs (<em>C</em><sub>FN</sub> = <em>C</em><sub>FP</sub>); disastrous in epidemics where false negatives drive silent community transmission.</li>
      <li style="margin-bottom: 8px;">⚪ <strong>Alternative Option:</strong> <strong>Emergency Cutoff (&tau; = 0.10)</strong> — 94.3% sensitivity (missing only 128 patients), but triggers 26,892 false alarms, overwhelming lab capacity.</li>
      <li style="margin-bottom: 0;">🛡️ <strong>Defensibility Rationale:</strong> In infectious disease control, the cost of a False Negative far exceeds the cost of a False Positive. Tuning &tau; = 0.20 aligns ML with asymmetric healthcare risk.</li>
    </ul>

  </div>

</div>
---

## 2. Quantitative Summary & Benchmark Scorecard

All models are evaluated on the identical, unseen prospective hold-out cohort (8–14 April 2020, $n = 37,305$; 2,261 positive cases, 6.06% prevalence, random seed = 5):

<div style="margin: 18px 12px 28px 4px; max-width: 96%; overflow-x: auto; padding-right: 12px;">
<table style="width: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 12.5px; line-height: 1.45; color: #1e293b; border: 1px solid #cbd5e1; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
  <thead>
    <tr style="background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); color: #ffffff;">
      <th style="padding: 10px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #0369a1;">Model Architecture & Configuration</th>
      <th style="padding: 10px 8px; text-align: center; font-weight: 600; border-bottom: 2px solid #0369a1;">Features</th>
      <th style="padding: 10px 10px; text-align: center; font-weight: 600; border-bottom: 2px solid #0369a1;">Train 5-Fold CV AUC</th>
      <th style="padding: 10px 10px; text-align: center; font-weight: 600; border-bottom: 2px solid #0369a1;">Hold-out ROC-AUC</th>
      <th style="padding: 10px 10px; text-align: center; font-weight: 600; border-bottom: 2px solid #0369a1;">Hold-out PR-AUC</th>
      <th style="padding: 10px 10px; text-align: center; font-weight: 600; border-bottom: 2px solid #0369a1;">Sensitivity (&tau;=0.20)</th>
      <th style="padding: 10px 10px; text-align: center; font-weight: 600; border-bottom: 2px solid #0369a1;">Specificity (&tau;=0.20)</th>
      <th style="padding: 10px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #0369a1;">Clinical Operational Assessment</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #ffffff; border-bottom: 1px solid #e2e8f0;">
      <td style="padding: 9px 12px; font-weight: 600; color: #334155;">Zero-Rule Dummy Baseline</td>
      <td style="padding: 9px 8px; text-align: center;">None</td>
      <td style="padding: 9px 10px; text-align: center;">0.5000</td>
      <td style="padding: 9px 10px; text-align: center;">0.5000</td>
      <td style="padding: 9px 10px; text-align: center;">0.0606</td>
      <td style="padding: 9px 10px; text-align: center;">0.0%</td>
      <td style="padding: 9px 10px; text-align: center;">100.0%</td>
      <td style="padding: 9px 12px; color: #64748b; font-size: 11.5px;">94.71% accuracy but misses 100% of sick patients (Accuracy Fallacy).</td>
    </tr>
    <tr style="background-color: #f8fafc; border-bottom: 1px solid #e2e8f0;">
      <td style="padding: 9px 12px; font-weight: 600; color: #334155;">Logistic Regression (Linear)</td>
      <td style="padding: 9px 8px; text-align: center;">8 Features</td>
      <td style="padding: 9px 10px; text-align: center;">0.9280 &plusmn; 0.0052</td>
      <td style="padding: 9px 10px; text-align: center;">0.8336</td>
      <td style="padding: 9px 10px; text-align: center;">0.4808</td>
      <td style="padding: 9px 10px; text-align: center;">77.4%</td>
      <td style="padding: 9px 10px; text-align: center;">72.1%</td>
      <td style="padding: 9px 12px; color: #64748b; font-size: 11.5px;">Strong baseline, but linear boundary misses symptom co-occurrences.</td>
    </tr>
    <tr style="background-color: #ffffff; border-bottom: 1px solid #e2e8f0;">
      <td style="padding: 9px 12px; font-weight: 600; color: #334155;">Decision Tree Baseline (depth=5)</td>
      <td style="padding: 9px 8px; text-align: center;">8 Features</td>
      <td style="padding: 9px 10px; text-align: center;">0.9226 &plusmn; 0.0045</td>
      <td style="padding: 9px 10px; text-align: center;">0.8357</td>
      <td style="padding: 9px 10px; text-align: center;">0.5062</td>
      <td style="padding: 9px 10px; text-align: center;">78.1%</td>
      <td style="padding: 9px 10px; text-align: center;">71.5%</td>
      <td style="padding: 9px 12px; color: #64748b; font-size: 11.5px;">Replicates <em>Zoabi et al.</em> baseline; interpretable clinical rules.</td>
    </tr>
    <tr style="background-color: #f8fafc; border-bottom: 1px solid #e2e8f0;">
      <td style="padding: 9px 12px; font-weight: 600; color: #334155;">Random Forest (100 Trees)</td>
      <td style="padding: 9px 8px; text-align: center;">8 Features</td>
      <td style="padding: 9px 10px; text-align: center;">0.9348 &plusmn; 0.0052</td>
      <td style="padding: 9px 10px; text-align: center;">0.8412</td>
      <td style="padding: 9px 10px; text-align: center;">0.5374</td>
      <td style="padding: 9px 10px; text-align: center;">80.5%</td>
      <td style="padding: 9px 10px; text-align: center;">68.4%</td>
      <td style="padding: 9px 12px; color: #64748b; font-size: 11.5px;">Parallel variance reduction; stable non-linear partitioning.</td>
    </tr>
    <tr style="background-color: #f0fdf4; border-bottom: 1px solid #bbf7d0;">
      <td style="padding: 9px 12px; font-weight: 700; color: #166534;">LightGBM Champion (Paper Params)</td>
      <td style="padding: 9px 8px; text-align: center; font-weight: 600;">8 Features</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">0.9351 &plusmn; 0.0025</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">0.8443</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">0.5448</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">81.2%</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">68.0%</td>
      <td style="padding: 9px 12px; color: #14532d; font-size: 11.5px; font-weight: 600;">Replicates <em>Zoabi et al.</em> (0.84–0.86); fast leaf-wise boosting.</td>
    </tr>
    <tr style="background-color: #f0fdf4; border-bottom: 1px solid #bbf7d0;">
      <td style="padding: 9px 12px; font-weight: 700; color: #166534;">LightGBM Symptom-Only (No Age)</td>
      <td style="padding: 9px 8px; text-align: center; font-weight: 600;">7 Features</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">0.9333 &plusmn; 0.0022</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">0.8448</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">0.5403</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">81.5%</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">67.8%</td>
      <td style="padding: 9px 12px; color: #14532d; font-size: 11.5px; font-weight: 600;">Resilient model: survives April 15 regulatory age privacy redaction.</td>
    </tr>
    <tr style="background-color: #f0fdf4;">
      <td style="padding: 9px 12px; font-weight: 700; color: #166534;">XGBoost Benchmark (max_depth=4)</td>
      <td style="padding: 9px 8px; text-align: center; font-weight: 600;">8 Features</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">0.9350 &plusmn; 0.0025</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">0.8448</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">0.5480</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">81.4%</td>
      <td style="padding: 9px 10px; text-align: center; font-weight: 700; color: #166534;">68.1%</td>
      <td style="padding: 9px 12px; color: #14532d; font-size: 11.5px; font-weight: 600;">Confirms boosting engine invariance; depth-wise tree growth.</td>
    </tr>
  </tbody>
</table>
</div>

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
  *Discovery:* Proved that the default machine learning cutoff of $	au = 0.50$ is clinically unsafe, missing 567 infected patients (74.9% sensitivity). Calibrated the triage cutoff to $	au = 0.20$ (81.2% sensitivity), mathematically capturing 137 additional infected patients for PCR swab confirmation.
