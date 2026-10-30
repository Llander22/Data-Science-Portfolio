[![Companion Workbook in Colab](https://img.shields.io/badge/Colab-Open_Workbook-orange?logo=googlecolab)](https://colab.research.google.com/github/Llander22/Data-Science-Portfolio/blob/main/CS_L7_M10_Hackathon/M10-02-02%20covid19_hackathon.ipynb)
[![Workbook PDF in Google Drive](https://img.shields.io/badge/Google_Drive-Workbook_PDF-34A853?logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1d3qpg47eca1DAK13S7Id_SMwCU_OfmSm/view?usp=drive_link)

# Decision Pathway: COVID-19 Clinical Triage & Population Screening
### Operational Standard: CRISP-DM & Machine Learning Engineering Lifecycle

> **Operational Standard:** Cross-Industry Standard Process for Data Mining (CRISP-DM) & MLOps Engineering Lifecycle  
> **Technical Implementation:** `M10-02-02 covid19_hackathon.ipynb`  
> **Benchmark Reference:** Zoabi, Y., Deri-Rozov, S. & Shomron, N. *Machine learning-based prediction of COVID-19 diagnosis based on symptoms.* **npj Digital Medicine** 4, 3 (2021). *(Extended out-of-time prospective evaluation on 8–14 April 2020).*  
> **Lifecycle Topology:** 6 Canonical CRISP-DM Phases + Post-Deployment Feedback Loop (Selected Path: Green, Rejected: Red, Alternative: Slate).  
> **Project Context & Working Approach:** Conceived and prototyped during an intensive **2-day collaborative hackathon**, followed by structured human-AI pair programming refinement cycles to audit demographic fairness, resolve operational edge cases, and address peer review challenges.  
> **AI Tools Used:** Gemini 3.8 Flash (High) in the Google Antigravity IDE (coding assistant); Claude Opus 5.5 by Anthropic, medium effort (independent post-hackathon review of the workbook and pathway against the source data). All analytical decisions and final wording are the author's own; every figure traces to re-run notebook code.

---

### 📖 Narrative Workbook vs. Engineering Lifecycle Crosswalk

<table width="100%" style="border-collapse: collapse; margin-top: 4px;">
<tr>
<td width="50%" valign="top" style="border: 1px solid #cbd5e1; padding: 6px 10px; background: #ffffff;">
<p style="margin: 0 0 2px 0; font-size: 11px; font-weight: 700; color: #1e3a8a;">🔷 Stage 1: Clinical Problem Formulation (Phase 1: Business Understanding)</p>
<p style="margin: 0 0 6px 0; font-size: 10px; color: #475569;">Binary triage risk under acute PCR scarcity; 30-second intake; chronological arrow of time.</p>

<p style="margin: 0 0 2px 0; font-size: 11px; font-weight: 700; color: #1e3a8a;">🔷 Stage 2: Data Ingestion & Quality Auditing (Phase 2: Data Understanding)</p>
<p style="margin: 0 0 6px 0; font-size: 10px; color: #475569;">278,848 tests; 94.7% vs 5.3% imbalance; unmasked accuracy fallacy; April 15 privacy cliff.</p>

<p style="margin: 0 0 2px 0; font-size: 11px; font-weight: 700; color: #1e3a8a;">🔷 Stage 3: Preprocessing & Pipeline Architecture (Phase 3: Data Preparation)</p>
<p style="margin: 0 0 2px 0; font-size: 10px; color: #475569;">Filtered inconclusive records (n = 255,668); zero-leakage Scikit-Learn Pipeline + prospective split.</p>
</td>
<td width="50%" valign="top" style="border: 1px solid #cbd5e1; padding: 6px 10px; background: #ffffff;">
<p style="margin: 0 0 2px 0; font-size: 11px; font-weight: 700; color: #1e3a8a;">🔷 Stage 4: Modelling & Boosting Benchmarks (Phase 4: Modeling)</p>
<p style="margin: 0 0 6px 0; font-size: 10px; color: #475569;">LightGBM Champion (0.8443 AUC / 0.5441 PR-AUC); XGBoost benchmark; 5-fold CV baseline.</p>

<p style="margin: 0 0 2px 0; font-size: 11px; font-weight: 700; color: #1e3a8a;">🔷 Stage 5: Explainability & Demographic Fairness (Phase 5: Evaluation)</p>
<p style="margin: 0 0 6px 0; font-size: 10px; color: #475569;">TreeSHAP feature ranking; geriatric sensitivity deficit audit; demographically blind ablation (0.8400 AUC).</p>

<p style="margin: 0 0 2px 0; font-size: 11px; font-weight: 700; color: #1e3a8a;">🔷 Stage 6: Operational Thresholding & Deployment (Phase 6: Deployment)</p>
<p style="margin: 0 0 6px 0; font-size: 10px; color: #475569;">Cutoff $\tau = 0.20$ (81.2% sensitivity; 23,829 swabs spared); 1-line frontline screening heuristic.</p>

<p style="margin: 0 0 2px 0; font-size: 11px; font-weight: 700; color: #1e3a8a;">🔄 CRISP-DM Iteration 2: Post-Hackathon Feedback Loop</p>
<p style="margin: 0 0 2px 0; font-size: 10px; color: #475569;">Brier calibration; minimal 3-feature intake (0.8248 AUC); 210-fit grid search & decision invariance.</p>
</td>
</tr>
</table>

---

## Stage-Aligned Decision Architecture & Ledger (CRISP-DM / MLOps Lifecycle)

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
      <h3 style="margin: 0 0 4px 0; color: #1a73e8;">🔷 STAGE 1: Clinical Problem Formulation & Triage Mandate</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 1: Clinical Problem Formulation & Operational Framing (CRISP-DM Phase 1: Business Understanding)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>Calibrated Binary Probability Classifier</b> — Estimating continuous infection risk <i>P(y=1|X) &isin; [0, 1]</i> from 8 self-reported indicators accessible in under 30 seconds to prioritise RT-PCR laboratory swabs.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Multiclass Categorisation</b> — Unnecessary clinical overhead when the immediate operational decision during testing scarcity is strictly binary (administer PCR swab vs. isolate safely at home).</li>
        <li style="margin-bottom: 8px;">⚪ <b>Alternative Option:</b> <b>Survival / Longitudinal Analysis</b> — Unfeasible due to absence of symptom onset dates in nationwide testing registry.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Generating continuous probabilities allows hospital systems to dynamically tune the decision cutoff $\tau$ based on daily swab supply and hospital bed capacity.</li>
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
      <h3 style="margin: 0 0 4px 0; color: #1a73e8;">🔷 STAGE 2: Data Ingestion & Quality Auditing</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 2: Ingestion & Epidemiological EDA (CRISP-DM Phase 2: Data Understanding)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>Strict Ground Truth Filtration (<i>n</i> = 255,668)</b> — Drop inconclusive <code>other</code> outcomes (3,892) and missing gender records (19,045 among positive/negative cases, &lt;7%) to guarantee clinical label fidelity for benchmark comparison. Acknowledges that 53.3% of the clean cohort (n = 136,294, 17/22 March–14 April) forms the core modelling window.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Synthetic Imputation of Missing Gender</b> — Distorts demographic distributions without adding predictive power; dataset size (&gt;255,000 verified cases) provides massive statistical power without synthetic risks.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Removing ambiguous outcomes preserves pristine ground-truth labels without sacrificing statistical power, retaining 91.7% of the original nationwide cohort. Unmasked the <b>April 15 privacy redaction</b>, where age was 100% redacted mid-epidemic. <i>(Note: While missing gender records are excluded here to align with the extended benchmark evaluation, Stage 5 evaluates a Demographically Blind model that safely triages incomplete records in live deployment.)</i></li>
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
      <h3 style="margin: 0 0 4px 0; color: #1a73e8;">🔷 STAGE 3: Preprocessing & Pipeline Architecture</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 2.3–2.4 & Phase 3: Zero-Leakage Pipeline (CRISP-DM Phase 3: Data Preparation)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>Encapsulated <code>Pipeline</code> + Prospective Temporal Partitioning</b> — The demographic ablations and the deployed Pure Clinical model wrap a <code>ColumnTransformer</code> (<code>OneHotEncoder(handle_unknown='ignore')</code> + symptom passthrough) inside a Scikit-Learn <code>Pipeline</code> fit exclusively on historical training data (March 22 – April 7, <i>n</i>=98,989); the Phase 3 benchmarks use fixed binary indicators, which have no fitted vocabulary to leak. Tested strictly on prospective hold-out (April 8–14, <i>n</i>=37,305).</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Random Shuffling (<code>train_test_split(shuffle=True)</code>)</b> — Disastrous look-ahead leakage across evolving epidemic waves; artificially inflates test metrics by predicting the past using future epidemic data and smoothing away real prevalence decay (8.45% down to 6.06%).</li>
        <li style="margin-bottom: 8px;">⚪ <b>Alternative Option:</b> <b>Global In-Place Transformation</b> — Leaks future category levels to training and causes training-serving skew in production deployment.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Respects the epidemiological &ldquo;Arrow of Time&rdquo; and MLOps encapsulation standards. While <code>test_date</code> is never fed to the model as an input feature, it is mandatory as the structural validation spine to simulate authentic weekly retraining and ensure genuine, leak-free prospective discrimination (0.8443 ROC-AUC) on unseen future patients.</li>
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
      <h3 style="margin: 0 0 4px 0; color: #1a73e8;">🔷 STAGE 4: Model Architecture & Boosting Benchmarks</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 3: Model Architecture Benchmarks & Phase 4: XGBoost (CRISP-DM Phase 4: Modeling)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>LightGBM Champion (<code>num_leaves=20</code>, <code>min_child_samples=4</code>, <code>learning_rate=0.05</code>, <code>is_unbalance=True</code>) & XGBoost Benchmark</b> — Fast histogram-based gradient boosting achieving ROC-AUC <b>0.8443</b> (PR-AUC 0.5441) and XGBoost ROC-AUC <b>0.8449</b> (PR-AUC 0.5496), verified via internal 5-fold CV (0.9351 &plusmn; 0.0025). Crucially, at tau = 0.20, Random Forest, XGBoost and both LightGBMs reach exactly the same operating point (81.2% sensitivity / 68.0% specificity), with the single Decision Tree close behind (80.4% / 68.0%); for the deployed Pure Clinical model this operating point is verified patient-by-patient to be the 1-line rule. LightGBM is retained for pipeline continuity, while Logistic Regression demonstrates comparable discrimination (0.8336 AUC) with superior probability calibration (Brier 0.0401). Robust <b>Symptom-Only LightGBM (Ablating Age)</b> achieves ROC-AUC <b>0.8448</b> (PR-AUC 0.5401).</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Deep Neural Networks</b> — Excessive compute overhead, slow inference, and inferior tabular performance compared to tree gradient boosting.</li>
        <li style="margin-bottom: 8px;">⚪ <b>Alternative Option:</b> <b>Direct Test-Set Tuning (Test Snooping)</b> — Tuning against the test set leaks future outcomes; validation must occur strictly within training folds.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Tree ensembles capture non-linear symptom co-occurrences without feature scaling, while <code>is_unbalance=True</code> mathematically corrects gradient updates for the 10.8:1 training imbalance. Standard Random Forest provides the highest hold-out discrimination (0.8451 ROC-AUC); however, LightGBM was selected for primary pipeline continuity due to native class-weighting and gradient boosting efficiency.</li>
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
      <h3 style="margin: 0 0 4px 0; color: #1a73e8;">🔷 STAGE 5: Clinical Explainability & Demographic Fairness</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 4: Advanced Modelling & Interpretability (CRISP-DM Phase 5: Evaluation)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>TreeSHAP Explainability & Subgroup Demographic Fairness Audit</b> — Exact game-theoretic Shapley values reveal patient risk log-odds (<code>Contact with confirmed</code> and <code>Fever</code> dominate). Subgroup auditing on the deployed Pure Clinical model reveals subtle demographic variation: gender parity (Male AUC 0.8255 vs. Female AUC 0.8554; 80.2% vs 82.2% sensitivity), but exposes a critical 10.2 percentage point geriatric sensitivity deficit for older adults (&lt;60 AUC 0.8414 with 83.0% sensitivity vs. 60+ AUC 0.8251 with 72.8% sensitivity at tau = 0.20, missing over 1 in 4 infected seniors: 109 of 400; 27.2% of infected seniors reported zero symptoms and indication 'Other', falling below the rule cutoff). Evaluates an age-60+ bypass as mitigation: it catches all 109 missed seniors (overall sensitivity 86.0%) but adds 6,821 swabs (+52%, 1.6% yield), so adoption is a service-level policy decision.</li>
        <li style="margin-bottom: 8px;">🟢 <b>Deployment Candidate Switch:</b> <b>Pure Clinical Demographically Blind Pipeline (ROC-AUC = 0.8400, PR-AUC = 0.5333)</b> — Formally pivot from the 8-feature benchmark to the Pure Clinical model for production deployment. Omitting age and gender drops only 0.0051 AUC vs benchmark, guarantees zero pipeline crashes when age is redacted mid-epidemic, and eliminates triage drop-out for the 19,045 patients with unrecorded demographics.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Aggregate-Only Metric Reporting</b> — Violates healthcare ethics by obscuring disparate clinical impact across vulnerable protected groups.</li>
        <li style="margin-bottom: 8px;">⚪ <b>Alternative Option:</b> <b>Permutation Importance</b> — Global ranking only, lacking individual-patient attribution.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Transparent attribution is essential for clinical adoption and regulatory compliance (FDA/NHS AI guidelines). Demographically blind ablation proves that frontline triage can operate ethically without demographic dependency.</li>
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
      <h3 style="margin: 0 0 4px 0; color: #1a73e8;">🔷 STAGE 6: Operational Decision Thresholding & Clinical Deployment</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📖 <b>Workbook Mapping:</b> Phase 5: Operational Decision Thresholding & Closing Summary (CRISP-DM Phase 6: Deployment)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">🟢 <b>Selected Option:</b> <b>Operational Clinical Cutoff ($\tau$ = 0.20) & Rule Distillation</b> — Elevates sensitivity to <b>81.16%</b> (capturing 1,835 of 2,261 infected patients), rescuing <b>221</b> infected individuals who would be missed under default $\tau$ = 0.50 (missed cuts from 647 to 426), delivering a <b>98.24% Negative Predictive Value (NPV)</b>. Spares 23,829 laboratory swabs on uninfected patients (24,255 sent home total). Analysis of score distributions reveals that zero patients fall between 0.11 and 0.36; setting $\tau$ = 0.20 distills the tree model into an ultra-transparent 1-line rule: swab if any symptom is present or test indication is Contact/Abroad.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Default $\tau$ = 0.50 Cutoff</b> — Implicitly assumes equal error costs; misses 647 infected patients (71.38% sensitivity), driving silent community transmission.</li>
        <li style="margin-bottom: 8px;">⚪ <b>Alternative Option:</b> <b>Emergency Cutoff ($\tau$ = 0.10)</b> — Captures 100% of infected patients (0 missed), but triggers 35,044 false alarms, swabbing 100% of patients and completely defeating triage purpose.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> In epidemic triage, the clinical cost of a False Negative far exceeds that of a False Positive. Setting $\tau$ = 0.20 provides high sensitivity while operating as an interpretable protocol executable even during hospital EHR downtime.</li>
      </ul>
    </td>
  </tr>
</table>

---

---
<table width="100%" border="0" cellpadding="0" cellspacing="14">
  <tr>
    <td width="42%" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/Llander22/Data-Science-Portfolio/main/CS_L7_M10_Hackathon/images/pathway_iteration_2_flowchart.png" width="100%" alt="Iteration 2 Flowchart" style="border: 1px solid #dadce0; border-radius: 6px; padding: 6px; background: #ffffff;">
    </td>
    <td width="58%" valign="top">
      <h3 style="margin: 0 0 4px 0; color: #1a73e8;">🔄 CRISP-DM ITERATION 2: Post-Hackathon Synthesis, Calibration & Prevalence Drift</h3>
      <p style="margin: 4px 0 12px 0; font-size: 13px; color: #5f6368;">📂 <b>Workbook Mapping:</b> Section 6: Further Improvement following the Hackathon Presentations (Continuous Feedback Loop)</p>
      <ul style="padding-left: 20px; margin: 8px 0; line-height: 1.55;">
        <li style="margin-bottom: 8px;">✅ <b>Selected Option:</b> <b>Dual Calibration & Conservative Operating Point Audit</b> — Evaluates probability calibration via Brier score on the deployed Pure Clinical Model. Pure clinical Logistic Regression achieves <b>0.0401</b> (Brier skill score +0.29 against the 0.0569 base-rate reference); LightGBM achieves <b>0.1271</b> (skill score −1.23, worse than the base rate) because positive-class weighting (<code>is_unbalance=True</code>) inflates every score without changing the patient ranking; sensitivity is set by the threshold, not the weighting. Evaluates conservative operating cutoffs: $\tau$ = 0.40 catches 77.22% of infections at 77.19% specificity (swabbing 26.1% of attendees); $\tau$ = 0.50 catches 71.38% of infections at 84.47% specificity (swabbing 18.9% of attendees). Models posterior PPV across waves (10% surge &rarr; ~22% PPV vs 1% lull &rarr; ~2.5% PPV) while proving that threshold shifts within the 0.12–0.36 score gap alter zero patient decisions.</li>
        <li style="margin-bottom: 8px;">✅ <b>Selected Option:</b> <b>Minimal Clinical Intake Ablation (3 Core Features)</b> — Stripping secondary symptoms (<code>head_ache</code>, <code>sore_throat</code>, <code>shortness_of_breath</code>) leaves only 3 core clinical features (<code>cough</code>, <code>fever</code>, <code>test_indication</code>). Prospective AUROC slips from 0.8400 to <b>0.8248</b> (delta 0.0152), retaining <b>95.5%</b> of discrimination above random chance (98.2% of total AUC). Enables an ultra-rapid 15-second frontline intake with zero demographic questions and zero missing-data risk.</li>
        <li style="margin-bottom: 8px;">✅ <b>Selected Option:</b> <b>Grid Search to Check Whether Tuning Matters (Model Complexity Audit)</b> — After peer review, we ran a 5-fold grid search (210 model fits) on the 17 March to 7 April training data. It covered LightGBM, Random Forest and the deployed Pure Clinical model, and each tuned model was compared with its untuned version. Training CV AUC moved by only +0.0001 to +0.0005, well inside the &plusmn;0.0025 fold spread. Hold-out AUC moved by &minus;0.0009 to &minus;0.0039, so tuning did not help. Crucially, at &tau; = 0.20 every tuned model gave the same sensitivity (81.16%), specificity (68.00%) and missed infections (426 of 2,261). LightGBM, Random Forest and Pure Clinical each made the same decision as their untuned versions for every patient (0 disagreements). The Pure Clinical pair also matched on the 15&ndash;30 April cohort (825 of 2,906 missed).</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Adopting the Grid-Search "Best" Settings</b> — LightGBM with 31 leaves and min_child 10, Random Forest with depth 12, and Pure Clinical with depth 8 gave no meaningful gain and identical patient decisions. Changing a published, working configuration for no benefit adds maintenance and monitoring work.</li>
        <li style="margin-bottom: 8px;">🔴 <b>Rejected Option:</b> <b>Blind Threshold Shifting Within the Score Gap</b> — Attempting to fine-tune decision thresholds between 0.12 and 0.36 alters zero patient outcomes due to the empty score distribution band.</li>
        <li style="margin-bottom: 0;">🛡️ <b>Defensibility Rationale:</b> Acknowledges documentation ascertainment bias on secondary symptoms (recorded primarily for high-suspicion patients). Grounding triage in discrete score distributions and longitudinal sensitivity drift surveillance guarantees operational safety.</li>
      </ul>
    </td>
  </tr>
</table>

## Quantitative Summary & Benchmark Scorecard

All models are evaluated on the identical, unseen prospective hold-out cohort (8–14 April 2020, n = 37,305; 2,261 positive cases, 6.06% prevalence, random seed = 5):

| Model Architecture & Configuration | Features | Train 5-Fold CV AUC | Hold-out ROC-AUC | Hold-out PR-AUC | Sensitivity (tau=0.20) | Specificity (tau=0.20) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Rule Dummy Baseline** | None | 0.5000 | 0.5000 | 0.0606 | 0.0% | 100.0% |
| **Logistic Regression (Linear)** | 8 | 0.9280 ± 0.0030 | 0.8336 | 0.4809 | 81.5% | 64.3% |
| **Decision Tree Baseline (`depth=5`)** | 8 | 0.9184 ± 0.0023 | 0.8302 | 0.4858 | 80.4% | 68.0% |
| **Random Forest (100 Trees, Bagging)** | 8 | 0.9347 ± 0.0025 | **0.8451** | 0.5419 | 81.2% | 68.0% |
| **LightGBM Champion (Paper Params)** | 8 | **0.9351 ± 0.0025** | 0.8443 | **0.5441** | **81.2%** | **68.0%** |
| **LightGBM Symptom-Only (No Age)** | 7 | 0.9333 ± 0.0022 | 0.8448 | 0.5401 | 81.2% | 68.0% |
| **XGBoost Benchmark (`max_depth=4`)** | 8 | 0.9348 ± 0.0026 | 0.8449 | 0.5496 | 81.2% | 68.0% |
| **LightGBM Pure Clinical (Deployed Policy)** | 6 | 0.9268 ± 0.0024 | 0.8400 | 0.5333 | **81.2%** | **68.0%** |
| **Minimal Clinical (3 Features: Rapid Intake)** | 3 | 0.9062 ± 0.0029 | 0.8248 | 0.4219 | 79.9% | 68.0% |

#### 🏥 Clinical Operational Assessments:
* **Zero-Rule Dummy Baseline:** 94.71% accuracy but misses 100% of sick patients (illustrates the Accuracy Fallacy).
* **Logistic Regression (Linear):** Strong linear baseline (0.8336 AUC); unweighted probabilities yield an excellent Brier score (0.0401) but miss non-linear symptom co-occurrences.
* **Decision Tree Baseline:** Interpretable depth-5 tree (0.8302 AUC; our choice, not taken from *Zoabi et al.*); provides transparent clinical splits.
* **Random Forest:** Strongest hold-out discrimination (0.8451 AUC). Remarkably, at tau = 0.20, Random Forest, XGBoost and LightGBM all reach the same operating point (81.2% sensitivity / 68.0% specificity), with the Decision Tree close behind (80.4% / 68.0%): the tree ensembles effectively collapse to the same 1-line rule.
* **LightGBM Champion:** Extended prospective evaluation of *Zoabi et al.* parameters (0.8443 AUC); fast histogram-based boosting with positive class re-weighting.
* **LightGBM Symptom-Only:** Resilient model that retains full discrimination (0.8448 AUC) after the April 15 age privacy redaction.
* **LightGBM Pure Clinical (Deployed Policy):** Our production model. Demographically blind pipeline achieving **0.8400 AUC** (PR-AUC 0.5331), enabling safe triage of all patients with unrecorded demographics without EHR crashes.
* **Minimal Clinical (3 Features - Rapid Intake):** Stripped-down frontline intake model using only cough, fever, and test indication. Achieves **0.8248 AUC** (PR-AUC 0.4219, retaining 95.5% of discrimination above chance; 79.9% sensitivity and 68.0% specificity at tau = 0.20), enabling a 15-second intake.
* **XGBoost Benchmark:** Confirms boosting engine invariance (0.8449 AUC).

---

## Engineering Analysis: Key Methodological Discoveries

This section documents key operational and methodological discoveries identified across the clinical triage engineering pipeline:

* **The Scikit-Learn Pipeline & Zero-Leakage Architecture (Stage 3 & 4):**  
  *Discovery:* Wrapping the ablation and deployed Pure Clinical models in an encapsulated `Pipeline` + `ColumnTransformer` (`OneHotEncoder(handle_unknown='ignore')`) guarantees that all transformers are fitted strictly on `X_train` and that unseen categories cannot crash scoring, giving production serving parity when new patient records arrive at a clinical API. (The Phase 3 benchmarks use fixed binary indicators, so they have no fitted encoder to leak.)
* **The April 15 Regulatory Privacy Redaction (Stage 2 & 4):**  
  *Discovery:* Auditing daily testing volume revealed why `age_60_and_above` became 100% missing after April 14—health authorities intervened mid-epidemic to redact age (presumably to safeguard citizen privacy). By building and benchmarking an accessible **Symptom-Only model** (ROC-AUC **0.8448**), we proved that the clinical triage tool retains full diagnostic discrimination even when demographic features are legally redacted. Furthermore, out-of-time longitudinal evaluation on the subsequent 119,374 patients from 15–30 April confirmed that the Pure Clinical model weathered the regulatory privacy cliff with zero imputation (0.8278 ROC-AUC). Crucially, this test exposed critical sensitivity drift: as prevalence fell from 6.06% to 2.43%, sensitivity collapsed from 81.2% down to 71.6% (missing 825 of 2,906 infections). A case-mix decomposition explains why: asymptomatic attendees rose from 69.4% to 86.3%, and the share of infected patients falling in the rule's home group (no symptoms + indication 'Other') rose from 18.8% to 28.4%, which is exactly the rule's miss rate. Stable ROC-AUC hid this, demonstrating why monitoring static ROC-AUC alone is a dangerous trap for healthcare MLOps.
* **The "Arrow of Time" Prospective Partitioning (Stage 3):**  
  *Discovery:* Highlighted that standard cross-sectional random shuffling (`train_test_split(shuffle=True)`) would risk temporal look-ahead leakage across evolving epidemic waves by predicting past encounters with future data and obscuring genuine prevalence decay. Validated that testing on prospective future dates (April 8–14) simulates authentic hospital intake conditions and captures true out-of-time clinical performance.
* **Demographic Fairness & Demographically Blind Governance (Stage 5):**  
  *Discovery:* Audited models across protected demographics to satisfy healthcare AI ethics (NHS/WHO guidelines). Verified that the deployed Pure Clinical model maintains balanced gender discrimination (Male AUC 0.8255 vs. Female AUC 0.8554), but uncovered a serious 10.2 percentage point geriatric sensitivity deficit (72.8% in 60+ vs 83.0% in <60; missing 109 of 400 infected seniors). Because the model collapses to the 1-line symptom rule, the 27.2% of infected seniors who reported zero symptoms and presented with indication 'Other' are filtered out as negative. An age-60+ bypass rule would close the gap at a cost of 6,821 extra swabs (+52%) for 109 extra cases (1.6% yield); it needs age, which a clinic can still ask at intake even though the public dataset redacted it. Further proved that an entirely demographically blind model achieves 0.8400 AUC, eliminating patient drop-out for the 19,045 individuals with unrecorded demographic fields.
* **Asymmetric Cost-Sensitive Threshold Calibration (Stage 6 / Deployment):**  
  *Discovery:* Proved that the default machine learning cutoff of tau = 0.50 is clinically unsafe, missing 647 infected patients (71.38% sensitivity). Setting the operational cutoff to tau = 0.20 elevates sensitivity to 81.16%, capturing **221** additional infected patients for PCR swab confirmation while sparing 23,829 laboratory swabs on uninfected individuals (24,255 total sent home). Detailed inspection of model scores reveals that zero patients fall between 0.11 and 0.36; setting tau = 0.20 is mathematically equivalent to the transparent 1-line frontline heuristic: swab if any symptom is reported or test indication is Contact/Abroad.

* **Pure Clinical Calibration & Rapid 3-Question Intake (Iteration 2 / Post-Hackathon Loop):**  
  *Discovery:* Post-hackathon peer challenge investigations demonstrated two vital operational truths on our deployed Pure Clinical model: First, probability calibration confirmed that while pure clinical Logistic Regression delivers an excellent Brier score of **0.0401** (comfortably beating the 0.0569 base-rate baseline), LightGBM's Brier score of **0.1271** is worse than the base-rate reference (skill score −1.23): positive-class weighting (`is_unbalance=True`) inflates every score without changing the ranking, so it does not itself add safety. As an operational queue-ranking trigger (14.06% empirical PPV at $\tau = 0.20$), it works effectively. Second, feature pruning on the Pure Clinical pipeline proved that stripping secondary symptoms yields a Minimal Clinical model achieving **0.8248 AUC** (PR-AUC 0.4219; delta of 0.0152 vs 0.8400) -- retaining 95.5% of discrimination above random chance (79.9% sensitivity and 68.0% specificity at tau = 0.20). Zoabi et al. flag headache, sore throat and shortness of breath as reporting-biased (recorded more thoroughly for positive cases), and they appear in only about 1% of hold-out patients, which together explain why pruning them costs so little discrimination. Evaluating conservative cutoffs confirms that tau = 0.40 yields 77.2% sensitivity / 77.2% specificity (swabbing 26.1% of attendees) and tau = 0.50 yields 71.4% sensitivity / 84.5% specificity (swabbing 18.9% of attendees).

---

* **Hyperparameter Sensitivity & Model Complexity Invariance (Iteration 2 / Peer Review Challenge):**  
  *Discovery:* A peer review challenge asked whether we should have tuned hyperparameters before choosing a model. We ran a 5-fold grid search (210 model fits) on the 17 March to 7 April training data for LightGBM, Random Forest and the deployed Pure Clinical model. We compared each tuned model with its untuned version. Training CV AUC moved by only +0.0001 to +0.0005, well inside the $\pm 0.0025$ fold spread. Hold-out AUC moved by −0.0009 to −0.0039, so tuning did not help. In a sweep of LightGBM `num_leaves`, only the shallowest trees (4 leaves) lost anything in training CV (0.9319 against 0.9349–0.9352). At $\tau = 0.20$, every tuned model gave the same sensitivity (81.16%), specificity (68.00%) and missed infections (426 of 2,261). The tuned LightGBM, Random Forest and Pure Clinical models each made the same decision as their untuned versions for every patient (0 disagreements). The Pure Clinical pair also matched on the 15–30 April cohort (825 of 2,906 missed). The inputs are yes/no flags plus a three-value test indication, so there are at most 384 patient profiles (96 for the deployed model). Once the models learn the main symptom combinations, tuning only shifts probability decimals. One caveat: the grid search shuffles days together instead of splitting by date, so the training scores run optimistic. Every model is scored the same way, so the comparison stays fair.

---

## Study Limitations & Threats to Validity

* **Retrospective design:** Models were trained only on data available before the test window.
* **Choices made after seeing results:** The decision threshold, model selection and age bypass were set after viewing the 8–14 April results. Those figures may be optimistic.
* **Most independent estimate:** The 15–30 April cohort was evaluated after all choices were fixed. It gives the most independent estimate.
* **Symptom recording bias:** Symptom data may have been recorded more completely for confirmed cases. That would inflate performance for all models, including the published benchmark.

---

## Addendum: Real-World Clinical Deployment in 2026

Deploying an epidemiological machine learning model into a live 2026 healthcare ecosystem requires transitioning from static retrospective validation to continuous, safety-critical MLOps governance:

<table width="100%" style="border-collapse: collapse; margin-top: 4px;">
<tr>
<th width="50%" style="background: #1e3a8a; color: #ffffff; padding: 6px 10px; text-align: left; font-size: 12px;">🏛️ Governance, Drift Monitoring & Architecture</th>
<th width="50%" style="background: #1e3a8a; color: #ffffff; padding: 6px 10px; text-align: left; font-size: 12px;">🏥 Clinical Operations, Delayed Feedback & Human-in-the-Loop</th>
</tr>
<tr>
<td width="50%" valign="top" style="padding: 6px 8px; border: 1px solid #cbd5e1; font-size: 11px; line-height: 1.5; vertical-align: top;">
<h4 style="margin: 0 0 4px 0;">📋 1. CRISP-DM to Mitchell et al. Model Cards (arXiv:1810.03993)</h4>
<p style="margin: 0 0 4px 0;">CRISP-DM standardises the engineering workflow (data preparation to threshold tuning). However, clinical deployment requires a standardised governance passport: the <b>Model Card</b>. Essential for UK MHRA Software as a Medical Device (SaMD) and EU AI Act high-risk compliance, the Model Card explicitly records intended use (gatekeeper triage acknowledging ~69% asymptomatic baseline), the 10.2 percentage point geriatric sensitivity deficit (72.8% in 60+ vs 83.0% in <60) with the cost of its candidate age-bypass mitigation (+52% swabs, 1.6% yield), post-peak sensitivity drift (81.2% -> 71.6%), and demographic ablation resilience (0.8400 AUC without age/gender).</p>
<h4 style="margin: 0 0 4px 0;">📈 2. Covariate Shift & Variant Drift Monitoring</h4>
<p style="margin: 0 0 4px 0;">Pathogens mutate and symptoms evolve across epidemic waves (e.g. anosmia in wild-type vs. upper respiratory symptoms in Omicron). The 2026 production pipeline integrates automated drift surveillance (Evidently AI / Great Expectations). It tracks the <b>Population Stability Index (PSI)</b> weekly across incoming symptoms; any shift exceeding <code>PSI &gt; 0.20</code> triggers clinical review and dataset refreshing before silent diagnostic decay occurs.</p>
<h4 style="margin: 0 0 4px 0;">⚡ 3. Sub-5ms Inference Serving & EHR Integration</h4>
<p style="margin: 0;">Clinical triage cannot tolerate server latency. The pipeline is containerised as a lightweight microservice (FastAPI + ONNX Runtime). Scoring our 6 pure clinical features (or 3 core questions in a rapid intake) executes in under <b>5 milliseconds</b> per patient, requiring zero demographic queries and completely eliminating EHR runtime crashes caused by missing age or gender fields. Embeds seamlessly into hospital Electronic Health Records (Epic, Cerner) and national intake systems (NHS 111) via HL7/FHIR standards.</p>
</td>
<td width="50%" valign="top" style="padding: 6px 8px; border: 1px solid #cbd5e1; font-size: 11px; line-height: 1.5; vertical-align: top;">
<h4 style="margin: 0 0 4px 0;">🧪 4. Delayed Ground-Truth Labels & Canary Deployments</h4>
<p style="margin: 0 0 4px 0;">RT-PCR swabs take 24–72 hours to return from regional laboratories, creating an unavoidable delayed feedback loop. Production MLOps reconciles intake predictions asynchronously against incoming laboratory LIMS outcomes. Candidate retrained models run on a <b>Champion-Challenger (90/10) canary split</b> and are only promoted after demonstrating superior prospective sensitivity across two weeks of verified laboratory tests.</p>
<h4 style="margin: 0 0 4px 0;">🛡️ 5. Calibrated Triage Policy & Clinician Override Auditing</h4>
<p style="margin: 0 0 4px 0;">The calibrated $\tau$ = 0.20 cutoff anchors a clear, transparent clinical triage policy:<br>&bull; <b>Home Monitoring (&lt; 0.20):</b> Self-care guidance, digital symptom check-ins; laboratory swabs safely spared.<br>&bull; <b>PCR Swab Referral (&ge; 0.20):</b> Priority RT-PCR swabbing and rapid isolation precautions.<br>&bull; <b>Clinician Override Governance:</b> Frontline clinicians retain full discretion to order swabs for patients scoring &lt;0.20 who exhibit severe clinical distress. All clinician overrides are logged and audited weekly to monitor evolving symptom patterns and avoid alert fatigue.</p>
</td>
</tr>
</table>

---

### Working Approach & Companion Deliverables

Conceived and prototyped during an intensive 2-day hackathon, this project was subsequently refined through structured human-AI pair-programming cycles to audit demographic fairness, resolve operational edge cases, and address peer review challenges. <b>AI tools:</b> Gemini 3.8 Flash (High) in Google Antigravity as coding assistant; Claude Opus 5.5 (Anthropic, medium effort) as independent reviewer, cross-checking both deliverables against the source dataset. Companion project deliverables:

* 📓 **Interactive Technical Workbook (Colab):** [Open in Google Colab](https://colab.research.google.com/github/Llander22/Data-Science-Portfolio/blob/main/CS_L7_M10_Hackathon/M10-02-02%20covid19_hackathon.ipynb)
* 📄 **Master Technical Workbook (PDF):** [View on Google Drive](https://drive.google.com/file/d/1d3qpg47eca1DAK13S7Id_SMwCU_OfmSm/view?usp=drive_link)

