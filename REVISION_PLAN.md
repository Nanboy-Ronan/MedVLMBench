# MedVLMBench Major Revision Plan

This plan maps the reviewer requests to code, analyses, experiments, manuscript changes, and information
that must be confirmed by the authors. It treats the submitted manuscript and decision letter as source
material only.

## Highest priority changes

1. Rename the comparison groups to **general-domain VLMs** and **medical-domain VLMs**. Reserve
   **specialty-specific VLMs** for models trained for one specialty. The current pool contains PLIP as a
   specialty-specific pathology model, but most medical models are broad medical-domain models.
2. Make matched family comparisons the primary inferential evidence. Describe cross-family maxima and
   best-model lines as exploratory oracle comparisons and identify the selected model in every panel.
3. Add symmetric off-the-shelf and adapted comparisons for both model types. State explicitly that the
   original RQ2 is a deployment comparison between adapted general-domain and off-the-shelf medical-domain
   models, rather than an estimate of the independent effect of medical pretraining.
4. Separate dataset-level metric bootstrapping from LMM inference. The 1,000 item bootstrap replicates
   estimate uncertainty for each model-dataset metric. REML LMM coefficients use Wald confidence intervals
   and two-sided Wald P values; the LMM is not itself bootstrapped in the current implementation.
5. Add learning curves and direct resource measurements before retaining cost, efficiency, or
   resource-constrained deployment claims.

## Code and analysis now available

- `run_train.py` supports `--train_fraction` and `--fraction_seed`. Each fractional run has a unique output
  directory and saves the exact selected indices in `train_subset_manifest.json`.
- `analysis/run_learning_curve.py` runs a grid of fractions and seeds through the standard training entrypoint.
- Training and evaluation write `experiment_manifest.json` with the checkpoint, full arguments, dataset size,
  total and trainable parameters, hardware, wall time, and peak GPU memory. Multi-GPU evaluation merges its
  worker manifests into one run-level manifest.
- `metadata/model_pairs.csv` records size, architecture, generation, and strict-match decisions.
- `metadata/model_dataset_exposure.csv` distinguishes documented direct exposure, modality-level exposure,
  no reported exposure, and uncertain exposure. MedGemma, Lingshu, MedSigLIP, BioMedCLIP, and LLaVA-Med
  combinations are conservatively marked uncertain where dataset-level exposure cannot be established.
- `analysis/revision_sensitivity.py` excludes proprietary o3 and Gemini baselines from inference, fits model
  type and adaptation status jointly, reports symmetric adapted comparisons, estimates strict-pair differences,
  removes documented direct exposure, and exports domain-drop summaries.

## Preliminary results from existing aggregate result files

These values are generated from the current repository CSVs and should be checked against item-level analyses
before entering the manuscript.

| Analysis | Diagnosis | VQA |
|---|---:|---:|
| Adaptation main effect | +0.263 (95% CI 0.215 to 0.312; P<0.001) | +0.083 (0.028 to 0.138; P=0.003) |
| General-domain versus medical-domain among adapted models | +0.023 (-0.029 to 0.075; P=0.388) | +0.049 (-0.006 to 0.104; P=0.084) |
| Strict matched-pair general-minus-medical OTS difference | +0.028 (-0.071 to 0.126; P=0.583) | -0.084 (-0.157 to -0.011; P=0.024) |
| General-domain versus medical-domain after excluding documented direct exposure | -0.027 (-0.105 to 0.050; P=0.492) | unchanged because no VQA pair is classified as documented direct exposure |

For diagnosis models with disclosed modality coverage, the adapted cross-domain minus in-domain mean changes
are negative for MedCLIP, PLIP, and PubMedCLIP. The generated `cross_domain_drops.csv` reports each adaptation
strategy separately. Broad medical models with incompletely disclosed corpora are excluded from this certain-
exposure summary.

Several variance components are near the boundary, and some full nested models require a simpler converged
random-effects structure. The diagnostics JSON records the optimizer, convergence status, variance components,
warnings, and every failed fit attempt. These limitations should be reported rather than hidden.

## Experiments to run

### Learning curves

Use fractions 1%, 5%, 10%, 25%, 50%, and 100% with at least three fraction seeds. Prioritize the strict pairs:

- CLIP versus PubMedCLIP and PLIP for diagnosis.
- LLaVA-1.5 versus LLaVA-Med and Qwen2.5-VL versus Lingshu for VQA.

Run at least one modality-aligned and one cross-domain dataset per pair. If compute allows, run every dataset.
Estimate the smallest labelled-data fraction at which the adapted general-domain model reaches the OTS
medical-domain model and report uncertainty across seeds.

### Resource analysis

Aggregate the generated manifests into a supplementary table with training samples, total and trainable
parameters, accelerator type and count, wall time, peak memory, and inference time per item. Report throughput
under the same batch size and precision. Do not compare proprietary API cost with local GPU cost unless the
pricing date and token/image accounting are fixed.

### Item-level VQA model

Use the saved per-item correctness values to fit a logistic mixed-effects model with fixed effects for model type,
adaptation, and their interaction, plus random intercepts for item/dataset and model/family where supported.
Keep the aggregate LMM as a sensitivity analysis. Resample items within each dataset for item-bootstrap metric
intervals; do not call the Wald LMM intervals bootstrap intervals.

### Validation-selected comparison

Replace test-set best-model lines with either a prespecified model per family or model selection on a validation
split. Keep test-selected maxima only as clearly labelled exploratory oracle results.

## Reviewer-specific manuscript actions

- **R1 comments 1-6 and R2 Methods:** add the model-pair table, exposure matrix, symmetric adapted results,
  learning curves, resource table, exact checkpoint IDs, preprocessing, prompts, answer extraction, LoRA settings,
  optimizer settings, epochs, validation rules, seeds, and deterministic decoding settings.
- **R1 comments 8-10:** state that o3 and Gemini are descriptive only; standardize Qwen2-VL and Qwen2.5-VL
  names; correct both DOI prefixes to `10.1136/bmjdhai-...`.
- **R2 Results and Discussion:** replace equivalence language with “no difference was detected”; replace “modest
  but consistent advantage” where the CI crosses zero; narrow clinical-readiness, procurement, policy, cost, and
  scalability claims unless the new analyses directly support them.
- **R3 terminology and novelty:** engage Jeong et al. directly and limit the novelty claim to the combined study of
  lightweight adaptation, cross-domain transfer, and agentic inference. Either add specialty-specific models or
  state clearly that broad conclusions do not cover that category.
- **Figures 2b and 3b:** plot both groups model-by-model, name each model, separate RQs, and label any oracle
  maximum with the selected model and selection rule.

## Author information still required

The following cannot be inferred safely from code or the manuscript and must be supplied before drafting the
final point-by-point response:

- Whether GPT-4 or GPT-4.1 generated the FairVLMed questions, including the exact dated model identifier.
- Number and expertise of FairVLMed reviewers; independent review procedure; disagreement resolution; number
  and reasons for exclusions; and the test used to confirm image-only answerability.
- Exact cluster hardware used for every completed run and whether logged wall time includes model loading.
- Validation and hyperparameter-selection procedure when an official validation split was unavailable.
- Exact checkpoints for every model and any local revisions or converted weights.
