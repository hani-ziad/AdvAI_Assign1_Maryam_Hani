# Predicting Late Sprint Closure and Unfinished Committed Scope from Reconstructed Jira Histories


This is our repo for CSBP711 Assignment 1 (datasets and algorithm comparison). We look at Agile sprints from eight open-source Jira projects and try to guess two things using only what the team knew when the sprint started:

- Will the sprint close late? (more than one day after its planned end)
- Will it spill over? (at least one committed issue is still not done at the planned end)

## 1. The dataset

- **Name:** TAWOS, a dataset of Agile open-source projects from Jira.
- **Where it comes from:** UCL Research Data Repository, https://rdr.ucl.ac.uk/articles/dataset/The_TAWOS_dataset/21308124 (DOI http://doi.org/10.5522/04/21308124).
- **Licence:** Apache 2.0. The authors say it is for research use, which is what we do.
- **Version and date:** one published release (MSR 2022). We downloaded it in September 2026 and first committed our copy on 2026-09-12.
- **What we use:** the Sprint, Issue and Change_Log tables for eight projects. We exported them from the full database with `scripts/export_tawos_subset.sql` and kept the CSVs in `data/raw/tawos/`, so you can run everything without downloading anything.
- **Cite as:** Tawosi, Al-Subaihin, Moussa, Sarro. A Versatile Dataset of Agile Open Source Software Projects. MSR 2022. https://doi.org/10.1145/3524842.3528029

| Table | Rows | What we use it for |
|---|---|---|
| `tawos_sprint.csv` | 2,657 | sprint dates and state |
| `tawos_issue.csv` | 145,309 | story points, status, resolution |
| `tawos_change_log.csv` | 88,364 | rebuilding what each sprint looked like on day one |

The projects are Apache Mesos, Appcelerator Studio, Confluence Cloud, Confluence Server, Lsstcorp Data management, Mule, The Titanium SDK and Titanium Mobile Platform.

Of the 2,657 raw sprints, 2,610 are closed and usable. We drop 8 cold starts (a project's very first sprint, where there is no history yet), which leaves **2,602 sprints**. 911 of those committed nothing, so **1,691 sprints have real scope**.

For every project we train on the earlier 70% of sprints and test on the later 30%. That gives 1,816 train and 786 test sprints for late closure, and 1,180 and 511 for spillover (nonempty sprints only). In the test sets 31.3% of sprints close late and 94.3% spill over.

### Problems we found in the data

- **Missing history, the big one.** The change log does not record every change. For 7.2% of the reconstructed commitments we only know the sprint from the current Jira value, and for 45.1% we only know story points that way. We kept them and used the current value as a fallback. To check this was not driving the results, we re-ran with only the projects that have enough observed history (6 and then 4 of 8). Scope-only logistic regression still beats the random forest for spillover (AUC .914 vs .879 with 6 projects). See `results/missing_history_sensitivity.csv`.
- **Duplicates.** None. No repeated rows or IDs in any of the three tables, and no repeated sprint and issue pairs among the 20,603 commitments. An issue can be in more than one sprint, and that is fine.
- **Bad timestamps.** 3 closed sprints have a close time before their start, so we removed them. We also remove FUTURE and ACTIVE sprints because we don't know how they ended.
- **Leakage.** Every feature comes from the state at sprint start: the change log is replayed to that moment, and history only uses earlier sprints of the same project. We never use the final status of an issue or the final sprint membership. `tests/test_no_leakage.py` and `tests/test_planning_snapshot.py` check this, and `docs/leakage_case_studies.docx` has real examples. The split is by time, never random.
- **Imbalance.** 94% of nonempty sprints spill over, so just guessing "spillover" every time already gets F1 .971. That is why we use ROC-AUC as the main metric and show F1 only for reference.

## 2. The algorithms

Every model gets the same data, the same time split, the same preprocessing (scaled numbers, one-hot project), the same 0.5 threshold and the same seed (42).

| Model | Why it is here | Settings |
|---|---|---|
| Always-positive | a baseline that learns nothing | predicts the positive class every time |
| Scope-only logistic regression | simple, readable baseline | 3 features: committed issues, committed story points, sprint length |
| Logistic regression | linear model | all 8 features, balanced class weights |
| Random forest | non-linear model | 300 trees, depth 5, min leaf 3, balanced class weights |
| XGBoost | boosted trees | 300 trees, depth 4, learning rate 0.05 |

## 3. Results

Late closure uses all 786 test sprints. Spillover uses the 511 test sprints that had scope. Training time is the median of 5 fits and only counts model fitting. The full table is in `results/assignment_comparison.csv`.

| Outcome | Model | F1 | ROC-AUC | Train time | Model size |
|---|---|---|---|---|---|
| Late closure | Always-positive | .477 | .500 | 0 s | no model |
| Late closure | Scope-only LR | .336 | .530 | 0.0053 s | 4 coefficients |
| Late closure | Logistic regression | .453 | .636 | 0.0081 s | 16 coefficients |
| Late closure | **Random forest** | .562 | **.705** | 0.4389 s | 300 trees, 15,374 nodes |
| Late closure | XGBoost | .535 | .695 | 1.7950 s * | 300 trees, depth 4 |
| Spillover | Always-positive | .971 | .500 | 0 s | no model |
| Spillover | **Scope-only LR** | .696 | **.885** | 0.0061 s | 4 coefficients |
| Spillover | Logistic regression | .763 | .829 | 0.0122 s | 16 coefficients |
| Spillover | Random forest | .875 | .852 | 0.3557 s | 300 trees, 10,308 nodes |
| Spillover | XGBoost | .963 | .862 | not measured * | 300 trees, depth 4 |

\* The XGBoost scores come from our saved Phase 1 runs (`results/baseline_results.json` and `results/conditional_spillover.json`). The 1.7950 s was timed on a team machine that has xgboost. It is not installed everywhere, so `scripts/run_assignment_comparison.py` skips it when missing and fills in the time and size when it is there.

## 4. Why it comes out this way

**Spillover is mostly about how much was committed.** A sprint that starts with a lot of issues and story points almost always leaves something unfinished. So a model that only sees three scope numbers ranks spillover better (.885) than the random forest with all the features (.852). More features mostly add noise.

**Late closure is a different story.** Scope tells us nothing (AUC .530, basically a coin flip). What helps is which project it is and how long the sprint is, and that is why the random forest wins there.

Permutation importance says the same thing (`results/permutation_importance.csv`). For spillover, shuffling the committed issue count costs 0.220 AUC and the next feature costs only 0.021. For late closure, shuffling the project costs 0.104 and sprint length 0.077, while scope costs about 0.01.

### Ablation

If scope really drives spillover, taking it away from the random forest should hurt a lot, and for late closure it should hardly matter. We tested exactly that (`results/assignment_ablation.csv`). Random forest AUC:

| Features used | Spillover | Late closure |
|---|---|---|
| All 8 | .852 | .705 |
| Planning features only (4) | .871 | .706 |
| History only (5) | .636 | .625 |
| Without project (7) | .866 | .696 |
| Without committed scope (6) | .652 | .690 |
| Scope-only LR, for reference | .885 | .530 |

Without scope, spillover drops from .852 to .652, while late closure only goes from .705 to .690. That is what we predicted. It could have gone the other way, for example if history alone had matched the full model, and it didn't.

### What to be careful about

This is one time split on eight projects, not random folds. The story-point fallback (45% of commitments) is the biggest weak spot in the data. Bootstrap intervals over projects (2,000 resamples) and leave-one-project-out results are in `results/cluster_uncertainty.csv` and `results/leave_one_project_out_summary.csv`.

## 5. How to run it

```
pip install -r requirements.txt
python -m scripts.run_tests                    # unit and leakage tests
python -m scripts.run_assignment_comparison    # comparison table + ablation
python -m scripts.run_baseline                 # saved Phase 1 baselines
python -m scripts.run_scope_decomposition      # conditional spillover results
python -m scripts.run_validation_sensitivity_analyses
```

`python -m scripts.run_full_reproduction` runs the whole Phase 1 pipeline. Everything is written to `results/`. The seed is fixed, so you should get the same numbers we report.

```
data/raw/tawos/    the TAWOS CSVs
src/               rebuilding sprints, labels, features, baselines, evaluation
scripts/           experiment scripts
tests/             unit and leakage tests
results/           saved tables and figures
docs/              data access notes, leakage cases, test plan
presentation/      the 5-slide deck (assignment package only)
```

## 6. Who did what

| Member | Contribution |
|---|---|
| Hani AbuSharkh | Rebuilt sprints from the change log, labels and features. Wrote the comparison and ablation script, the results tables, the data audit, and most of the writing (this README and the explanation). |
| Maryam Hanif | Baseline and sensitivity scripts, including the training-time measurements. Scope-decomposition experiments. Designed and presented the slide deck. |
| Samra Nawazish | Dataset background and literature check. Reviewed the data-problems section. Slide 2. |
| Najwa Suleiman | Test plan and leakage tests. Re-ran everything on a clean machine. Slide 3. |
| Alyah Alameeri | Slide review and rehearsal. Formatting of figures and tables. Proofread this README. |

## 7. AI assistant use

We used Claude (Anthropic) to help write and tidy some of the code, including the comparison and ablation script. We ran all the code ourselves and checked every number in the README and slides against the result files in `results/`. We edited the text afterwards. No number here comes from the AI. They all come from scripts in this repo.
