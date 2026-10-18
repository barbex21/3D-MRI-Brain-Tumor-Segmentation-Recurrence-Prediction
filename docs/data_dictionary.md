# Results table data dictionary

| Column | Definition | Primary source |
| --- | --- | --- |
| target | Binary segmentation target (et or necr). | config.json → target_mask |
| modalities | Input modalities, joined with +. | config.json → modalities |
| validation_dice | Best full-volume validation mean Dice used to select the checkpoint. | metrics.json → best_validation_dice; recovered rows use segmentation_results_summary.csv |
| test_dice | Full-volume test mean Dice at the selected validation threshold. | metrics.json → test.dice_mean; recovered rows use segmentation_results_summary.csv |
| selected_threshold | Operating threshold selected on validation and applied to test. | metrics.json → validation_selected_threshold; recovered rows use the individual report when available |
| experiment_id | Run directory name. | Experiment folder |
| source_file | Relative names of source files supporting this row. | Run folder, summary CSV, or recovery report |
| verification_status | verified_json, recovered_summary_and_report, or recovered_comparison_report. | Audit process |

## Metric calculation

In the trainer, binary Dice is computed per volume after thresholding probabilities. Empty-target volumes are excluded from the primary Dice, precision, and recall averages; their counts and false-positive behavior are tracked separately. The summary is the arithmetic mean over non-empty volumes. Validation evaluates the configured threshold candidates and selects the one with the highest validation mean Dice. The best-checkpoint threshold is then passed unchanged to the test evaluation.

## Recovery status

Ten runs have parseable config.json and metrics.json files. Three NECR runs lack metrics.json; their Dice values were recovered from the project summary CSV, rounded there to four decimal places, and cross-checked against reports. Thresholds for T1ce+FLAIR and T1ce+T2+FLAIR come from their individual reports. The T1ce+T2 NECR threshold (0.80) comes from the ET-vs-NECR comparison report. The audit marks all three rows as recovered rather than directly verified from JSON; the rounded Dice values should be replaced with full-precision metrics if the original metrics files are recovered.

No patient-level records are present in this table.

