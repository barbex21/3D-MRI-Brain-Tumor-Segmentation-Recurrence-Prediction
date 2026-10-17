# Descriptive results summary

This page summarizes the current 13-row audit in [experiment_audit.csv](../Results/experiment_audit.csv). It is a descriptive snapshot of the available runs, not a confirmatory modality comparison.

## Largest observed test Dice in this audit

| Target | Modality run | Validation Dice | Test Dice | Selected threshold | Audit status |
| --- | --- | ---: | ---: | ---: | --- |
| ET | T1ce + FLAIR | 0.7085 | 0.6402 | 0.99 | Directly verified from JSON |
| NECR | T1ce | 0.5102 | 0.3770 | 0.99 | Directly verified from JSON |

The values are taken from individual run records. NECR has three additional rows reconstructed from the aggregate summary; their provenance and rounding are documented in the audit and [data dictionary](data_dictionary.md).

## Interpretation limits

- These are the largest observed test scores among the runs currently listed; selecting a model based on repeated comparisons of test scores would make the reported maximum optimistic.
- The audit has no repeated seeds, confidence intervals, or paired statistical analysis.
- Three NECR rows use summary values rounded to four decimal places, and one recovered threshold is unavailable.
- No independent external validation is included.
- The scores do not establish clinical utility or generalization to another site, scanner, population, or preprocessing pipeline.

Use validation results for development decisions and reserve a test set for a pre-specified final evaluation. Revisit this summary when the missing run artifacts or additional experiments are audited.
