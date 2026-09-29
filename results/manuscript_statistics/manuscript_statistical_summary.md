# Manuscript statistical analysis

Internal confidence intervals use 20,000 bootstrap resamples of the 15 repeated-CV fold estimates. Adjacent panel sizes are compared on matched repeat-fold units using two-sided Wilcoxon signed-rank tests; Holm correction is applied within each metric.

External top-1 and top-3 intervals are exact 95% binomial confidence intervals. Adjacent panel sizes are compared on matched samples using an exact McNemar test. Confidence intervals for mean model confidence use 20,000 sample-level bootstrap resamples.

## Internal CV macro F1

| panel_size | metric | n_folds | mean | sd | bootstrap_95ci_low | bootstrap_95ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | macro_f1 | 15 | 0.8583 | 0.01208 | 0.8524 | 0.8641 |
| 200 | macro_f1 | 15 | 0.8845 | 0.01139 | 0.8788 | 0.8898 |
| 500 | macro_f1 | 15 | 0.9214 | 0.0139 | 0.9144 | 0.9279 |
| 1000 | macro_f1 | 15 | 0.9293 | 0.008362 | 0.9252 | 0.9333 |

## Paired internal comparisons

| comparison | metric | n_paired_folds | mean_delta | median_delta | wilcoxon_statistic | p_value | holm_significant_0_05 | holm_adjusted_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200 vs 100 | macro_f1 | 15 | 0.02618 | 0.02536 | 0 | 6.104e-05 | True | 0.0001831 |
| 500 vs 200 | macro_f1 | 15 | 0.03691 | 0.03985 | 0 | 6.104e-05 | True | 0.0001831 |
| 1000 vs 500 | macro_f1 | 15 | 0.007945 | 0.007753 | 18 | 0.01508 | True | 0.01508 |

## Formal external validation

| cohort | panel_size | n_samples | accepted_top1 | accepted_top1_95ci_low | accepted_top1_95ci_high | accepted_top3 | accepted_top3_95ci_low | accepted_top3_95ci_high | mean_confidence | mean_confidence_bootstrap_95ci_low | mean_confidence_bootstrap_95ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GSE56044 | 100 | 106 | 0.9434 | 0.8809 | 0.9789 | 0.9811 | 0.9335 | 0.9977 | 0.8876 | 0.8568 | 0.9157 |
| GSE56044 | 200 | 106 | 0.8962 | 0.8219 | 0.947 | 0.9811 | 0.9335 | 0.9977 | 0.9235 | 0.8932 | 0.9504 |
| GSE56044 | 500 | 106 | 0.9623 | 0.9062 | 0.9896 | 1 | 0.9658 | 1 | 0.9338 | 0.904 | 0.9595 |
| GSE56044 | 1000 | 106 | 0.9717 | 0.9195 | 0.9941 | 1 | 0.9658 | 1 | 0.9456 | 0.9187 | 0.9693 |
| GSE48684 | 100 | 64 | 0.8438 | 0.7314 | 0.9224 | 0.9219 | 0.827 | 0.9741 | 0.7824 | 0.7339 | 0.8295 |
| GSE48684 | 200 | 64 | 0.8906 | 0.7875 | 0.9549 | 0.9375 | 0.8476 | 0.9827 | 0.8506 | 0.8013 | 0.8964 |
| GSE48684 | 500 | 64 | 0.8906 | 0.7875 | 0.9549 | 0.9375 | 0.8476 | 0.9827 | 0.8975 | 0.8571 | 0.9326 |
| GSE48684 | 1000 | 64 | 0.8594 | 0.7498 | 0.9336 | 0.9219 | 0.827 | 0.9741 | 0.862 | 0.813 | 0.9072 |
| GSE53051 | 100 | 102 | 0.7255 | 0.6282 | 0.8092 | 0.8824 | 0.8035 | 0.9377 | 0.8099 | 0.7646 | 0.8537 |
| GSE53051 | 200 | 102 | 0.8725 | 0.7919 | 0.9304 | 0.9314 | 0.8637 | 0.972 | 0.8662 | 0.8249 | 0.9044 |
| GSE53051 | 500 | 102 | 0.8529 | 0.7691 | 0.9153 | 0.9412 | 0.8764 | 0.9781 | 0.895 | 0.8577 | 0.9297 |
| GSE53051 | 1000 | 102 | 0.902 | 0.8271 | 0.952 | 0.9412 | 0.8764 | 0.9781 | 0.9068 | 0.8718 | 0.9391 |
| GSE105260_official_series | 100 | 35 | 0.8 | 0.6306 | 0.9156 | 0.9143 | 0.7694 | 0.982 | 0.7865 | 0.7158 | 0.853 |
| GSE105260_official_series | 200 | 35 | 0.7429 | 0.5674 | 0.8751 | 0.9143 | 0.7694 | 0.982 | 0.8069 | 0.7389 | 0.8706 |
| GSE105260_official_series | 500 | 35 | 0.8286 | 0.6635 | 0.9344 | 0.8857 | 0.7326 | 0.968 | 0.8528 | 0.7851 | 0.9133 |
| GSE105260_official_series | 1000 | 35 | 0.8857 | 0.7326 | 0.968 | 0.8857 | 0.7326 | 0.968 | 0.9096 | 0.8462 | 0.959 |

## External matched-panel comparisons

| cohort | comparison | small_only_correct | large_only_correct | discordant_pairs | exact_mcnemar_p_value |
| --- | --- | --- | --- | --- | --- |
| GSE56044 | 200 vs 100 | 6 | 1 | 7 | 0.125 |
| GSE56044 | 500 vs 200 | 0 | 7 | 7 | 0.01562 |
| GSE56044 | 1000 vs 500 | 0 | 1 | 1 | 1 |
| GSE48684 | 200 vs 100 | 0 | 3 | 3 | 0.25 |
| GSE48684 | 500 vs 200 | 2 | 2 | 4 | 1 |
| GSE48684 | 1000 vs 500 | 2 | 0 | 2 | 0.5 |
| GSE53051 | 200 vs 100 | 1 | 16 | 17 | 0.0002747 |
| GSE53051 | 500 vs 200 | 5 | 3 | 8 | 0.7266 |
| GSE53051 | 1000 vs 500 | 0 | 5 | 5 | 0.0625 |
| GSE105260_official_series | 200 vs 100 | 3 | 1 | 4 | 0.625 |
| GSE105260_official_series | 500 vs 200 | 2 | 5 | 7 | 0.4531 |
| GSE105260_official_series | 1000 vs 500 | 0 | 2 | 2 | 0.5 |
