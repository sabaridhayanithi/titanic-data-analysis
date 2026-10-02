# Can Data Predict Human Fate?
## Five-minute Titanic survival investigation

**Visual direction:** midnight navy, warm ivory, and signal red; large editable titles; one main chart per data slide; consistent page numbers. Use the licensed archival photo only on the cover. The PowerPoint contains speaker notes and source credits.

| # | Slide | Main point |
|---|---|---|
| 1 | Can Data Predict Human Fate? | A question about survival in the Kaggle training sample. |
| 2 | 891 passenger records, one stark outcome | 342 of 891 records show survival (38.4%). |
| 3 | Before the clues: inspect the gaps | Cabin has 687 missing values, age 177, embarkation 2; no exact duplicate rows. |
| 4 | Keep the analysis honest | Use complete observed values per comparison; do not impute missing age/cabin. |
| 5 | The survival gap by recorded sex was large | Women 74.2% (233/314); men 18.9% (109/577). |
| 6 | Survival rates fell with ticket class | First 63.0% (136/216), second 47.3% (87/184), third 24.2% (119/491). |
| 7 | Higher fares lined up with higher survival | Compare survival across sample fare quartiles. |
| 8 | Class and survival were statistically associated | Pearson χ²(2)=102.89, p=4.55×10⁻²³. |
| 9 | The sex gap appears within every class | Compare rates by sex within each ticket class. |
| 10 | Patterns are clear; causes are not | Summarize the association and its limits without claiming causation. |

## Appendix

11. Variable dictionary and dataset scope.
12. Survival by age band; age-known sample size and mean age by outcome.
13. Age and fare: mean, median, mode, sample variance, standard deviation, coefficient of variation, and pairwise covariance.
14. Chi-square test, Kaggle data reference, Wikimedia Commons image credit, and study limitations.

The figures are calculated from Kaggle's standard 891-row Titanic training CSV. The file is an observational subset with 177 missing ages and 687 missing cabin values. Statistical association does not establish cause or recover the full rescue decision process.
