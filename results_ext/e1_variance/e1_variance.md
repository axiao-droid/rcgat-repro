# E1 -- variance decomposition (seed/split vs model)

Runs read: 140 (datasets maven, npm; models gat_dir, gat_time, gatv2_dir, gcn_dir, ragat_sym, ragat_time, sage_dir).


## test MRR

| component | variance | share | sd |
|---|---|---|---|
| split | 4.475e-04 | 18.5% | 0.02115 |
| model | 3.405e-04 | 14.1% | 0.01845 |
| interaction | 9.283e-05 | 3.8% | 0.00964 |
| dataset | 1.538e-03 | 63.6% | 0.03921 |

**maven** -- pooled best `gcn_dir`; per-seed best agrees on 50% of seeds; mean Spearman(seed, pooled) = 0.50 (min 0.04); mean across-seed SD 0.0203 vs across-model SD 0.0252; mean seed-correlation between models -0.007 (min -0.456).

**npm** -- pooled best `gcn_dir`; per-seed best agrees on 80% of seeds; mean Spearman(seed, pooled) = 0.67 (min 0.21); mean across-seed SD 0.0095 vs across-model SD 0.0179; mean seed-correlation between models -0.079 (min -0.636).

## Delta vs content floor (paired)

| component | variance | share | sd |
|---|---|---|---|
| split | 4.488e-04 | 49.9% | 0.02118 |
| model | 3.405e-04 | 37.9% | 0.01845 |
| interaction | 9.271e-05 | 10.3% | 0.00963 |
| dataset | 1.677e-05 | 1.9% | 0.00410 |

**maven** -- pooled best `gcn_dir`; per-seed best agrees on 50% of seeds; mean Spearman(seed, pooled) = 0.50 (min 0.04); mean across-seed SD 0.0202 vs across-model SD 0.0252; mean seed-correlation between models -0.048 (min -0.604).

**npm** -- pooled best `gcn_dir`; per-seed best agrees on 80% of seeds; mean Spearman(seed, pooled) = 0.67 (min 0.21); mean across-seed SD 0.0098 vs across-model SD 0.0179; mean seed-correlation between models 0.018 (min -0.552).
