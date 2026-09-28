# E5 -- source-level inference on the paired contrasts

Unit of replication: the *source* (the node that does the ranking), averaged over the seeds in which it is rankable; intervals are a source-level bootstrap (10000 resamples) and a paired t interval.

| dataset | contrast | sources | mean per-source Δ | 95% t CI | 95% bootstrap CI | improved | sign test p | Wilcoxon p | trimmed 5% mean |
|---|---|---|---|---|---|---|---|---|---|
| npm | gat_dir | 181 | +0.0087 | [-0.0000,+0.0173] | [+0.0015,+0.0182] | 56% | 0.1369 | 0.0000 | +0.0022 |
| npm | ragat_sym | 181 | +0.0051 | [+0.0003,+0.0099] | [+0.0013,+0.0105] | 63% | 0.0006 | 0.0000 | +0.0019 |
| npm | gate_contrast_per_source | 181 | -0.0036 | [-0.0091,+0.0020] | [-0.0094,+0.0015] | 38% | 0.0010 | 0.2283 | -0.0005 |
| maven | gat_dir | 29 | +0.0101 | [-0.0078,+0.0279] | [-0.0028,+0.0292] | 52% | 1.0000 | 0.1285 | +0.0034 |
| maven | ragat_sym | 29 | +0.0109 | [-0.0145,+0.0362] | [-0.0081,+0.0379] | 45% | 0.7111 | 0.6567 | +0.0029 |
| maven | gate_contrast_per_source | 29 | +0.0008 | [-0.0254,+0.0270] | [-0.0197,+0.0285] | 45% | 0.7111 | 0.3995 | -0.0073 |
