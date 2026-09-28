# E4 -- gate behaviour, gradient norms, initialisation ablation

## maven/gat_dir/rand

5 seeds, config c4_h128_d0.5; test MRR 0.1024 (floor 0.0923, paired Δ +0.0101 [-0.0022,+0.0223]); epoch-0 anchor == floor(val): True.
- grad `encoder`: median norm 156812.6297 at epoch 1 -> 0.3803 at epoch 100, max median 156812.6297 at epoch 1

## maven/gat_dir/zero

5 seeds, config c4_h128_d0.5; test MRR 0.1024 (floor 0.0923, paired Δ +0.0101 [-0.0022,+0.0223]); epoch-0 anchor == floor(val): True.
- grad `encoder`: median norm 156812.6297 at epoch 1 -> 0.3803 at epoch 100, max median 156812.6297 at epoch 1

## maven/gat_dir/zero/text-only

10 seeds, config c4_h128_d0.5; test MRR 0.0929 (floor 0.0922, paired Δ +0.0006 [-0.0066,+0.0079]); epoch-0 anchor == floor(val): True.
- grad `encoder`: median norm 302682.2688 at epoch 1 -> 0.2029 at epoch 100, max median 302682.2688 at epoch 1

## maven/ragat_sym/rand

5 seeds, config c3_h64_d0.5; test MRR 0.1069 (floor 0.0923, paired Δ +0.0146 [+0.0061,+0.0231]); epoch-0 anchor == floor(val): True.
- gate `encoder.layer1.gate_in`: g in [0.297,1.808], mean 0.947, sd 0.202, |g-1|>0.01 for 95.9% of nodes; pre-act |a| p95 0.435; weight norm 0.207; descriptors: cos_out (-0.26), cit_age (+0.20), log1p_recent_out (-0.13), log1p_out_deg (-0.13), cos_in (-0.09)
- gate `encoder.layer1.gate_out`: g in [0.454,1.607], mean 0.930, sd 0.164, |g-1|>0.01 for 95.3% of nodes; pre-act |a| p95 0.347; weight norm 0.153; descriptors: log1p_recent_in (+0.39), log1p_in_deg (+0.39), log1p_mean_in_nbr_deg (+0.37), recent_in_share (+0.36), cit_age (+0.26)
- gate `encoder.layer2.gate_in`: g in [0.229,1.770], mean 1.007, sd 0.197, |g-1|>0.01 for 96.5% of nodes; pre-act |a| p95 0.382; weight norm 0.188; descriptors: has_edge (-0.35), log1p_out_deg (-0.32), log1p_recent_out (-0.31), cos_in (-0.29), log1p_mean_out_nbr_deg (-0.26)
- gate `encoder.layer2.gate_out`: g in [0.164,1.551], mean 1.040, sd 0.212, |g-1|>0.01 for 97.2% of nodes; pre-act |a| p95 0.410; weight norm 0.202; descriptors: has_edge (-0.47), log1p_mean_out_nbr_deg (-0.46), log1p_out_deg (-0.38), log1p_recent_out (-0.32), node_recency (-0.30)
- grad `gate`: median norm 0.0000 at epoch 1 -> 0.0405 at epoch 100, max median 0.0851 at epoch 88
- grad `encoder`: median norm 157352.3073 at epoch 1 -> 0.1609 at epoch 100, max median 157352.3073 at epoch 1

## maven/ragat_sym/zero

5 seeds, config c3_h64_d0.5; test MRR 0.1032 (floor 0.0923, paired Δ +0.0109 [+0.0001,+0.0217]); epoch-0 anchor == floor(val): True.
- gate `encoder.layer1.gate_in`: g in [0.672,1.609], mean 0.997, sd 0.066, |g-1|>0.01 for 54.8% of nodes; pre-act |a| p95 0.126; weight norm 0.046; descriptors: log1p_mean_out_nbr_deg (+0.45), has_edge (+0.44), log1p_recent_out (+0.39), log1p_out_deg (+0.37), log1p_mean_in_nbr_deg (+0.36)
- gate `encoder.layer1.gate_out`: g in [0.562,1.686], mean 0.990, sd 0.086, |g-1|>0.01 for 66.7% of nodes; pre-act |a| p95 0.151; weight norm 0.050; descriptors: log1p_recent_in (+0.38), log1p_in_deg (+0.38), log1p_mean_in_nbr_deg (+0.38), recent_in_share (+0.36), has_edge (+0.33)
- gate `encoder.layer2.gate_in`: g in [0.285,1.425], mean 1.004, sd 0.118, |g-1|>0.01 for 73.3% of nodes; pre-act |a| p95 0.192; weight norm 0.068; descriptors: log1p_recent_out (-0.65), log1p_out_deg (-0.64), log1p_mean_out_nbr_deg (-0.62), has_edge (-0.59), cit_age (-0.45)
- gate `encoder.layer2.gate_out`: g in [0.368,1.516], mean 1.012, sd 0.134, |g-1|>0.01 for 73.4% of nodes; pre-act |a| p95 0.229; weight norm 0.070; descriptors: has_edge (-0.62), log1p_recent_out (-0.60), log1p_mean_out_nbr_deg (-0.59), log1p_out_deg (-0.58), recent_in_share (-0.52)
- grad `gate`: median norm 0.0000 at epoch 1 -> 0.0275 at epoch 100, max median 0.1057 at epoch 96
- grad `encoder`: median norm 136978.7053 at epoch 1 -> 0.1090 at epoch 100, max median 136978.7053 at epoch 1

## maven/ragat_sym/zero/text-only

10 seeds, config c3_h64_d0.5; test MRR 0.0920 (floor 0.0922, paired Δ -0.0003 [-0.0051,+0.0046]); epoch-0 anchor == floor(val): True.
- gate `encoder.layer1.gate_in`: g in [0.996,1.034], mean 1.021, sd 0.000, |g-1|>0.01 for 80.0% of nodes; pre-act |a| p95 0.021; weight norm 0.000; descriptors: log1p_in_deg (+0.00), log1p_out_deg (+0.00), log1p_mean_in_nbr_deg (+0.00), log1p_mean_out_nbr_deg (+0.00), cos_in (+0.00)
- gate `encoder.layer1.gate_out`: g in [1.000,1.037], mean 1.022, sd 0.000, |g-1|>0.01 for 90.0% of nodes; pre-act |a| p95 0.022; weight norm 0.000; descriptors: log1p_in_deg (+0.00), log1p_out_deg (+0.00), log1p_mean_in_nbr_deg (+0.00), log1p_mean_out_nbr_deg (+0.00), cos_in (+0.00)
- gate `encoder.layer2.gate_in`: g in [0.932,0.997], mean 0.960, sd 0.000, |g-1|>0.01 for 90.0% of nodes; pre-act |a| p95 0.040; weight norm 0.000; descriptors: log1p_in_deg (+0.00), log1p_out_deg (+0.00), log1p_mean_in_nbr_deg (+0.00), log1p_mean_out_nbr_deg (+0.00), cos_in (+0.00)
- gate `encoder.layer2.gate_out`: g in [0.940,0.997], mean 0.965, sd 0.000, |g-1|>0.01 for 90.0% of nodes; pre-act |a| p95 0.035; weight norm 0.000; descriptors: log1p_in_deg (+0.00), log1p_out_deg (+0.00), log1p_mean_in_nbr_deg (+0.00), log1p_mean_out_nbr_deg (+0.00), cos_in (+0.00)
- grad `gate`: median norm 0.0000 at epoch 1 -> 0.0053 at epoch 100, max median 0.0186 at epoch 14
- grad `encoder`: median norm 213498.0679 at epoch 1 -> 0.1604 at epoch 100, max median 213498.0679 at epoch 1

## npm/gat_dir/rand

5 seeds, config c1_h128_d0.2; test MRR 0.1656 (floor 0.1569, paired Δ +0.0087 [+0.0013,+0.0160]); epoch-0 anchor == floor(val): True.
- grad `encoder`: median norm 113783.5497 at epoch 1 -> 1.9167 at epoch 100, max median 113783.5497 at epoch 1

## npm/gat_dir/zero

5 seeds, config c1_h128_d0.2; test MRR 0.1656 (floor 0.1569, paired Δ +0.0087 [+0.0013,+0.0160]); epoch-0 anchor == floor(val): True.
- grad `encoder`: median norm 113783.5497 at epoch 1 -> 1.9167 at epoch 100, max median 113783.5497 at epoch 1

## npm/gat_dir/zero/text-only

5 seeds, config c1_h128_d0.2; test MRR 0.1562 (floor 0.1569, paired Δ -0.0007 [-0.0055,+0.0042]); epoch-0 anchor == floor(val): True.
- grad `encoder`: median norm 125968.0193 at epoch 1 -> 5.0148 at epoch 86, max median 125968.0193 at epoch 1

## npm/ragat_sym/rand

5 seeds, config c1_h128_d0.2; test MRR 0.1615 (floor 0.1569, paired Δ +0.0046 [+0.0001,+0.0091]); epoch-0 anchor == floor(val): True.
- gate `encoder.layer1.gate_in`: g in [0.000,1.983], mean 1.052, sd 0.217, |g-1|>0.01 for 94.9% of nodes; pre-act |a| p95 0.926; weight norm 0.172; descriptors: log1p_mean_in_nbr_deg (-0.24), recip_out (-0.22), log1p_in_deg (-0.22), has_edge (-0.21), cos_out (-0.19)
- gate `encoder.layer1.gate_out`: g in [0.001,1.994], mean 0.988, sd 0.219, |g-1|>0.01 for 96.0% of nodes; pre-act |a| p95 0.586; weight norm 0.189; descriptors: cit_age (-0.30), node_recency (+0.29), log1p_mean_out_nbr_deg (-0.26), cos_out (-0.24), log1p_out_deg (-0.21)
- gate `encoder.layer2.gate_in`: g in [0.005,1.575], mean 0.981, sd 0.200, |g-1|>0.01 for 95.9% of nodes; pre-act |a| p95 0.554; weight norm 0.153; descriptors: node_recency (+0.46), cos_out (-0.26), log1p_out_deg (-0.25), has_edge (-0.25), log1p_mean_out_nbr_deg (-0.19)
- gate `encoder.layer2.gate_out`: g in [0.042,2.000], mean 0.928, sd 0.251, |g-1|>0.01 for 96.8% of nodes; pre-act |a| p95 0.901; weight norm 0.166; descriptors: node_recency (+0.58), log1p_mean_out_nbr_deg (+0.17), cit_age (+0.17), log1p_out_deg (+0.15), cos_out (+0.12)
- grad `gate`: median norm 0.0000 at epoch 1 -> 0.0611 at epoch 100, max median 0.6559 at epoch 4
- grad `encoder`: median norm 118936.2989 at epoch 1 -> 1.9864 at epoch 100, max median 118936.2989 at epoch 1

## npm/ragat_sym/zero

5 seeds, config c1_h128_d0.2; test MRR 0.1620 (floor 0.1569, paired Δ +0.0051 [+0.0015,+0.0086]); epoch-0 anchor == floor(val): True.
- gate `encoder.layer1.gate_in`: g in [0.142,1.667], mean 1.019, sd 0.102, |g-1|>0.01 for 84.8% of nodes; pre-act |a| p95 0.292; weight norm 0.064; descriptors: node_recency (-0.58), log1p_mean_in_nbr_deg (-0.45), log1p_in_deg (-0.44), has_edge (-0.32), cit_age (+0.18)
- gate `encoder.layer1.gate_out`: g in [0.324,1.401], mean 1.021, sd 0.079, |g-1|>0.01 for 92.1% of nodes; pre-act |a| p95 0.204; weight norm 0.042; descriptors: log1p_mean_out_nbr_deg (-0.74), log1p_out_deg (-0.69), cit_age (-0.65), node_recency (-0.63), cos_out (-0.56)
- gate `encoder.layer2.gate_in`: g in [0.101,1.351], mean 0.962, sd 0.123, |g-1|>0.01 for 92.5% of nodes; pre-act |a| p95 0.344; weight norm 0.063; descriptors: node_recency (+0.68), cos_in (-0.30), cit_age (-0.28), log1p_mean_out_nbr_deg (-0.27), cos_out (-0.25)
- gate `encoder.layer2.gate_out`: g in [0.198,1.316], mean 0.955, sd 0.133, |g-1|>0.01 for 94.0% of nodes; pre-act |a| p95 0.402; weight norm 0.065; descriptors: node_recency (+0.81), cit_age (-0.24), log1p_mean_out_nbr_deg (-0.20), log1p_out_deg (-0.20), log1p_in_deg (+0.20)
- grad `gate`: median norm 0.0000 at epoch 1 -> 0.0272 at epoch 100, max median 0.6258 at epoch 4
- grad `encoder`: median norm 107029.9786 at epoch 1 -> 2.0949 at epoch 100, max median 107029.9786 at epoch 1

## npm/ragat_sym/zero/text-only

5 seeds, config c1_h128_d0.2; test MRR 0.1563 (floor 0.1569, paired Δ -0.0006 [-0.0019,+0.0007]); epoch-0 anchor == floor(val): True.
- gate `encoder.layer1.gate_in`: g in [1.001,1.002], mean 1.001, sd 0.000, |g-1|>0.01 for 0.0% of nodes; pre-act |a| p95 0.001; weight norm 0.000; descriptors: log1p_in_deg (+0.00), log1p_out_deg (+0.00), log1p_mean_in_nbr_deg (+0.00), log1p_mean_out_nbr_deg (+0.00), cos_in (+0.00)
- gate `encoder.layer1.gate_out`: g in [1.001,1.002], mean 1.001, sd 0.000, |g-1|>0.01 for 0.0% of nodes; pre-act |a| p95 0.001; weight norm 0.000; descriptors: log1p_in_deg (+0.00), log1p_out_deg (+0.00), log1p_mean_in_nbr_deg (+0.00), log1p_mean_out_nbr_deg (+0.00), cos_in (+0.00)
- gate `encoder.layer2.gate_in`: g in [0.994,0.998], mean 0.995, sd 0.000, |g-1|>0.01 for 0.0% of nodes; pre-act |a| p95 0.005; weight norm 0.000; descriptors: log1p_in_deg (+0.00), log1p_out_deg (+0.00), log1p_mean_in_nbr_deg (+0.00), log1p_mean_out_nbr_deg (+0.00), cos_in (+0.00)
- gate `encoder.layer2.gate_out`: g in [0.993,0.998], mean 0.995, sd 0.000, |g-1|>0.01 for 0.0% of nodes; pre-act |a| p95 0.005; weight norm 0.000; descriptors: log1p_in_deg (+0.00), log1p_out_deg (+0.00), log1p_mean_in_nbr_deg (+0.00), log1p_mean_out_nbr_deg (+0.00), cos_in (+0.00)
- grad `gate`: median norm 0.0000 at epoch 1 -> 0.0148 at epoch 50, max median 0.3768 at epoch 4
- grad `encoder`: median norm 118576.7718 at epoch 1 -> 4.1085 at epoch 50, max median 118576.7718 at epoch 1

# Ablation contrasts (paired by seed)

## rand_vs_zero

| cell | mean Δ | 95% CI | seeds + | p |
|---|---|---|---|---|
| maven|gat_dir | +0.0000 | [+0.0000,+0.0000] | 0/5 | nan |
| maven|ragat_sym | +0.0037 | [-0.0139,+0.0213] | 2/5 | 0.5892 |
| npm|gat_dir | +0.0000 | [+0.0000,+0.0000] | 0/5 | nan |
| npm|ragat_sym | -0.0005 | [-0.0082,+0.0072] | 2/5 | 0.8696 |

## textonly_vs_zero

| cell | mean Δ | 95% CI | seeds + | p |
|---|---|---|---|---|
| maven|gat_dir | -0.0062 | [-0.0263,+0.0139] | 2/5 | 0.4400 |
| maven|ragat_sym | -0.0113 | [-0.0189,-0.0038] | 0/5 | 0.0138 |
| npm|gat_dir | -0.0093 | [-0.0159,-0.0028] | 0/5 | 0.0167 |
| npm|ragat_sym | -0.0057 | [-0.0085,-0.0028] | 0/5 | 0.0052 |

