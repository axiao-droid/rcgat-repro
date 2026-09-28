# E7 -- citation-network control rebuilt from the public sources

SNAP directed citation lists (``cit-HepTh`` / ``cit-HepPh``) restricted to the
arXiv submission years 2002-2003, node text and node dates from the arXiv Atom API
(title + abstract + first submission date), four gated cells, five seeds per graph.

| graph | nodes | edges | span | within 3 months of newest | within 12 months |
|---|---|---|---|---|---|
| hepth | 4183 | 19748 | 2002-01-01 .. 2003-04-30 (484 d) | 18.0% | 76.7% |
| hepph | 4669 | 12397 | 2001-12-31 .. 2003-02-28 (424 d) | 20.2% | 87.8% |

## hepth

Content floor: **0.6119** over 5 seeds (the frozen table reports 0.4533 for the full graph under a different build, so the levels are not comparable)

| model | test MRR | Δ vs floor | 95% CI | cfg |
|---|---|---|---|---|
| gat_dir | 0.6237±0.0072 | +0.0118 | [+0.0020,+0.0216] | c2_h64_d0.2 |
| gat_time | 0.6387±0.0108 | +0.0267 | [+0.0157,+0.0378] | c3_h64_d0.5 |
| ragat_sym | 0.6131±0.0069 | +0.0012 | [-0.0052,+0.0076] | c2_h64_d0.2 |
| ragat_time | 0.6134±0.0116 | +0.0015 | [-0.0139,+0.0168] | c2_h64_d0.2 |

| contrast | mean | 95% CI | signs + | p |
|---|---|---|---|---|
| gate | -0.0106 | [-0.0208,-0.0004] | 0/5 | 0.0444 |
| time | +0.0149 | [+0.0026,+0.0273] | 5/5 | 0.0285 |
| gate_under_time | -0.0253 | [-0.0431,-0.0075] | 0/5 | 0.0170 |
| full_vs_gatdir | -0.0103 | [-0.0193,-0.0014] | 0/5 | 0.0329 |

## hepph

Content floor: **0.4889** over 5 seeds (the frozen table reports 0.4209 for the full graph under a different build, so the levels are not comparable)

| model | test MRR | Δ vs floor | 95% CI | cfg |
|---|---|---|---|---|
| gat_dir | 0.4878±0.0022 | -0.0011 | [-0.0020,-0.0003] | c2_h64_d0.2 |
| gat_time | 0.5093±0.0426 | +0.0204 | [-0.0323,+0.0731] | c1_h128_d0.2 |
| ragat_sym | 0.4877±0.0015 | -0.0012 | [-0.0020,-0.0005] | c2_h64_d0.2 |
| ragat_time | 0.5085±0.0417 | +0.0195 | [-0.0317,+0.0708] | c1_h128_d0.2 |

| contrast | mean | 95% CI | signs + | p |
|---|---|---|---|---|
| gate | -0.0001 | [-0.0010,+0.0008] | 2/5 | 0.7806 |
| time | +0.0215 | [-0.0308,+0.0739] | 5/5 | 0.3168 |
| gate_under_time | -0.0009 | [-0.0824,+0.0806] | 2/5 | 0.9779 |
| full_vs_gatdir | +0.0207 | [-0.0310,+0.0724] | 3/5 | 0.3292 |
