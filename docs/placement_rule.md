# Where the receivers and nostrils go: calibrate on two species, transfer to five

Jan, 2026-10-27: "get close to the pdisc and mmicr measurements, learn from this, and then know how to adjust to the
other models".

## 1. Calibration on the reference species (scripts/placement_study.py)

On the clean CT meshes (CT frame), dense candidate patches (0.5 mm apart) around each ear canal region (4 mm) and the
noseleaf base, all solved on the GPU at ten frequencies over the reference band. Pattern correlation with the
reference (frontal hemisphere, per-frequency normalized dB clipped at -30 dB, mean over frequency):

| | Left ear | Right ear | Emitter | Orientation |
|---|---|---|---|---|
| *M. microtis*, best placement | 0.94 | 0.92 | 0.93 | CT frame, refit changes < 1.5 deg |
| *M. microtis*, median candidate | 0.60 | 0.55 | 0.24 | |
| *P. discolor*, best placement | 0.78 | 0.83 | 0.81 | De Mey frame (fitted) |
| *P. discolor*, median candidate | 0.70 | 0.63 | 0.41 | |
| earlier: smoothed meshes, occlusion-rule ears, fitted frame | 0.83 / 0.75 | 0.79 / 0.80 | 0.93 / 0.75 | |

The receiver position dominates: within the search region the agreement ranges from about 0.1 to 0.94. The
best-matching positions are well localized (the five best within 1 mm) and nearly mirror each other between ears.
For P. discolor the reference itself is the limit: it is not smooth between frequency steps 0.5 kHz apart.

## 2. What the best positions have in common (scripts/learn_placement_rule.py)

Descriptors of every candidate, against its agreement with the reference (Spearman rank correlation, 4 ears):

| Descriptor | mmicr L | mmicr R | pdisc L | pdisc R |
|---|---|---|---|---|
| front/back ratio (frontal vs rear half-space power) | 0.79 | 0.94 | 0.81 | 0.90 |
| directivity index | 0.83 | 0.92 | 0.51 | 0.85 |
| occlusion | 0.84 | 0.87 | 0.81 | 0.87 |
| frontal gain | 0.75 | 0.73 | 0.44 | 0.45 |

Physically: the ear canal entrance (and the nostrils) are where pinna and noseleaf focus sound forward most strongly.

## 3. The rule

- **Ears:** score = z(front/back ratio) + z(occlusion), averaged over 0.75 mm; the receiver is the maximum.
- **Nostrils:** among mirror-symmetric pairs (mismatch < 1 mm across the symmetry plane) 1.5-2.5 mm apart, the pair
  with the highest front/back ratio of the summed field.
- **Oral emitters:** analytical piston calibrated to published beam widths (docs/mouth_emitter_calibration.md).

Agreement the rule reaches on the calibration species (pattern correlation, same metric):

| | mmicr L | mmicr R | pdisc L | pdisc R | mmicr emitter | pdisc emitter |
|---|---|---|---|---|---|---|
| rule | 0.78 | 0.86 | 0.77 | 0.83 | 0.93 | 0.74 |
| best possible | 0.94 | 0.92 | 0.78 | 0.83 | 0.93 | 0.81 |
| occlusion alone (previous rule) | 0.67 | 0.62 | 0.77 | 0.68 | | |

Six candidate rules were compared on four ears and two emitters, so the rule is fitted to little data. It was the best
choice for each species separately, and it has a physical reading, but its accuracy on the other five species is
not validated. Each dataset entry therefore also carries the next-best distinct positions (at least 1 mm apart) as an
ensemble, so users can see how much the ERTF depends on the placement.

## 4. Out-of-sample check (scripts/placement_out_of_sample.py, results/validation/placement_oos.json)

The rule has no continuous fitted parameters; what was learned are discrete choices (descriptors, equal weights,
smoothing radius; for the nostrils the score, spacing window and mirror tolerance). We re-made these choices on a
training subset, from a family of 60 ear rules (each of five descriptors alone or the equal-weight sum of two, smoothed
over 0/0.5/0.75/1.0 mm) and 16 nostril rules (four scores x two spacing windows x two mirror tolerances), and applied
the winner to the held-out species. With two ears per species many rules tie on the training data; the held-out value
is then the mean over the tied rules (range in brackets). Reference match = pattern correlation as in section 1;
percentile = share of all candidates the selected one is not worse than.

| Held out | selected (LOSO) | published rule | best | median | most occluded | percentile (LOSO) |
|---|---|---|---|---|---|---|
| *P. discolor* left ear | 0.77 (0.77-0.77) | 0.77 | 0.78 | 0.70 | 0.77 | 98 |
| *P. discolor* right ear | 0.75 (0.62-0.83) | 0.83 | 0.83 | 0.63 | 0.68 | 80 |
| *M. microtis* left ear | 0.76 | 0.78 | 0.94 | 0.60 | 0.67 | 87 |
| *M. microtis* right ear | 0.81 | 0.86 | 0.92 | 0.55 | 0.62 | 97 |
| *P. discolor* nostrils | 0.70 (0.65-0.74) | 0.74 | 0.81 | 0.41 (sym. 0.65) | 0.57 | 95 (sym. 78) |
| *M. microtis* nostrils | 0.90 (0.88-0.91) | 0.93 | 0.93 | 0.24 (sym. 0.57) | 0.36 | 99.9 (sym. 96) |

"Most occluded" = most occluded candidate (ears; identical to the previous occlusion rule's point) or mirror-symmetric
pair with the highest summed occlusion (nostrils). "sym." = among the mirror-symmetric pairs 1.5-2.5 mm apart that the
rule chooses from (23 and 54 pairs); the remaining reference set is all pairs 0.8-4 mm apart (1308 and 3290).
Leave-one-ear-out (train on three ears) gives 0.77/0.75/0.78/0.78 for pdisc L/R, mmicr L/R. Within the family, the
published ear rule ranks 6th of 60 on all four ears (mean 0.810 against 0.813 for the best, all with 1 mm smoothing),
so it is not the in-sample optimum. Reading: the transferred rule lands in the top 3-20 % of candidates and 0.0-0.18
below the best candidate on held-out data, losing up to 0.08 against the in-sample choice; for the nostrils most of
the gain over the median comes from the mirror-symmetry constraint itself. Two species are little evidence; the held-
out numbers are the honest estimate of what the rule reaches on the other five species, given that their references
would be of similar kind.

## Methods text

Candidates were defined automatically in the CT frame. For the ears, we took the most occluded vertex in each lateral
region (|y| > 4 mm, rear 55 % of the head) as the centre of the ear canal region and sampled the vertices within 4 mm
of it by farthest-point sampling at 0.5 mm spacing; each candidate is a 0.4 mm disc patch. Nostril candidates were
sampled the same way on the forward-facing surface (outward normal x-component > 0.2) within 3 mm (calibration
species) or 4 mm (other species) of a manually set noseleaf centre, with 0.3 mm patches. All candidates were solved at
the ten interior points of twelve equally spaced frequencies over the reference band for the calibration species (*P.
discolor* 31.5-88.5 kHz, *M. microtis* 51-149 kHz) and over the dataset band otherwise. The front/back ratio is the
mean power over the half-space x > 0 of the CT frame divided by that over x < 0 (cos-elevation weights), in dB,
averaged over frequency. Occlusion is the fraction of 96 random directions in the outward hemisphere of the vertex
normal along which a ray, started 20 µm off the surface, hits the mesh (no length limit). For each ear, both
descriptors were standardized to z-scores over that ear's candidates, summed, and averaged over all candidates within
0.75 mm; the receiver is the maximum. For the nostrils, we fitted the midsagittal plane by minimizing the mean
distance between mirrored and original vertices over the 70 % best-matching vertices, and kept pairs 1.5-2.5 mm apart
whose mirror mismatch is below 1 mm; the nominal pair maximizes the front/back ratio of the area-weighted, in-phase
sum of both patches. Agreement with a reference is the solid-angle-weighted correlation of the dB patterns over its
frontal hemisphere (normalized to the maximum, clipped at -30 dB), averaged over frequency. Alternatives are the next-
best positions under the rule, selected greedily at least 1 mm from every position already chosen (pair midpoints for
nostrils), excluding any within 0.5 mm of the dataset landmark; up to four are kept. For the five rule-placed species
they lie 1.0-5.0 mm from the nominal position; for *P. discolor* and *M. microtis*, whose landmarks are the reference-
best positions, the first alternative is the rule's own choice (0.5-0.8 mm away for the ears, 2.1 and 3.1 mm for the
nostril pairs), and all lie at least 0.5 mm from the nominal.

| Parameter | Value | Code |
|---|---|---|
| Ear region | most occluded vertex with \|y\| > 4 mm in rear 55 % of head, radius 4 mm | ear_landmarks.ear_candidates, placement_study.EAR_REGION |
| Nose region | forward normals (n_x > 0.2), radius 3 mm (calibration) / 4 mm (transfer) around config nose_centre | placement_study / apply_placement_rule NOSE_REGION |
| Candidate spacing | 0.5 mm (farthest-point sampling of mesh vertices) | SPACING |
| Patch radius | ear 0.4 mm, nostril 0.3 mm (geodesically connected faces) | EAR_PATCH, NOSE_PATCH |
| Frequencies | 10 interior points of 12 equally spaced over the reference band (calibration) or dataset band (transfer: 32.7-147.3 kHz; *R. formosae* 30.5-124.5 kHz) | N_FREQ |
| Front/back ratio | mean power x > 0 over x < 0 (CT frame), cos(el) weights, dB, mean over frequency | learn_placement_rule.descriptors |
| Occlusion | 96 uniform random rays (seed 0) flipped into outward hemisphere, 20 µm offset, unlimited range | mesh.ambient_occlusion |
| Standardization | z-score over the candidates of one ear (population std) | apply_placement_rule |
| Smoothing | mean over candidates within 0.75 mm | SMOOTH |
| Symmetry plane | yaw/roll/offset; trimmed (70 %) mean mirror distance, 8000-vertex subsample, Nelder-Mead from 9 starts | symmetry.fit |
| Nostril pair | spacing 1.5-2.5 mm, mirror mismatch < 1 mm, max front/back ratio of area-weighted sum | apply_placement_rule |
| Match metric | cos(el)-weighted correlation of dB patterns, frontal hemisphere, -30 dB floor, mean over 10 frequencies | reference.metrics |
| Alternatives | greedy, >= 1 mm from each other (MIN_SEP), dropped if < 0.5 mm from landmark (run_ertf.ensemble min_sep), up to 4 | apply_placement_rule, run_ertf.ensemble |
| Alternative distances | results/validation/alt_distances.json | scripts/alt_distances.py |
