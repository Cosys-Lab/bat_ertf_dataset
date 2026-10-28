# Mouth emitters: piston calibrated to published beam widths

Jan, 2026-10-27: "myotis (and other mouth emitters) we need to calibrate from literature with a piston model".

*Eptesicus fuscus* and *Myotis daubentonii* (Vespertilionidae) emit through the open mouth. The head meshes have
closed mouths, so the emitter is modelled as a **disc-shaped piston patch at the mouth** (centred on the lip seam,
in front of the snout), vibrating in phase, on the rigid head. The head's own diffraction comes from the BEM. The
**piston radius** is the free parameter, calibrated so that the simulated horizontal half-amplitude angle (-6 dB,
measured from the beam axis) matches the published measurements below.

This differs on purpose from the "equivalent piston radius" in the literature: that radius belongs to a piston in an
infinite baffle and absorbs the head's baffling. Ours sits on the head, so the fitted radius should come out smaller
and closer to the real gape.

## Targets

| Species | Condition | f (kHz) | Half-amplitude angle (deg, horizontal) | Other | Source |
|---|---|---|---|---|---|
| *M. daubentonii* | field, search | 45 / 50 / 55 / 60 / 65 / 70 / 75 | 25 / 22 / 20 / 18 / 17 / 16 / 14 | DI 16.2 dB at 55 kHz; equivalent baffled piston radius 6.4 mm | Surlykke, Pedersen & Jakobsen 2009, Proc. R. Soc. B 276:853, Table 1 |
| *M. daubentonii* | lab | 55 | 40 | DI 10.7 dB; equivalent piston radius 3.4 mm | same |
| *E. fuscus* | field, search | 35 | 30 | | Hulgård et al. 2016, J. Exp. Biol. 219:334 (field microphone array) |
| *E. fuscus* | lab (various) | | about 30-50 | the main lobe follows a piston with radius near the open mouth; there is also a ventral lobe | Hartley & Suthers 1989; Ghose & Moss 2003 (as cited in Surlykke et al. 2009) |

TODO (verify before submission): read the full texts and check the numbers against them. The values above come from
summaries of Surlykke et al. 2009 (full text, PMC2664374) and from a search snippet for Hulgård et al. 2016.

## Plan

- Primary calibration: **field, search-phase** values. They represent the free-flying animal, and the M. daubentonii
  field table gives a frequency trend, which tests the piston model and not just a single point.
- Also publish the **lab calibration** for M. daubentonii (40 deg at 55 kHz) as an alternative wide-beam emitter, since
  bats change their gape (Jakobsen & Surlykke 2010).
- Fit: run piston patches with several radii (BEM linear: one run per radius) at the target frequencies, interpolate
  the half-amplitude angle against radius, and pick the radius with the least squared error in degrees across the
  target frequencies.
