# Official 10 vs 30 Ico product (not a Δv A/B)

Candidate is the v1.3 **10 km/s Ico product** (`KGAS66_Ico_K_kms-1.fits`, Briggs, 1.04″, 0.3″, 2871 pix).
Not cube M0. Not a channel-width A/B. Confounders: natural vs Briggs, beam-area 1.541, cell-area 1.778, mask 1.680, BPA 26.5°, CLEAN 0.80 vs 1.48 mJy.

30 km/s identity χ² = 168675.596 (official 168675.6; |Δ|=0.004; ok=True).
Conditional χ² at official MAP θ: 30=168675.596, 10=168701.213, Δ(10−30)=25.617. Joint 50–146 kλ Δχ²=-83.255.
P(k) median R(k) on k>1/θ_30 and below 30 km/s Nyquist = 0.7899954936749194.

**Lock DEC-066-SB at 30 km/s.** Gate is archive-only. Do not edit `DEC-066-SB.md`, `sb.py`, or `BMAJ_ICO_ARCSEC`. No re-MAP. `quote_inner_slope: false`. Do not start G4.

Wiener path: production `load_sb_template` K=(0.02)^2. Empty-corner K is undefined (n=0 finite in all four corners).

Would-unlock conventions (archive only): joint 50–146 kλ Δχ² = −83.3 (< −9) and median R(k) = 0.79 (≤ 1.2). Total χ² still favours 30 km/s by +25.6. Production stays 30 km/s.

