"""Series J Stage 4: S3a (alpha = 0) phase markers and the tests P1-P6.

Data: ``figs/experiments/knee/runs`` (simulations), ``analysis/field`` (frozen field on the
simulated states, ``frozen_field.py``), ``analysis/{optimum,phi_alpha0}.json``.
Writes ``figs/experiments/knee/analysis/s3a_numbers.json`` and prints markdown tables
(``> s3a_tables.md``). Run: ``real/bin/python experiments/knee/analysis/analyze_knee.py``.
"""
from __future__ import annotations

import json

import numpy as np

import lib_knee as K

NUM = {"markers": {}, "P1": {}, "P2": {}, "P3": {}, "P4": {}, "P5": {}, "P6": {}, "finiteD": {}}
BOOT = 200


def lane_names(g, d):
    return [f"{b:g}" for b in d["beta2s"]] if g["opt"] == "adam" else ["sgd"]


def key(g, lane):
    e = f"_e{g['eps']:g}" if g["opt"] == "adam" else ""
    return f"C{g['C']}_k{g['kappa']:g}{e}_D{g['D']}{('_' + g['lane_tag']) if g['lane_tag'] else ''}_{lane}"


def fmt(x, f=".3g"):
    return "–" if x is None or not np.isfinite(x) else format(x, f)


# --------------------------------------------------------------------------- #
def markers_table():
    st = K.status()
    print("\n## Phase markers (seed-mean class-mean loss; 5-95% bootstrap over runs)\n")
    print("| C | kappa | eps | D | lane | T | ell(T)/ell0 | t_c | t_knee | ell_knee/ell0 | m_knee |"
          " frac_eps knee / T | cliff R2 (lin t / ln t), span | tail exp (prev / last decade) |"
          " certificate (F/T/M/U) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in K.groups():
        d = K.load(g["name"])
        cert = st.get(g["name"], {}).get("counts_by_beta2", [])
        for lane, ln in enumerate(lane_names(g, d)):
            m = K.lane_markers(d, lane, boot=BOOT, seed=lane)
            c = cert[lane] if lane < len(cert) else {}
            m["certificate"] = c
            NUM["markers"][key(g, ln)] = m
            ci = m.get("t_c_ci")
            cis = f" [{fmt(ci[0])}, {fmt(ci[1])}]" if ci else ""
            print(f"| {g['C']} | {g['kappa']:g} | {fmt(g['eps'], 'g') if g['eps'] else '–'} |"
                  f" {g['D']} | {ln} | {m['T']:.0f} | {m['ell_T_rel']:.2e} |"
                  f" {fmt(m['t_c'])}{cis} | {fmt(m.get('t_knee'))}{'*' if m.get('knee_fallback') else ''} |"
                  f" {fmt(m.get('ell_knee_rel'), '.2e')} | {fmt(m.get('m_knee'), '.2f')} |"
                  f" {fmt(m.get('frac_eps_knee'), '.2f')} / {fmt(m.get('frac_eps_T'), '.2f')} |"
                  f" {fmt(m.get('R2_lin_t'), '.3f')} / {fmt(m.get('R2_log_t'), '.3f')}, {fmt(m.get('cliff_span'), '.1f')}x |"
                  f" {fmt(m.get('tail_prev'), '.2f')} / {fmt(m.get('tail_last'), '.2f')} |"
                  f" {c.get('FIXED', '–')}/{c.get('TAIL', '–')}/{c.get('MOVING', '–')}/{c.get('UNRESOLVED', '–')} |")
    print("\n(* knee fallback: no clean local maximum, the global maximum of the exponent after t_c)")


# --------------------------------------------------------------------------- #
def get(C, kappa, eps, lane, D=None, opt="adam", tag=""):
    g = K.find(opt, C, kappa, eps, D=D, lane_tag=tag)
    if g is None:
        return None
    ln = "sgd" if opt == "sgd" else f"{lane:g}"
    return NUM["markers"].get(key(g, ln))


def p2():
    """Low noise: the eps floor and the knee; eps = 1e-16 moves the knee down; tail ratio G."""
    print("\n## P2 (kappa <= 0.25): eps floor, knee and tail\n")
    print("| C | kappa | beta2 | ell_knee/ell0 eps 1e-8 / 1e-16 (ratio) | m_knee 1e-8 / 1e-16 |"
          " frac_eps at knee (1e-8) | ell(T) 1e-8 / 1e-16 | G = (R_A/R_S)/(sqrt(kappa) 10 sqrt(eps)) at common T |")
    print("|---|---|---|---|---|---|---|---|")
    for C in (128, 32):
        for kappa in (0.25, 0.1):
            gs = K.find("sgd", C, kappa)
            ds = K.load(gs["name"]) if gs else None
            for b2 in K.BETA2S:
                a, b = get(C, kappa, 1e-8, b2), get(C, kappa, 1e-16, b2)
                if a is None or b is None:
                    continue
                ratio = (b.get("ell_knee_rel", np.nan) / a.get("ell_knee_rel", np.nan))
                G = np.nan
                if ds is not None:
                    ga = K.find("adam", C, kappa, 1e-8)
                    da = K.load(ga["name"])
                    lane = list(da["beta2s"]).index(b2)
                    Tc = min(da["time"][-1], ds["time"][-1])
                    la = np.interp(np.log(Tc), np.log(da["time"][1:]),
                                   np.log(da["loss"][1:, lane].mean((1, 2))))
                    ls = np.interp(np.log(Tc), np.log(ds["time"][1:]),
                                   np.log(ds["loss"][1:, 0].mean((1, 2))))
                    G = np.exp(la - ls) / (np.sqrt(kappa) * 10 * np.sqrt(1e-8))
                NUM["P2"][f"C{C}_k{kappa:g}_{b2:g}"] = {
                    "ell_knee_e8": a.get("ell_knee_rel"), "ell_knee_e16": b.get("ell_knee_rel"),
                    "ratio": ratio, "m_knee_e8": a.get("m_knee"), "m_knee_e16": b.get("m_knee"),
                    "frac_eps_knee_e8": a.get("frac_eps_knee"), "ellT_e8": a["ell_T_rel"],
                    "ellT_e16": b["ell_T_rel"], "G": G}
                print(f"| {C} | {kappa:g} | {b2:g} | {fmt(a.get('ell_knee_rel'), '.2e')} /"
                      f" {fmt(b.get('ell_knee_rel'), '.2e')} ({fmt(ratio, '.1e')}) |"
                      f" {fmt(a.get('m_knee'), '.1f')} / {fmt(b.get('m_knee'), '.1f')} |"
                      f" {fmt(a.get('frac_eps_knee'), '.2f')} | {a['ell_T_rel']:.2e} / {b['ell_T_rel']:.2e} |"
                      f" {fmt(G, '.2f')} |")


def p3():
    """High noise: the late loss approaches the population optimum; eps changes nothing."""
    print("\n## P3 (kappa >= 0.5): late loss against the population optimum ell*\n")
    print("| C | kappa | lane | ell*/ell0 | ell(T)/ell* eps 1e-8 | eps 1e-16 / 1e-8 at T | m(T) / m* |"
          " tail exp (last decade) |")
    print("|---|---|---|---|---|---|---|---|")
    for C in (128, 32):
        for kappa in (2.0, 1.0, 0.5):
            opt = K.optimum(C, kappa)
            for b2 in list(K.BETA2S) + ["sgd"]:
                if b2 == "sgd":
                    a, b = get(C, kappa, None, 0, opt="sgd"), None
                else:
                    a, b = get(C, kappa, 1e-8, b2), get(C, kappa, 1e-16, b2)
                if a is None:
                    continue
                r = a["ell_T_rel"] / opt["ell_rel"]
                e = b["ell_T_rel"] / a["ell_T_rel"] if b else np.nan
                NUM["P3"][f"C{C}_k{kappa:g}_{b2}"] = {"ell_star_rel": opt["ell_rel"], "ratio": r,
                                                      "eps_effect": e, "m_T": a.get("m_T"),
                                                      "m_star": opt["m"], "tail": a.get("tail_last")}
                print(f"| {C} | {kappa:g} | {b2} | {opt['ell_rel']:.2e} | {r:.2f} | {fmt(e, '.3f')} |"
                      f" {fmt(a.get('m_T'), '.2f')} / {opt['m']:.1f} | {fmt(a.get('tail_last'), '.2f')} |")


def p4():
    """K2 stalls: measured stall margin against the frozen drift zero (scaled by 1/kappa)."""
    print("\n## P4: stall margins against the frozen K2 zero (`phi_alpha0`, x 1/kappa)\n")
    print("| C | kappa | beta2 | FIXED classes | m(T) | predicted zero | ell(T)/ell0 | tail exp |")
    print("|---|---|---|---|---|---|---|---|")
    for C in (128, 32):
        for kappa in K.KAPPAS:
            for b2 in K.BETA2S:
                a = get(C, kappa, 1e-8, b2)
                if a is None:
                    continue
                nfix = a["certificate"].get("FIXED", 0)
                z = K.frozen_zero(C, b2, kappa)
                if nfix == 0 and (z is None or a.get("tail_last", 1) > 0.05):
                    continue
                zs = f"{z[0]:.2f}–{z[1]:.2f}" if z else "none"
                NUM["P4"][f"C{C}_k{kappa:g}_{b2:g}"] = {"fixed": nfix, "m_T": a.get("m_T"),
                                                        "zero": z, "ell_T_rel": a["ell_T_rel"],
                                                        "tail": a.get("tail_last")}
                print(f"| {C} | {kappa:g} | {b2:g} | {nfix}/{C} | {fmt(a.get('m_T'), '.2f')} | {zs} |"
                      f" {a['ell_T_rel']:.3f} | {fmt(a.get('tail_last'), '.3f')} |")


def p5():
    """C = 32: near-optimal convergence for beta2 <= 0.95; beta2 = 0.999 cliff then SGD-like."""
    print("\n## P5 (C = 32, eps = 1e-8): convergence and the beta2 = 0.999 lane\n")
    print("| kappa | lane | t_c | cliff slope d ln ell/dt | ell(T)/ell* | tail exp | FIXED |")
    print("|---|---|---|---|---|---|---|")
    for kappa in K.KAPPAS:
        opt = K.optimum(32, kappa)
        for b2 in list(K.BETA2S) + ["sgd"]:
            a = get(32, kappa, None, 0, opt="sgd") if b2 == "sgd" else get(32, kappa, 1e-8, b2)
            if a is None:
                continue
            NUM["P5"][f"k{kappa:g}_{b2}"] = {"t_c": a["t_c"], "cliff": a.get("cliff_slope_t"),
                                            "ratio_opt": a["ell_T_rel"] / opt["ell_rel"],
                                            "tail": a.get("tail_last")}
            print(f"| {kappa:g} | {b2} | {fmt(a['t_c'])} | {fmt(a.get('cliff_slope_t'), '.3f')} |"
                  f" {a['ell_T_rel'] / opt['ell_rel']:.3g} | {fmt(a.get('tail_last'), '.2f')} |"
                  f" {a['certificate'].get('FIXED', '–')} |")


def finite_d():
    print("\n## Finite-D checks\n")
    print("| check | lane | t_c | t_knee | ell_knee/ell0 | ell(T)/ell0 (T) |")
    print("|---|---|---|---|---|---|")
    for D in (5000, 2500):
        for b2 in K.BETA2S:
            a = get(128, 0.1, 1e-8, b2, D=D)
            if a:
                NUM["finiteD"][f"C128_k0.1_D{D}_{b2:g}"] = a
                print(f"| C128 k0.1 D={D} | {b2:g} | {fmt(a['t_c'])} | {fmt(a.get('t_knee'))} |"
                      f" {fmt(a.get('ell_knee_rel'), '.2e')} | {a['ell_T_rel']:.2e} ({a['T']:.0f}) |")
    for D, tag in ((1000, ""), (4000, "b0.999")):
        a = get(32, 1.0, 1e-8, 0.999, D=D, tag=tag)
        if a:
            NUM["finiteD"][f"C32_k1_D{D}_0.999"] = a
            print(f"| C32 k1 beta2 0.999 D={D} | 0.999 | {fmt(a['t_c'])} | {fmt(a.get('t_knee'))} |"
                  f" {fmt(a.get('ell_knee_rel'), '.2e')} | {a['ell_T_rel']:.2e} ({a['T']:.0f}) |")


# --------------------------------------------------------------------------- #
def field_tests():
    """P1 (frozen field predicts d ln ell/dt) and P6 (late rise = Ito term)."""
    files = sorted(K.FIELD.glob("J_*.npz")) if K.FIELD.exists() else []
    print("\n## P1: frozen field vs simulation, d ln ell/dt on [t_c, t_knee] (and all t >= 0.1)\n")
    print("| group | lane | n cliff | median pred/meas (cliff) | within 25% (cliff) | median pred/meas (all) |"
          " state loss / seed-mean loss |")
    print("|---|---|---|---|---|---|---|")
    rows6 = []
    for f in files:
        z = np.load(f)
        g = K.parse(f.stem)
        d = K.load(f.stem)
        t_all = d["time"]
        for lane in range(z["total"].shape[1]):
            ln = "sgd" if g["opt"] == "sgd" else f"{z['beta2s'][lane]:g}"
            mk = NUM["markers"].get(key(g, ln), {})
            t = z["time"]
            ell_state = z["loss_state"][:, lane].mean(-1)
            pred = z["total"][:, lane].mean(-1) / ell_state             # d ln ell/dt (state)
            # measured slope of the seed-mean class-mean loss at the same records
            ell_mean = d["loss"][:, lane].mean((1, 2))
            lt = np.log(np.maximum(ell_mean, 1e-300))
            k = t_all > 0
            meas_all = np.gradient(lt[k], t_all[k])
            meas = np.interp(t, t_all[k], meas_all)
            jensen = np.interp(t, t_all, ell_mean) / ell_state
            ok = (t >= 0.1) & np.isfinite(pred) & np.isfinite(meas) & (np.abs(meas) > 1e-12)
            cliff = ok & (t >= mk.get("t_c", np.inf)) & (t <= mk.get("t_knee", -np.inf))
            r_all = pred[ok] / meas[ok]
            r_cl = pred[cliff] / meas[cliff]
            rec = {"n_cliff": int(cliff.sum()),
                   "median_cliff": float(np.median(r_cl)) if cliff.any() else np.nan,
                   "within25_cliff": float(np.mean(np.abs(r_cl - 1) < 0.25)) if cliff.any() else np.nan,
                   "median_all": float(np.median(r_all)) if ok.any() else np.nan,
                   "jensen_median": float(np.median(jensen[ok])) if ok.any() else np.nan}
            NUM["P1"][f"{f.stem}_{ln}"] = rec
            print(f"| {f.stem.replace('_R32_s2026_f64', '')} | {ln} | {rec['n_cliff']} |"
                  f" {fmt(rec['median_cliff'], '.3f')} | {fmt(rec['within25_cliff'], '.2f')} |"
                  f" {fmt(rec['median_all'], '.3f')} | {fmt(rec['jensen_median'], '.3f')} |")
            # P6: late states (last 3 field records)
            late = slice(-3, None)
            parts = {q: z[q][late, lane].mean(-1).mean() for q in ("part_O", "part_B", "part_ito", "total")}
            rows6.append((f.stem, ln, parts, float(meas[-1]) * float(ell_state[-1]),
                          mk.get("tail_last", np.nan), mk.get("certificate", {})))
    print("\n## P6: late-time split of d ell/dt (class mean, last 3 field records)\n")
    print("| group | lane | O part | B drift part | Ito part | Ito / -(O + B drift) | predicted total |"
          " measured d ell/dt | tail exp | FIXED |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, ln, p, meas, tail, cert in rows6:
        drift = p["part_O"] + p["part_B"]
        rat = p["part_ito"] / -drift if drift < 0 else np.nan
        NUM["P6"][f"{name}_{ln}"] = {**p, "ito_over_drift": rat, "measured": meas, "tail": tail}
        print(f"| {name.replace('_R32_s2026_f64', '')} | {ln} | {p['part_O']:.2e} | {p['part_B']:.2e} |"
              f" {p['part_ito']:.2e} | {fmt(rat, '.3f')} | {p['total']:.2e} | {meas:.2e} |"
              f" {fmt(tail, '.3f')} | {cert.get('FIXED', '–') if isinstance(cert, dict) else '–'} |")


def main():
    markers_table()
    p2()
    p3()
    p4()
    p5()
    finite_d()
    field_tests()
    (K.OUTDIR / "s3a_numbers.json").write_text(json.dumps(K.jsonable(NUM), indent=1))
    print(f"\nwrote {K.OUTDIR / 's3a_numbers.json'}")


if __name__ == "__main__":
    main()
