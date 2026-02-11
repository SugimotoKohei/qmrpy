from __future__ import annotations


def main() -> None:
    import numpy as np

    from qmrpy import fit_t2_epg, simulate_t2_epg
    from qmrpy.models.t2 import T2EPG

    t2_true = 80.0
    b1_scale = 0.95

    model = T2EPG(n_te=16, te_ms=10.0, t1_ms=1000.0, alpha_deg=180.0)

    signal = model.forward(m0=1.0, t2_ms=t2_true, b1=b1_scale)
    fit = model.fit(signal, b1=b1_scale)
    print(f"object_fit: t2_ms={fit['t2_ms']:.3f}, m0={fit['m0']:.3f}")

    signal_func = simulate_t2_epg(
        m0=1.0,
        t2_ms=t2_true,
        n_te=16,
        te_ms=10.0,
        t1_ms=1000.0,
        alpha_deg=180.0,
        b1=b1_scale,
    )
    fit_func = fit_t2_epg(
        signal_func,
        n_te=16,
        te_ms=10.0,
        t1_ms=1000.0,
        alpha_deg=180.0,
        b1=b1_scale,
    )
    print(f"functional_fit: t2_ms={fit_func['t2_ms']:.3f}, m0={fit_func['m0']:.3f}")

    signal_lo = model.forward(m0=1.0, t2_ms=t2_true, b1=0.9)
    signal_hi = model.forward(m0=1.0, t2_ms=t2_true, b1=1.0)
    img = np.stack([signal_lo, signal_hi], axis=0).reshape(2, 1, -1)
    b1_map = np.array([[0.9], [1.0]], dtype=float)
    fit_img = model.fit_image(img, b1_map=b1_map)
    print(f"image_fit: t2_ms={fit_img['t2_ms'].ravel().tolist()}")


if __name__ == "__main__":
    main()
