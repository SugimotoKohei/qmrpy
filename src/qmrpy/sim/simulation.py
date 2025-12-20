from __future__ import annotations

from typing import Any, Callable, Mapping


def simulate_single_voxel(
    model: Any,
    *,
    params: Mapping[str, float],
    noise_model: str = "none",
    noise_sigma: float = 0.0,
    noise_snr: float | None = None,
    rng: Any | None = None,
    fit: bool = False,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate a single-voxel signal (optionally add noise and fit it back).

    This is the Python counterpart of qMRLab's *SingleVoxel* addon at the core level.

    Parameters
    ----------
    model:
        qmrpy model instance with `.forward(**params)`.
        If `fit=True`, it must also have either `.fit_linear(signal, **kwargs)` or `.fit(signal, **kwargs)`.
    params:
        Parameter dict passed to `model.forward`.
    noise_model:
        "none" | "gaussian" | "rician".
    noise_sigma:
        Noise standard deviation (for "gaussian" and "rician").
    rng:
        NumPy Generator-compatible object.
    fit:
        If True, fit back the (noisy) signal.
    fit_kwargs:
        Passed to the model's fit method.
    """
    import numpy as np

    from .noise import add_gaussian_noise, add_rician_noise

    if fit_kwargs is None:
        fit_kwargs = {}

    signal_clean = np.asarray(model.forward(**{k: float(v) for k, v in params.items()}), dtype=np.float64)

    nm = noise_model.lower().strip()
    if nm in {"none", "", "no"}:
        signal = signal_clean
    else:
        if rng is None:
            rng = np.random.default_rng(0)

        sigma = float(noise_sigma)
        if noise_snr is not None:
            snr = float(noise_snr)
            if snr <= 0:
                raise ValueError("noise_snr must be > 0")
            peak = float(np.max(np.abs(signal_clean)))
            sigma = 0.0 if peak == 0.0 else peak / snr

        if nm == "gaussian":
            signal = add_gaussian_noise(signal_clean, sigma=sigma, rng=rng)
        elif nm == "rician":
            signal = add_rician_noise(signal_clean, sigma=sigma, rng=rng)
        else:
            raise ValueError(f"unknown noise_model: {noise_model}")

    out: dict[str, Any] = {"signal_clean": signal_clean, "signal": signal}
    if fit:
        out["fit"] = _fit_model(model, signal, fit_kwargs=fit_kwargs)
    return out


def sensitivity_analysis(
    model: Any,
    *,
    nominal_params: Mapping[str, float],
    vary_param: str,
    lb: float,
    ub: float,
    n_steps: int = 10,
    n_runs: int = 20,
    noise_model: str = "gaussian",
    noise_sigma: float = 0.0,
    noise_snr: float | None = None,
    rng: Any | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One-parameter-at-a-time sensitivity analysis (qMRLab: SimVary).

    Returns a dict with:
      - x: varied parameter values (shape: [n_steps])
      - fit: per-run fitted params (dict of arrays, each shape: [n_steps, n_runs])
      - mean/std: aggregated over runs (dict of arrays, each shape: [n_steps])
    """
    import numpy as np

    if n_steps <= 1:
        raise ValueError("n_steps must be >= 2")
    if n_runs <= 0:
        raise ValueError("n_runs must be >= 1")

    if fit_kwargs is None:
        fit_kwargs = {}
    if rng is None:
        rng = np.random.default_rng(0)

    x = np.linspace(float(lb), float(ub), int(n_steps), dtype=np.float64)

    # Probe one fit to establish the output keys.
    probe_params = dict(nominal_params)
    probe_params[vary_param] = float(x[0])
    probe = simulate_single_voxel(
        model,
        params=probe_params,
        noise_model=noise_model,
        noise_sigma=noise_sigma,
        noise_snr=noise_snr,
        rng=rng,
        fit=True,
        fit_kwargs=fit_kwargs,
    )["fit"]

    fit_store: dict[str, Any] = {k: np.full((n_steps, n_runs), np.nan, dtype=np.float64) for k in probe}

    for i, xv in enumerate(x):
        for r in range(n_runs):
            params = dict(nominal_params)
            params[vary_param] = float(xv)
            fitted = simulate_single_voxel(
                model,
                params=params,
                noise_model=noise_model,
                noise_sigma=noise_sigma,
                noise_snr=noise_snr,
                rng=rng,
                fit=True,
                fit_kwargs=fit_kwargs,
            )["fit"]
            for k, v in fitted.items():
                fit_store[k][i, r] = float(v)

    mean = {k: np.mean(v, axis=1) for k, v in fit_store.items()}
    std = {k: np.std(v, axis=1, ddof=0) for k, v in fit_store.items()}

    return {
        "vary_param": str(vary_param),
        "x": x,
        "fit": fit_store,
        "mean": mean,
        "std": std,
    }


def simulate_parameter_distribution(
    model: Any,
    *,
    true_params: Mapping[str, Any],
    noise_model: str = "gaussian",
    noise_sigma: float = 0.0,
    noise_snr: float | None = None,
    rng: Any | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Multi-voxel simulation given parameter distributions (qMRLab: SimRnd).

    Parameters
    ----------
    true_params:
        Mapping of parameter name -> array-like (length n_samples) OR scalar.

    Returns
    -------
    dict with keys:
      - true, hat, err: dict[str, ndarray]
      - metrics: dict[str, float]
    """
    import numpy as np

    if fit_kwargs is None:
        fit_kwargs = {}
    if rng is None:
        rng = np.random.default_rng(0)

    keys = list(true_params.keys())
    values = [np.asarray(true_params[k], dtype=np.float64) for k in keys]
    n_samples = int(max(v.size for v in values))

    true: dict[str, Any] = {}
    for k, v in zip(keys, values, strict=True):
        if v.ndim == 0:
            true[k] = np.full(n_samples, float(v), dtype=np.float64)
        else:
            if int(v.size) != n_samples:
                raise ValueError("all non-scalar true_params must have the same length")
            true[k] = v.astype(np.float64)

    hat: dict[str, Any] = {}
    err: dict[str, Any] = {}

    # Determine output keys from one fit.
    p0 = {k: float(true[k][0]) for k in keys}
    probe = simulate_single_voxel(
        model,
        params=p0,
        noise_model=noise_model,
        noise_sigma=noise_sigma,
        noise_snr=noise_snr,
        rng=rng,
        fit=True,
        fit_kwargs=fit_kwargs,
    )["fit"]
    for k in probe:
        hat[k] = np.full(n_samples, np.nan, dtype=np.float64)

    for i in range(n_samples):
        params = {k: float(true[k][i]) for k in keys}
        fitted = simulate_single_voxel(
            model,
            params=params,
            noise_model=noise_model,
            noise_sigma=noise_sigma,
            noise_snr=noise_snr,
            rng=rng,
            fit=True,
            fit_kwargs=fit_kwargs,
        )["fit"]
        for k, v in fitted.items():
            hat[k][i] = float(v)

    for k, v_hat in hat.items():
        if k in true:
            err[k] = v_hat - true[k]

    metrics: dict[str, float] = {"n_samples": float(n_samples)}
    for k, e in err.items():
        metrics[f"{k}_mae"] = float(np.mean(np.abs(e)))
        metrics[f"{k}_rmse"] = float(np.sqrt(np.mean(e**2)))

    return {"true": true, "hat": hat, "err": err, "metrics": metrics}


def fisher_information_gaussian(
    model: Any,
    *,
    params: Mapping[str, float],
    variables: list[str] | None = None,
    sigma: float,
    step_rel: float = 1e-2,
    step_abs: float = 1e-10,
) -> Any:
    """Compute Fisher information matrix under i.i.d. Gaussian noise.

    This mirrors qMRLab's SimFisherMatrix (finite-difference Jacobian).
    """
    import numpy as np

    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    if variables is None:
        variables = list(params.keys())

    base = {k: float(v) for k, v in params.items()}
    s0 = np.asarray(model.forward(**base), dtype=np.float64).reshape(-1)

    j = np.zeros((s0.size, len(variables)), dtype=np.float64)
    for col, name in enumerate(variables):
        x = dict(base)
        v0 = float(x[name])
        dv = max(float(step_abs), abs(v0) * float(step_rel))
        x[name] = v0 + dv
        s1 = np.asarray(model.forward(**x), dtype=np.float64).reshape(-1)
        j[:, col] = (s1 - s0) / dv

    f = (j.T @ j) / (float(sigma) ** 2)
    return f


def crlb_from_fisher(
    fisher: Any,
    *,
    variables: list[str] | None = None,
    return_matrix: bool = False,
) -> Any:
    """Compute CRLB (covariance) from Fisher information.

    If `variables` is provided and `return_matrix=False`, returns a dict of per-parameter
    variances (diag of CRLB). Otherwise returns the full covariance matrix (ndarray).
    """
    import numpy as np

    f = np.asarray(fisher, dtype=np.float64)
    cov = np.linalg.inv(f + np.eye(f.shape[0]) * 0.0)
    if return_matrix or variables is None:
        return cov
    if len(variables) != cov.shape[0]:
        raise ValueError("variables length must match Fisher dimension")
    return {str(name): float(cov[i, i]) for i, name in enumerate(variables)}


def crlb_cov_mean(
    model: Any,
    *,
    params: Mapping[str, float],
    variables: list[str] | None = None,
    sigma: float,
) -> float:
    """Mean COV proxy used in qMRLab's SimCRLB: mean(diag(CRLB)/x^2)."""
    import numpy as np

    if variables is None:
        variables = list(params.keys())

    fisher = fisher_information_gaussian(model, params=params, variables=variables, sigma=sigma)
    crlb = np.linalg.inv(fisher + np.eye(len(variables)) * 1e-15)
    x = np.array([float(params[v]) for v in variables], dtype=np.float64)
    return float(np.mean(np.diag(crlb) / (x**2)))


def optimize_protocol_grid(
    model_factory: Callable[[Any], Any],
    *,
    protocol_candidates: list[Any],
    params: Mapping[str, float],
    variables: list[str] | None = None,
    sigma: float,
) -> dict[str, Any]:
    """Grid-search protocol optimization using CRLB objective (qMRLab: SimProtocolOpt).

    `protocol_candidates` is user-defined; it is passed to `model_factory`.
    """
    if not protocol_candidates:
        raise ValueError("protocol_candidates must not be empty")

    best_idx = -1
    best_obj = float("inf")
    objs: list[float] = []

    for i, proto in enumerate(protocol_candidates):
        model = model_factory(proto)
        obj = crlb_cov_mean(model, params=params, variables=variables, sigma=sigma)
        objs.append(float(obj))
        if obj < best_obj:
            best_obj = float(obj)
            best_idx = int(i)

    return {
        "best_protocol": protocol_candidates[best_idx],
        "best_objective": float(best_obj),
        "objectives": objs,
        "best_index": int(best_idx),
    }


# -----------------------------------------------------------------------------
# qMRLab-compatible wrappers (naming and return shapes)
# -----------------------------------------------------------------------------


def SimVary(
    model: Any,
    runs: int,
    OptTable: Any,
    Opts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """qMRLab-like SimVary wrapper.

    Notes
    -----
    - `OptTable` is expected to have arrays/lists: fx, st, lb, ub, and an `xnames` list.
    - This wrapper performs one-parameter-at-a-time sweeps for each non-fixed parameter.
    """
    import numpy as np

    if Opts is None:
        Opts = {"SNR": 50}

    if runs <= 0:
        raise ValueError("runs must be >= 1")

    # qMRLab default: Nsteps=10
    n_steps = int(Opts.get("Nsteps", 10))

    xnames = list(getattr(OptTable, "xnames", getattr(model, "xnames", [])))
    if not xnames:
        raise ValueError("OptTable.xnames (or model.xnames) is required for SimVary")

    fx = np.asarray(getattr(OptTable, "fx"), dtype=bool)
    st = np.asarray(getattr(OptTable, "st"), dtype=np.float64)
    lb = np.asarray(getattr(OptTable, "lb"), dtype=np.float64)
    ub = np.asarray(getattr(OptTable, "ub"), dtype=np.float64)

    snr = float(Opts.get("SNR", 50.0))

    results: dict[str, Any] = {}
    for i, name in enumerate(xnames):
        if bool(fx[i]):
            continue
        res = sensitivity_analysis(
            model,
            nominal_params={k: float(v) for k, v in zip(xnames, st, strict=True)},
            vary_param=name,
            lb=float(lb[i]),
            ub=float(ub[i]),
            n_steps=n_steps,
            n_runs=int(runs),
            noise_model="gaussian",
            noise_sigma=0.0,
            noise_snr=(snr if snr > 0 else None),
            fit_kwargs=Opts.get("fit_kwargs", None),
        )
        # qMRLab adds GroundTruth per fitted param name
        for k in res["mean"]:
            res.setdefault("GroundTruth", {})[k] = float(st[xnames.index(k)]) if k in xnames else None
        results[name] = res

    return results


def SimRnd(Model: Any, RndParam: Mapping[str, Any], Opt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """qMRLab-like SimRnd wrapper (multi-voxel distribution simulation)."""
    import numpy as np

    if Opt is None:
        Opt = {"SNR": 50}

    snr = float(Opt.get("SNR", 50.0))

    out = simulate_parameter_distribution(
        Model,
        true_params=RndParam,
        noise_model="gaussian",
        noise_sigma=0.0,
        noise_snr=(snr if snr > 0 else None),
        fit_kwargs=Opt.get("fit_kwargs", None),
    )

    # qMRLab-style error stats
    error: dict[str, Any] = {}
    pct_error: dict[str, Any] = {}
    mpe: dict[str, float] = {}
    rmse: dict[str, float] = {}
    nrmse: dict[str, float] = {}

    for k in set(out["true"]).intersection(out["hat"]):
        e = np.asarray(out["hat"][k] - out["true"][k], dtype=np.float64)
        error[k] = e
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_error[k] = 100.0 * e / np.asarray(out["true"][k], dtype=np.float64)
        mpe[k] = float(np.nanmean(pct_error[k]))
        rmse[k] = float(np.sqrt(np.nanmean(e**2)))
        denom = float(np.max(out["true"][k]) - np.min(out["true"][k]))
        nrmse[k] = float(rmse[k] / denom) if denom != 0 else float("nan")

    out.update({"Error": error, "PctError": pct_error, "MPE": mpe, "RMSE": rmse, "NRMSE": nrmse})
    return out


def SimFisherMatrix(
    obj: Any,
    Prot: Any,  # kept for signature compatibility; protocol is expected to be embedded in the model
    x: Any,
    variables: list[int] | None = None,
    sigma: float = 0.1,
) -> Any:
    """qMRLab-like SimFisherMatrix wrapper.

    In qMRLab, `Prot` is stored into `obj.Prot` before evaluating `equation`.
    In qmrpy, the protocol is typically embedded in the model instance, so `Prot` is unused.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64).reshape(-1)

    xnames = list(getattr(obj, "xnames", []))
    if not xnames:
        raise ValueError("obj.xnames is required for SimFisherMatrix")

    if variables is None:
        variables = list(range(1, min(5, len(xnames)) + 1))

    vars0 = [xnames[i - 1] for i in variables]
    params = {k: float(v) for k, v in zip(xnames, x, strict=True)}
    return fisher_information_gaussian(obj, params=params, variables=vars0, sigma=float(sigma))


def SimCRLB(
    obj: Any,
    Prot: Any,
    xvalues: Any,
    sigma: float = 0.1,
    vars: list[int] | None = None,
) -> tuple[float, list[str], Any, Any]:
    """qMRLab-like SimCRLB.

    Returns
    -------
    (F, xnames, CRLB, Fall)
    """
    import numpy as np

    xnames_all = list(getattr(obj, "xnames", []))
    if not xnames_all:
        raise ValueError("obj.xnames is required for SimCRLB")

    xvalues = np.asarray(xvalues, dtype=np.float64)
    if xvalues.ndim == 1:
        xvalues = xvalues[None, :]

    if vars is None:
        fx = np.asarray(getattr(obj, "fx", np.zeros(len(xnames_all), dtype=bool)), dtype=bool)
        variables = [i + 1 for i, fixed in enumerate(fx) if not fixed]
    else:
        variables = list(vars)

    var_names = [xnames_all[i - 1] for i in variables]

    F_each = np.zeros((xvalues.shape[0], len(var_names)), dtype=np.float64)
    for ix in range(xvalues.shape[0]):
        params = {k: float(v) for k, v in zip(xnames_all, xvalues[ix, :], strict=True)}
        fisher = fisher_information_gaussian(obj, params=params, variables=var_names, sigma=float(sigma))
        CRLB = np.linalg.inv(np.asarray(fisher, dtype=np.float64) + np.eye(len(var_names)) * np.finfo(float).eps)
        xsel = np.array([params[n] for n in var_names], dtype=np.float64)
        F_each[ix, :] = np.diag(CRLB) / (xsel**2)

    Fall = F_each.reshape(-1)
    F = float(np.mean(Fall))
    return F, var_names, CRLB, Fall


def _fit_model(model: Any, signal: Any, *, fit_kwargs: Mapping[str, Any]) -> dict[str, float]:
    if hasattr(model, "fit_linear"):
        fitted = model.fit_linear(signal, **dict(fit_kwargs))
    elif hasattr(model, "fit"):
        fitted = model.fit(signal, **dict(fit_kwargs))
    else:
        raise TypeError("model must provide fit_linear(...) or fit(...)")

    return {k: float(v) for k, v in dict(fitted).items() if isinstance(v, (int, float))}
