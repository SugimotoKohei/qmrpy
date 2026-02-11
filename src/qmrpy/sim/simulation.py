from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True, slots=True)
class SimulationProtocol:
    """Unified simulation input configuration."""

    # model_protocol can optionally be nested:
    #   {"model": {...}, "mrzero": {...}}
    model_protocol: Mapping[str, Any] = field(default_factory=dict)
    simulation_backend: str = "mrzero_bloch"
    noise_model: str = "none"
    noise_sigma: float = 0.0
    noise_snr: float | None = None
    rng: Any | None = None
    fit: bool = False
    fit_kwargs: Mapping[str, Any] = field(default_factory=dict)


def _as_scalar_float(value: Any) -> float | None:
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return None
    if np.isscalar(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _extract_scalar_mapping(values: Mapping[str, Any] | Any) -> dict[str, float]:
    if not isinstance(values, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in values.items():
        scalar = _as_scalar_float(value)
        if scalar is not None:
            out[str(key)] = scalar
    return out


def _resolve_protocol(
    protocol: SimulationProtocol | None,
    *,
    model_protocol: Mapping[str, Any] | None = None,
    simulation_backend: str,
    noise_model: str,
    noise_sigma: float,
    noise_snr: float | None,
    rng: Any | None,
    fit: bool,
    fit_kwargs: Mapping[str, Any] | None,
) -> SimulationProtocol:
    if protocol is None:
        return SimulationProtocol(
            model_protocol=dict(model_protocol or {}),
            simulation_backend=str(simulation_backend),
            noise_model=str(noise_model),
            noise_sigma=float(noise_sigma),
            noise_snr=noise_snr,
            rng=rng,
            fit=bool(fit),
            fit_kwargs=dict(fit_kwargs or {}),
        )
    return SimulationProtocol(
        model_protocol=dict(protocol.model_protocol or {}),
        simulation_backend=str(protocol.simulation_backend),
        noise_model=str(protocol.noise_model),
        noise_sigma=float(protocol.noise_sigma),
        noise_snr=protocol.noise_snr,
        rng=protocol.rng if protocol.rng is not None else rng,
        fit=bool(fit),
        fit_kwargs=dict(fit_kwargs) if fit_kwargs is not None else dict(protocol.fit_kwargs or {}),
    )


def _ensure_model_instance(model: Any, protocol: SimulationProtocol | None) -> Any:
    if model is None:
        raise ValueError("model must not be None")
    if protocol is None or not protocol.model_protocol:
        if isinstance(model, type):
            raise ValueError("model is a class; provide protocol.model_protocol to instantiate it")
        return model

    proto_dict = dict(protocol.model_protocol or {})
    model_block = proto_dict.get("model")
    if model_block is None:
        model_kwargs = proto_dict
    else:
        if not isinstance(model_block, Mapping):
            raise ValueError("model_protocol['model'] must be a mapping")
        model_kwargs = dict(model_block)
    if isinstance(model, type):
        return model(**model_kwargs)
    if callable(model) and not hasattr(model, "forward"):
        return model(**model_kwargs)
    return model


def simulate_single_voxel(
    model: Any,
    *,
    params: Mapping[str, float],
    protocol: SimulationProtocol | None = None,
    model_protocol: Mapping[str, Any] | None = None,
    simulation_backend: str = "mrzero_bloch",
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
    model : object
        qmrpy model instance with ``forward(**params)``, or a model class/factory
        if ``protocol.model_protocol`` is provided.
        If ``fit=True``, it must also have ``fit`` (legacy: ``fit_linear``).
    params : dict
        Parameter dict passed to ``model.forward``.
    protocol : SimulationProtocol, optional
        Unified simulation input configuration. If provided, overrides the noise/fit arguments.
    model_protocol : dict, optional
        Model construction parameters (e.g., TE/TR/FA/TI). Used only when ``protocol`` is None.
    simulation_backend : {"mrzero_bloch", "analytic"}, optional
        Simulation backend. Defaults to MRzero Bloch simulation.
    noise_model : {"none", "gaussian", "rician"}, optional
        Noise model.
    noise_sigma : float, optional
        Noise standard deviation (for ``gaussian`` and ``rician``).
    noise_snr : float, optional
        If set, sigma is derived from peak/snr.
    rng : object, optional
        NumPy Generator-compatible object.
    fit : bool, optional
        If True, fit back the (noisy) signal.
    fit_kwargs : dict, optional
        Passed to the model's fit method.

    Returns
    -------
    dict
        ``signal_clean``, ``signal``, and optional ``fit``.

    Notes
    -----
    - ``mrzero_bloch`` requires ``model_protocol`` to include ``seq_or_path`` (or ``sequence``)
      and either ``data`` or ``data_factory``.
    """
    import numpy as np

    from .noise import add_gaussian_noise, add_rician_noise

    proto = _resolve_protocol(
        protocol,
        model_protocol=model_protocol,
        simulation_backend=simulation_backend,
        noise_model=noise_model,
        noise_sigma=noise_sigma,
        noise_snr=noise_snr,
        rng=rng,
        fit=protocol.fit if protocol is not None else fit,
        fit_kwargs=fit_kwargs,
    )

    fit_kwargs = dict(proto.fit_kwargs or {})
    backend = str(proto.simulation_backend).lower().strip()
    model_instance = None
    if backend == "analytic":
        model_instance = _ensure_model_instance(model, proto)
    elif backend == "mrzero_bloch":
        if proto.fit:
            model_instance = _ensure_model_instance(model, proto)
    else:
        raise ValueError(f"unknown simulation_backend: {proto.simulation_backend}")

    if backend == "analytic":
        assert model_instance is not None
        signal_clean = np.asarray(
            model_instance.forward(**{k: float(v) for k, v in params.items()}),
            dtype=np.float64,
        )
    else:
        from .mrzero import simulate_bloch

        mp = dict(proto.model_protocol or {})
        mrzero_block = mp.get("mrzero", mp)
        if not isinstance(mrzero_block, Mapping):
            raise ValueError("model_protocol['mrzero'] must be a mapping")
        mrzero_cfg = dict(mrzero_block)
        seq_or_path = mrzero_cfg.get("seq_or_path", mrzero_cfg.get("sequence"))
        data_factory = mrzero_cfg.get("data_factory")
        data = mrzero_cfg.get("data")
        if callable(data_factory):
            data = data_factory({k: float(v) for k, v in params.items()})
        if seq_or_path is None or data is None:
            raise ValueError("mrzero_bloch requires model_protocol['seq_or_path' or 'sequence'] and data/data_factory")

        mrzero_kwargs = {
            "spin_count": mrzero_cfg.get("spin_count", 5000),
            "perfect_spoiling": mrzero_cfg.get("perfect_spoiling", False),
            "print_progress": mrzero_cfg.get("print_progress", True),
            "spin_dist": mrzero_cfg.get("spin_dist", "rand"),
            "r2_seed": mrzero_cfg.get("r2_seed"),
        }
        signal_clean = np.asarray(simulate_bloch(seq_or_path, data, **mrzero_kwargs))
        if np.iscomplexobj(signal_clean):
            signal_clean = np.abs(signal_clean)
        signal_clean = np.asarray(signal_clean, dtype=np.float64)
        if signal_clean.ndim > 1:
            signal_clean = signal_clean.reshape(signal_clean.shape[0], -1)
            if signal_clean.shape[1] > 1:
                signal_clean = signal_clean.mean(axis=1)
            else:
                signal_clean = signal_clean[:, 0]

    nm = proto.noise_model.lower().strip()
    if nm in {"none", "", "no"}:
        signal = signal_clean
    else:
        if proto.rng is None:
            rng = np.random.default_rng(0)
        else:
            rng = proto.rng

        sigma = float(proto.noise_sigma)
        if proto.noise_snr is not None:
            snr = float(proto.noise_snr)
            if snr <= 0:
                raise ValueError("noise_snr must be > 0")
            peak = float(np.max(np.abs(signal_clean)))
            sigma = 0.0 if peak == 0.0 else peak / snr

        if nm == "gaussian":
            signal = add_gaussian_noise(signal_clean, sigma=sigma, rng=rng)
        elif nm == "rician":
            signal = add_rician_noise(signal_clean, sigma=sigma, rng=rng)
        else:
            raise ValueError(f"unknown noise_model: {proto.noise_model}")

    out: dict[str, Any] = {"signal_clean": signal_clean, "signal": signal}
    if proto.fit:
        if model_instance is None:
            raise ValueError("fit=True requires a model instance")
        out["fit"] = _fit_model(model_instance, signal, fit_kwargs=fit_kwargs)
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
    protocol: SimulationProtocol | None = None,
    model_protocol: Mapping[str, Any] | None = None,
    simulation_backend: str = "mrzero_bloch",
    noise_model: str = "gaussian",
    noise_sigma: float = 0.0,
    noise_snr: float | None = None,
    rng: Any | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One-parameter-at-a-time sensitivity analysis (qMRLab: SimVary).

    Parameters
    ----------
    model : object
        qmrpy model instance.
    nominal_params : dict
        Nominal parameter values.
    vary_param : str
        Parameter to vary.
    lb, ub : float
        Lower/upper bounds for the varied parameter.
    n_steps : int, optional
        Number of steps.
    n_runs : int, optional
        Number of runs per step.
    protocol : SimulationProtocol, optional
        Unified simulation input configuration. If provided, overrides the noise/fit arguments.
    model_protocol : dict, optional
        Model construction parameters (e.g., TE/TR/FA/TI). Used only when ``protocol`` is None.
    simulation_backend : {"mrzero_bloch", "analytic"}, optional
        Simulation backend. Defaults to MRzero Bloch simulation.
    noise_model : {"gaussian", "rician"}, optional
        Noise model.
    noise_sigma : float, optional
        Noise standard deviation.
    noise_snr : float, optional
        If set, sigma is derived from peak/snr.
    rng : object, optional
        NumPy Generator-compatible object.
    fit_kwargs : dict, optional
        Passed to the model's fit method.

    Returns
    -------
    dict
        ``x`` values, per-run ``fit``, and aggregated ``mean``/``std``.
    """
    import numpy as np

    if n_steps <= 1:
        raise ValueError("n_steps must be >= 2")
    if n_runs <= 0:
        raise ValueError("n_runs must be >= 1")

    proto = _resolve_protocol(
        protocol,
        model_protocol=model_protocol,
        simulation_backend=simulation_backend,
        noise_model=noise_model,
        noise_sigma=noise_sigma,
        noise_snr=noise_snr,
        rng=rng,
        fit=True,
        fit_kwargs=fit_kwargs,
    )
    fit_kwargs = dict(proto.fit_kwargs or {})
    if proto.rng is None:
        rng = np.random.default_rng(0)
    else:
        rng = proto.rng
    model_instance = _ensure_model_instance(model, proto) if proto.fit else None

    x = np.linspace(float(lb), float(ub), int(n_steps), dtype=np.float64)

    # Probe one fit to establish the output keys.
    probe_params = dict(nominal_params)
    probe_params[vary_param] = float(x[0])
    probe = simulate_single_voxel(
        model_instance,
        params=probe_params,
        protocol=proto,
    )["fit"]

    fit_store: dict[str, dict[str, Any]] = {
        "params": {k: np.full((n_steps, n_runs), np.nan, dtype=np.float64) for k in probe.get("params", {})},
        "quality": {k: np.full((n_steps, n_runs), np.nan, dtype=np.float64) for k in probe.get("quality", {})},
        "diagnostics": {
            k: np.full((n_steps, n_runs), np.nan, dtype=np.float64)
            for k in probe.get("diagnostics", {})
        },
    }

    for i, xv in enumerate(x):
        for r in range(n_runs):
            params = dict(nominal_params)
            params[vary_param] = float(xv)
            fitted = simulate_single_voxel(
                model_instance,
                params=params,
                protocol=proto,
            )["fit"]
            for section in ("params", "quality", "diagnostics"):
                section_values = fitted.get(section, {})
                if not isinstance(section_values, Mapping):
                    continue
                for key, value in section_values.items():
                    scalar = _as_scalar_float(value)
                    if scalar is None:
                        continue
                    store = fit_store.setdefault(section, {})
                    if key not in store:
                        store[key] = np.full((n_steps, n_runs), np.nan, dtype=np.float64)
                    store[key][i, r] = scalar

    mean = {
        section: {k: np.mean(v, axis=1) for k, v in values.items()}
        for section, values in fit_store.items()
    }
    std = {
        section: {k: np.std(v, axis=1, ddof=0) for k, v in values.items()}
        for section, values in fit_store.items()
    }

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
    protocol: SimulationProtocol | None = None,
    model_protocol: Mapping[str, Any] | None = None,
    simulation_backend: str = "mrzero_bloch",
    noise_model: str = "gaussian",
    noise_sigma: float = 0.0,
    noise_snr: float | None = None,
    rng: Any | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Multi-voxel simulation given parameter distributions (qMRLab: SimRnd).

    Parameters
    ----------
    model : object
        qmrpy model instance.
    true_params : dict
        Mapping of parameter name -> array-like (length n_samples) or scalar.
    protocol : SimulationProtocol, optional
        Unified simulation input configuration. If provided, overrides the noise/fit arguments.
    model_protocol : dict, optional
        Model construction parameters (e.g., TE/TR/FA/TI). Used only when ``protocol`` is None.
    simulation_backend : {"mrzero_bloch", "analytic"}, optional
        Simulation backend. Defaults to MRzero Bloch simulation.
    noise_model : {"gaussian", "rician"}, optional
        Noise model.
    noise_sigma : float, optional
        Noise standard deviation.
    noise_snr : float, optional
        If set, sigma is derived from peak/snr.
    rng : object, optional
        NumPy Generator-compatible object.
    fit_kwargs : dict, optional
        Passed to the model's fit method.

    Returns
    -------
    dict
        ``true``, ``hat``, ``err`` dicts and ``metrics``.
    """
    import numpy as np

    proto = _resolve_protocol(
        protocol,
        model_protocol=model_protocol,
        simulation_backend=simulation_backend,
        noise_model=noise_model,
        noise_sigma=noise_sigma,
        noise_snr=noise_snr,
        rng=rng,
        fit=True,
        fit_kwargs=fit_kwargs,
    )
    fit_kwargs = dict(proto.fit_kwargs or {})
    if proto.rng is None:
        rng = np.random.default_rng(0)
    else:
        rng = proto.rng
    model_instance = _ensure_model_instance(model, proto) if proto.fit else None

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
        model_instance,
        params=p0,
        protocol=proto,
    )["fit"]
    probe_params = probe.get("params", {}) if isinstance(probe, Mapping) else {}
    for k in probe_params:
        hat[k] = np.full(n_samples, np.nan, dtype=np.float64)

    for i in range(n_samples):
        params = {k: float(true[k][i]) for k in keys}
        fitted = simulate_single_voxel(
            model_instance,
            params=params,
            protocol=proto,
        )["fit"]
        fitted_params = fitted.get("params", {}) if isinstance(fitted, Mapping) else {}
        for key, value in fitted_params.items():
            if key not in hat:
                hat[key] = np.full(n_samples, np.nan, dtype=np.float64)
            scalar = _as_scalar_float(value)
            if scalar is not None:
                hat[key][i] = scalar

    for k, v_hat in hat.items():
        if k in true:
            err[k] = v_hat - true[k]

    metrics: dict[str, float] = {"n_samples": float(n_samples)}
    for k, e in err.items():
        metrics[f"{k}_mae"] = float(np.mean(np.abs(e)))
        metrics[f"{k}_rmse"] = float(np.sqrt(np.mean(e**2)))

    return {
        "true": {"params": true},
        "hat": {"params": hat},
        "err": {"params": err},
        "metrics": metrics,
    }


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


def sim_vary(
    model: Any,
    runs: int,
    opt_table: Any,
    opts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """qMRLab-like SimVary wrapper.

    Notes
    -----
    - `opt_table` is expected to have arrays/lists: fx, st, lb, ub, and an `xnames` list.
    - This wrapper performs one-parameter-at-a-time sweeps for each non-fixed parameter.
    """
    import numpy as np

    if opts is None:
        opts = {"SNR": 50}

    if runs <= 0:
        raise ValueError("runs must be >= 1")

    # qMRLab default: Nsteps=10
    n_steps = int(opts.get("Nsteps", 10))

    xnames = list(getattr(opt_table, "xnames", getattr(model, "xnames", [])))
    if not xnames:
        raise ValueError("opt_table.xnames (or model.xnames) is required for sim_vary")

    fx = np.asarray(getattr(opt_table, "fx"), dtype=bool)
    st = np.asarray(getattr(opt_table, "st"), dtype=np.float64)
    lb = np.asarray(getattr(opt_table, "lb"), dtype=np.float64)
    ub = np.asarray(getattr(opt_table, "ub"), dtype=np.float64)

    snr = float(opts.get("SNR", 50.0))

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
            simulation_backend="analytic",
            noise_model="gaussian",
            noise_sigma=0.0,
            noise_snr=(snr if snr > 0 else None),
            fit_kwargs=opts.get("fit_kwargs", None),
        )
        # qMRLab adds ground_truth per fitted param name
        for k in res.get("mean", {}).get("params", {}):
            res.setdefault("ground_truth", {})[k] = float(st[xnames.index(k)]) if k in xnames else None
        results[name] = res

    return results


def sim_rnd(model: Any, rnd_param: Mapping[str, Any], opt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """qMRLab-like SimRnd wrapper (multi-voxel distribution simulation)."""
    import numpy as np

    if opt is None:
        opt = {"SNR": 50}

    snr = float(opt.get("SNR", 50.0))

    out = simulate_parameter_distribution(
        model,
        true_params=rnd_param,
        simulation_backend="analytic",
        noise_model="gaussian",
        noise_sigma=0.0,
        noise_snr=(snr if snr > 0 else None),
        fit_kwargs=opt.get("fit_kwargs", None),
    )

    # qMRLab-style error stats
    error: dict[str, Any] = {}
    pct_error: dict[str, Any] = {}
    mpe: dict[str, float] = {}
    rmse: dict[str, float] = {}
    nrmse: dict[str, float] = {}

    true_params = out.get("true", {}).get("params", {})
    hat_params = out.get("hat", {}).get("params", {})

    for k in set(true_params).intersection(hat_params):
        e = np.asarray(hat_params[k] - true_params[k], dtype=np.float64)
        error[k] = e
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_error[k] = 100.0 * e / np.asarray(true_params[k], dtype=np.float64)
        mpe[k] = float(np.nanmean(pct_error[k]))
        rmse[k] = float(np.sqrt(np.nanmean(e**2)))
        denom = float(np.max(true_params[k]) - np.min(true_params[k]))
        nrmse[k] = float(rmse[k] / denom) if denom != 0 else float("nan")

    out.update({"error": error, "pct_error": pct_error, "mpe": mpe, "rmse": rmse, "nrmse": nrmse})
    return out


def sim_fisher_matrix(
    obj: Any,
    prot: Any,  # kept for signature compatibility; protocol is expected to be embedded in the model
    x: Any,
    variables: list[int] | None = None,
    sigma: float = 0.1,
) -> Any:
    """qMRLab-like SimFisherMatrix wrapper.

    In qMRLab, `prot` is stored into `obj.Prot` before evaluating `equation`.
    In qmrpy, the protocol is typically embedded in the model instance, so `prot` is unused.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64).reshape(-1)

    xnames = list(getattr(obj, "xnames", []))
    if not xnames:
        raise ValueError("obj.xnames is required for sim_fisher_matrix")

    if variables is None:
        variables = list(range(1, min(5, len(xnames)) + 1))

    vars0 = [xnames[i - 1] for i in variables]
    params = {k: float(v) for k, v in zip(xnames, x, strict=True)}
    return fisher_information_gaussian(obj, params=params, variables=vars0, sigma=float(sigma))


def sim_crlb(
    obj: Any,
    prot: Any,
    xvalues: Any,
    sigma: float = 0.1,
    vars: list[int] | None = None,
) -> tuple[float, list[str], Any, Any]:
    """qMRLab-like SimCRLB.

    Returns
    -------
    (F, xnames, crlb, fall)
    """
    import numpy as np

    xnames_all = list(getattr(obj, "xnames", []))
    if not xnames_all:
        raise ValueError("obj.xnames is required for sim_crlb")

    xvalues = np.asarray(xvalues, dtype=np.float64)
    if xvalues.ndim == 1:
        xvalues = xvalues[None, :]

    if vars is None:
        fx = np.asarray(getattr(obj, "fx", np.zeros(len(xnames_all), dtype=bool)), dtype=bool)
        variables = [i + 1 for i, fixed in enumerate(fx) if not fixed]
    else:
        variables = list(vars)

    var_names = [xnames_all[i - 1] for i in variables]

    f_each = np.zeros((xvalues.shape[0], len(var_names)), dtype=np.float64)
    for ix in range(xvalues.shape[0]):
        params = {k: float(v) for k, v in zip(xnames_all, xvalues[ix, :], strict=True)}
        fisher = fisher_information_gaussian(obj, params=params, variables=var_names, sigma=float(sigma))
        crlb = np.linalg.inv(np.asarray(fisher, dtype=np.float64) + np.eye(len(var_names)) * np.finfo(float).eps)
        xsel = np.array([params[n] for n in var_names], dtype=np.float64)
        f_each[ix, :] = np.diag(crlb) / (xsel**2)

    fall = f_each.reshape(-1)
    f = float(np.mean(fall))
    return f, var_names, crlb, fall


def _fit_model(model: Any, signal: Any, *, fit_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if hasattr(model, "fit"):
        fitted = model.fit(signal, **dict(fit_kwargs))
    elif hasattr(model, "fit_linear"):
        fitted = model.fit_linear(signal, **dict(fit_kwargs))
    else:
        raise TypeError("model must provide fit(...) or fit_linear(...)")

    params_attr = getattr(fitted, "params", None)
    quality_attr = getattr(fitted, "quality", None)
    diagnostics_attr = getattr(fitted, "diagnostics", None)
    if isinstance(params_attr, Mapping):
        return {
            "params": _extract_scalar_mapping(params_attr),
            "quality": _extract_scalar_mapping(quality_attr if isinstance(quality_attr, Mapping) else {}),
            "diagnostics": _extract_scalar_mapping(
                diagnostics_attr if isinstance(diagnostics_attr, Mapping) else {}
            ),
        }

    fit_dict = dict(fitted)
    if "params" in fit_dict and isinstance(fit_dict["params"], Mapping):
        return {
            "params": _extract_scalar_mapping(fit_dict.get("params", {})),
            "quality": _extract_scalar_mapping(fit_dict.get("quality", {})),
            "diagnostics": _extract_scalar_mapping(fit_dict.get("diagnostics", {})),
        }

    return {
        "params": _extract_scalar_mapping(fit_dict),
        "quality": {},
        "diagnostics": {},
    }
