from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

import numpy as np


def _as_4d_float_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3:
        # Assume (X, Y, T) or (X, Y, Z)?
        # Usually DWI is 4D (X, Y, Z, B)
        # If 3D input, maybe (X, Y, T)?
        # We enforce 4D for consistency with qMRLab logic (x,y,z,M)
        # If user provides 3D (x,y,t), we extend to (x,y,1,t)
        if array.shape[-1] < array.shape[0] and array.shape[-1] < array.shape[1]:
            # heuristic: last dim is time/b-value
            array = array[:, :, np.newaxis, :]
        else:
            raise ValueError(
                f"{name} must be 4D (x,y,z,t), got 3D but ambiguous shape {array.shape}"
            )
    elif array.ndim != 4:
        raise ValueError(f"{name} must be 4D (x,y,z,t), got {array.ndim}D")
    return array


@dataclass(frozen=True, slots=True)
class MPPCA:
    """Marchenko-Pastur PCA Denoising (MP-PCA).
    
    Reference:
        Veraart J, Fieremans E, Novikov DS. 
        Diffusion MRI noise mapping using random matrix theory. 
        Magn Reson Med. 2016;76(5):1582-1593.
        
    Matches qMRLab `denoising_mppca` / `MPdenoising.m`.
    """

    kernel: tuple[int, int, int] = (5, 5, 5)

    def fit(self, data: ArrayLike, mask: ArrayLike | None = None) -> dict[str, Any]:
        """Denoise 4D data.

        Args:
            data: 4D array (x, y, z, t).
            mask: Optional 3D mask (x, y, z).

        Returns:
            dict with:
                "denoised": 4D array
                "sigma": 3D noise map
                "n_pars": 3D map of number of signal components
        """
        data_arr = _as_4d_float_array(data, name="data")
        sx, sy, sz, m_dim = data_arr.shape
        
        if mask is None:
            mask_arr = np.ones((sx, sy, sz), dtype=bool)
        else:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != (sx, sy, sz):
                raise ValueError("mask shape must match spatial dims of data")

        # Sliding window implementation (equivalent to qMRLab 'full' sampling)
        denoised, sigma, n_pars = _mp_denoising(data_arr, mask_arr, self.kernel)
        
        return {
            "denoised": denoised,
            "sigma": sigma,
            "n_pars": n_pars,
        }

    def fit_image(self, data: ArrayLike, mask: ArrayLike | None = None) -> dict[str, Any]:
        return self.fit(data, mask=mask)


def _mp_denoising(
    data: NDArray[np.float64], 
    mask: NDArray[np.bool_], 
    kernel: tuple[int, int, int]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Core MP-PCA logic translating `MPdenoising.m`."""
    
    sx, sy, sz, m_dim = data.shape
    kx, ky, kz = kernel
    
    # Kernel half-width (qMRLab: k = (kernel-1)/2)
    # Ensure odd kernel
    if kx % 2 == 0 or ky % 2 == 0 or kz % 2 == 0:
        # qMRLab enforces odd: kernel = kernel + (mod(kernel, 2)-1)
        # We will just raise or adjust. Let's adjust for parity.
        kx = kx + (kx % 2 - 1)
        ky = ky + (ky % 2 - 1)
        kz = kz + (kz % 2 - 1)
        
    hkx, hky, hkz = (kx - 1) // 2, (ky - 1) // 2, (kz - 1) // 2
    n_pixels_patch = kx * ky * kz
    
    # Output arrays
    denoised = np.zeros_like(data)
    sigma_map = np.zeros((sx, sy, sz), dtype=np.float64)
    n_pars_map = np.zeros((sx, sy, sz), dtype=np.float64)
    
    # Precompute scaling for Eq 1
    # R = min(M, N)
    # scaling = (max(M, N) - (0:R-centering-1)) / N
    # centering = False in qMRLab defaults.
    r_dim = min(m_dim, n_pixels_patch)
    # MATLAB: 0:R-1 (inclusive) -> 0..R-1
    # scaling = (max(M, N) - np.arange(R)) / N
    scaling = (max(m_dim, n_pixels_patch) - np.arange(r_dim, dtype=np.float64)) / n_pixels_patch
    
    # Gamma for Eq 2
    gamma = (m_dim - np.arange(r_dim, dtype=np.float64)) / n_pixels_patch
    range_mp = 4.0 * np.sqrt(gamma)
    
    # Loop over pixels
    # qMRLab 'full' sampling: mask boundaries are ignored.
    # qMRLab: for i = k(3)+1:sz-k(3) ...
    # Python 0-indexed: range(hkz, sz - hkz)
    
    # Determine valid centers
    # We only process if mask[x,y,z] is True?
    # qMRLab code:
    # [x_, y_] = find(mask(:,:,i) == 1);
    # x = [x; x_]; ...
    # So it processes only where mask is 1.
    
    # Boundaries
    x_start, x_end = hkx, sx - hkx
    y_start, y_end = hky, sy - hky
    z_start, z_end = hkz, sz - hkz
    
    for z in range(z_start, z_end):
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                if not mask[x, y, z]:
                    continue
                
                # Extract patch
                # X = data(x-kx:x+kx, ...) 
                # MATLAB indices are 1-based, center x(nn).
                # Python indices: x is center. window [x-hkx : x+hkx+1]
                # Check bounds again just in case? Loop limits usually safe.
                
                patch = data[
                    x - hkx : x + hkx + 1,
                    y - hky : y + hky + 1,
                    z - hkz : z + hkz + 1,
                    :
                ]
                
                # Reshape to (N, M) -> (pixels, time)
                # qMRLab: X = reshape(X, N, M); X = X'; -> (M, N) ?
                # Wait. qMRLab says: [sx, sy, sz, M] = size(data).
                # X = reshape(.., N, M). X becomes (N, M) matrix.
                # X = X'. X becomes (M, N). Rows are timepoints, Cols are pixels.
                # svd(X, 'econ').
                # Python svd(X): X (M, N). U (M, K), S (K,), Vh (K, N).
                
                x_mat = patch.reshape(-1, m_dim).T # (M, N)
                
                # Center? qMRLab default centering=false.
                
                # SVD
                # numpy.linalg.svd returns S sorted descending.
                # qMRLab uses [u, vals, v] = svd(X, 'econ'); vals = diag(vals).^2 / N;
                # vals are eigenvalues of covariance matrix (1/N * X * X')?
                # Actually if X is (M, N), X*X' is (M, M).
                # If N > M, then full rank is M.
                # vals are squared singular values / N.
                
                try:
                    # full_matrices=False is 'econ'
                    u, s_vals, vh = np.linalg.svd(x_mat, full_matrices=False)
                except np.linalg.LinAlgError:
                    # Fallback for convergence failure
                    denoised[x, y, z, :] = data[x, y, z, :]
                    continue

                eig_vals = (s_vals ** 2) / n_pixels_patch
                
                # Veraart 2016 Logic
                # Eq 1: First estimation of Sigma^2
                # csum = cumsum(vals(R:-1:1)); cmean = ...
                # MATLAB R:-1:1 is reverse order (smallest first).
                
                # Python: eig_vals are descending (largest first).
                # We need smallest first for cumulative sum of noise tail.
                
                eig_vals_rev = eig_vals[::-1] # Ascending
                # We only consider up to R elements (in case M!=N)
                # SVD econ returns min(M, N) singular values.
                # So len(eig_vals) == R.
                
                # Cumulative mean of noise tail
                # csum[k] = sum(smallest k eigenvalues)
                csum = np.cumsum(eig_vals_rev)
                # Count: 1, 2, ..., R
                counts = np.arange(1, r_dim + 1, dtype=np.float64)
                cmean = csum / counts
                
                # sigmasq_1 = cmean ./ scaling (reversed scaling?)
                # qMRLab: scaling was created as (max(M,N) - (0:R-1))/N.
                # cmean reverses vals. scaling corresponds to which P?
                # Formula: sigma^2 = mean(\lambda) / (1 - P/N)? or similar.
                # qMRLab uses scaling vector. 
                # qMRLab: cmean is [mean(last 1), mean(last 2), ...].
                # scaling should correspond to [P=M-1, P=M-2 ...]?
                # qMRLab: scaling = (max(M, N) - (0:R-1)) / N.
                # scaling(1) corresponds to (M-0)/N?
                # Let's check qMRLab indices.
                # vals(R:-1:1) -> smallest first.
                # scaling(:) -> scaling is a vector 1..R.
                # sigmasq_1 = cmean ./ scaling; 
                # If cmean[0] is smallest eval (last component),
                # scaling[0] is (M)/N.
                # This seems to align with (M - const) / N.
                # In Veraart paper, <lambda> = sigma^2 * (1 ± sqrt(gamma))^2 ?
                # Or <lambda>_noise = sigma^2.
                # The scaling factor adjusts for the fact that we look at specific portion of spectrum?
                # I will trust qMRLab impl: `sigmasq_1 = cmean ./ scaling` strictly.
                # Note: `scaling` in Python created above was `(max(m_dim, n) - arange(r))`...
                # We need to reverse scaling to match `cmean` (which is reversed vals)?
                # qMRLab: `cmean = csum(R:-1:1) ...` -> No, csum is on reversed vals.
                # qMRLab: `cmean = csum ...`. `cmean` is [sum(v_R)/1, sum(v_R, v_R-1)/2 ...].
                # Wait. vals(R:-1:1) is smallest, 2nd smallest...
                # `scaling` is defined as `(max(M,N) - (0:R-1))/N`.
                # Is `scaling` reversed in qMRLab? No.
                # `sigmasq_1 = cmean ./ scaling`.
                # If this is elementwise division, then scaling[0] divides cmean[0].
                # scaling[0] is big (M/N).
                # cmean[0] is small (smallest eval).
                
                # However, python `scaling` above is generic. Let's align with qMRLab exactly.
                # qMRLab scaling: `(max(M, N) - (0:R-1)) / N`.
                # Python `scaling`: same.
                
                # In qMRLab:
                # csum = cumsum(vals(R:-1:1));
                # cmean = csum(R:-1:1) ... WAIT.
                # `cmean = csum(R:-1:1) ./ (R:-1:1)'`.
                # It reverses csum output back to interacting with largest components?
                # values: [L, ..., S].
                # vals(R:-1:1): [S, ..., L].
                # csum: [S, S+2ndS, ..., Total].
                # csum(R:-1:1): [Total, ..., S+2ndS, S]. 
                # (R:-1:1)': [R, ..., 1].
                # cmean: [Total/R, ..., (S+2ndS)/2, S/1].
                # This means cmean[0] corresponds to Mean of ALL (P=0).
                # cmean[R-1] corresponds to Mean of smallest (P=R-1).
                
                # So cmean indices align with `vals` (Largest...Smallest).
                # `scaling` indices 0..R-1 align with `vals` indices?
                # `scaling` (Eq 1 denominator?): (M - P)/N.
                # If index 0 (Largest), P=0? scaling=(M)/N.
                # If index R-1 (Smallest), P=R-1? scaling=(M-(R-1))/N.
                
                # So I need to replicate `cmean` logic.
                
                # Python:
                # eig_vals (descending).
                # csum_rev = cumsum(eig_vals[::-1]) # [S, S+2ndS, ...]
                # csum_orig_order = csum_rev[::-1]  # [Total, ..., S]
                # counts_rev = np.arange(1, r_dim + 1)
                # counts_orig_order = counts_rev[::-1] # [R, ..., 1]
                
                csum_rev = np.cumsum(eig_vals[::-1])
                csum_orig = csum_rev[::-1]
                counts_orig = np.arange(r_dim, 0, -1, dtype=np.float64)
                
                cmean = csum_orig / counts_orig
                sigmasq_1 = cmean / scaling
                
                # Eq 2: Second estimation (Range)
                # rangeData = vals(1:R) - vals(R); 
                # (Assuming centering=0)
                # qMRLab uses vals(1:R-centering). If R=min(M,N), and centering=0, it's all.
                
                range_data = eig_vals - eig_vals[-1]
                sigmasq_2 = range_data / range_mp
                
                # t = find(sigmasq_2 < sigmasq_1, 1);
                # Find first index where range-based estimate < mean-based estimate
                # indicating we entered the noise tail?
                
                # comparison
                mask_cond = sigmasq_2 < sigmasq_1
                # np.argmax on boolean returns first True index.
                if np.any(mask_cond):
                    t_idx = np.argmax(mask_cond)
                else:
                    t_idx = r_dim # None found
                
                if not np.any(mask_cond):
                    # No fit found? (e.g. extremely high SNR or low pixels)
                    # qMRLab sets sigma=NaN.
                    # We'll set sigma=0 and return raw.
                    sigma_map[x, y, z] = 0.0
                    n_pars_map[x, y, z] = r_dim
                    denoised[x, y, z, :] = data[x, y, z, :]
                else:
                    # t in qMRLab is 1-based index of where tail starts (noise components start).
                    # python t_idx is 0-based.
                    # eigenvalues[t_idx : ] are noise.
                    # signal components are 0 : t_idx.
                    
                    # sigma estimation from sigmasq_1[t]
                    # qMRLab: sigma(nn) = sqrt(sigmasq_1(t));
                    sigma_est = np.sqrt(sigmasq_1[t_idx])
                    sigma_map[x, y, z] = sigma_est
                    
                    # Denoise
                    # vals(t:R) = 0;
                    # s = u*diag(sqrt(N*vals))*v';
                    # We want to zero out noise components in reconstruction.
                    # so s_vals[t_idx:] = 0.
                    
                    s_vals_recon = s_vals.copy()
                    s_vals_recon[t_idx:] = 0.0
                    
                    # Reconstruct
                    # X_denoised = U * Sigma * Vh
                    # X_mat was (M, N).
                    # S is min(M, N).
                    
                    # np.dot(u * s, vh)
                    # u is (M, K), s is (K,), vh is (K, N).
                    
                    # s_vals_recon is length K=R.
                    
                    # s = u @ np.diag(s_vals_recon) @ vh
                    # To avoid full matrix mult, we can use slicing if needed, but R is small (30-60).
                    
                    x_denoised = u @ (s_vals_recon[:, None] * vh)
                    
                    # x_denoised is (M, N). Rows are timepoints.
                    # We need center pixel (N columns).
                    # qMRLab 'full' sampling takes center pixel of patch.
                    # Patch index range: 0..N-1.
                    # Center index: ceil(N/2) (1-based) -> (N-1)//2 (0-based)?
                    # qMRLab kernel: kx, ky, kz. Center is usually kx*ky*kz / 2 ?
                    # qMRLab: ceil(prod(kernel)/ 2).
                    # For 5x5x5=125. ceil(62.5)=63. 
                    # 1-based index 63.
                    # Python index 62.
                    
                    center_idx = (n_pixels_patch - 1) // 2
                    
                    # Extract column corresponding to center pixel
                    # x_denoised[:, center_idx] is (M,)
                    
                    denoised[x, y, z, :] = x_denoised[:, center_idx]
                    n_pars_map[x, y, z] = t_idx 

    return denoised, sigma_map, n_pars_map
