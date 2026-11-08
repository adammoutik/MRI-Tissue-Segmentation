"""
Core module for Project 4: Brain MRI Tissue Segmentation via Split & Merge.

Implements:
1. Synthetic Brain MRI Phantom (256x256) with exact ground-truth labels (Background, CSF, Gray Matter, White Matter, Lesion) and optional Rician noise + Intensity Non-Uniformity (INU / bias field).
2. Real Public Brain MRI Loader (3D volume from scikit-image `data.brain()`, 10x256x256).
3. Quadtree Split Phase with configurable homogeneity predicates (std, variance, range, hybrid) and minimum block size.
4. Region Adjacency Graph (RAG) Merge Phase with hierarchical agglomerative merging based on mean difference and statistical compatibility.
5. Quantitative Evaluation Metrics: Dice Similarity Coefficient (DSC), Intersection over Union (IoU), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM), and region counts.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage import data, filters, metrics


@dataclass
class QuadNode:
    """Node in the Quadtree decomposition of an image."""
    x: int
    y: int
    w: int
    h: int
    depth: int
    mean: float
    std: float
    min_val: float
    max_val: float
    leaf_id: int = -1
    children: List["QuadNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def area(self) -> int:
        return self.w * self.h


def generate_synthetic_brain_mri(
    size: int = 256,
    noise_sigma: float = 0.03,
    bias_strength: float = 0.10,
    include_lesion: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a realistic 2D axial T1-weighted Brain MRI phantom with known ground-truth tissue masks.

    Tissues (labels):
        0: Background (air / skull exterior) -> base intensity 0.02
        1: CSF (Cerebrospinal Fluid / Ventricles & Sulci) -> base intensity 0.22
        2: Gray Matter (Cortical ribbon & deep nuclei) -> base intensity 0.52
        3: White Matter (Parenchyma) -> base intensity 0.78
        4: Hyperintense Lesion / Tumor (optional) -> base intensity 0.94

     Note on Problematic ("Certaines régions sont contrastées, d'autres beaucoup moins"):
        - High contrast boundary: Skull/Background (0.02) vs Tissue (>0.50), Lesion (0.94) vs GM (0.52).
        - Low contrast boundary: Deep Gray Matter (caudate/thalamus ~0.52) vs White Matter (0.78) under
          RF bias field and Rician noise, and cortical GM/WM transitions.

    Returns:
        mri_noisy: float64 image in [0, 1] with Rician noise and RF bias field
        mri_clean: float64 noise-free image in [0, 1]
        gt_labels: int32 ground-truth tissue map (0=BG, 1=CSF, 2=GM, 3=WM, 4=Lesion)
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]

    # Polar coordinates for cortical folding modulation
    r_ell = np.sqrt((xx / 0.72) ** 2 + (yy / 0.86) ** 2)
    theta = np.arctan2(yy, xx)

    # Outer brain boundary with subtle gyral undulations
    gyri_mod = (
        0.025 * np.cos(8 * theta)
        + 0.018 * np.sin(14 * theta + 0.6)
        + 0.012 * np.cos(20 * theta - 0.4)
    )
    brain_mask = r_ell <= (0.92 + gyri_mod)
    subarachnoid_mask = (r_ell <= 0.96) & (~brain_mask)

    # Cortical Gray Matter ribbon (outer shell of parenchyma)
    wm_boundary = 0.77 + 0.045 * np.cos(10 * theta + 0.2) + 0.03 * np.sin(16 * theta - 0.5)
    wm_mask = r_ell <= wm_boundary
    cortical_gm_mask = brain_mask & (~wm_mask)

    # Longitudinal fissure (interhemispheric CSF cleft)
    fissure = (np.abs(xx) < 0.022) & (np.abs(yy) < 0.80)

    # Lateral ventricles (butterfly-shaped central CSF cavities)
    vent_left = (((xx + 0.13) / 0.095) ** 2 + ((yy + 0.03) / 0.26) ** 2 <= 1.0) & (
        ((xx + 0.20) / 0.09) ** 2 + ((yy + 0.03) / 0.22) ** 2 >= 0.35
    )
    vent_right = (((xx - 0.13) / 0.095) ** 2 + ((yy + 0.03) / 0.26) ** 2 <= 1.0) & (
        ((xx - 0.20) / 0.09) ** 2 + ((yy + 0.03) / 0.22) ** 2 >= 0.35
    )
    ant_horn_l = ((xx + 0.09) / 0.07) ** 2 + ((yy + 0.22) / 0.09) ** 2 <= 1.0
    ant_horn_r = ((xx - 0.09) / 0.07) ** 2 + ((yy + 0.22) / 0.09) ** 2 <= 1.0
    post_horn_l = ((xx + 0.11) / 0.06) ** 2 + ((yy - 0.24) / 0.09) ** 2 <= 1.0
    post_horn_r = ((xx - 0.11) / 0.06) ** 2 + ((yy - 0.24) / 0.09) ** 2 <= 1.0
    ventricles = vent_left | vent_right | ant_horn_l | ant_horn_r | post_horn_l | post_horn_r

    # Deep Gray Matter nuclei (Thalamus + Caudate/Putamen - low contrast against WM!)
    thalamus_l = ((xx + 0.14) / 0.09) ** 2 + ((yy - 0.06) / 0.12) ** 2 <= 1.0
    thalamus_r = ((xx - 0.14) / 0.09) ** 2 + ((yy - 0.06) / 0.12) ** 2 <= 1.0
    putamen_l = ((xx + 0.29) / 0.065) ** 2 + ((yy + 0.06) / 0.14) ** 2 <= 1.0
    putamen_r = ((xx - 0.29) / 0.065) ** 2 + ((yy + 0.06) / 0.14) ** 2 <= 1.0
    deep_gm = (thalamus_l | thalamus_r | putamen_l | putamen_r) & (~ventricles)

    # Sulcal CSF indentations
    sulci = (
        (r_ell > 0.70)
        & (r_ell <= 0.92 + gyri_mod)
        & (np.cos(12 * theta + 0.3) > 0.82)
    )

    # Assemble Ground Truth Labels
    gt_labels = np.zeros((size, size), dtype=np.int32)
    gt_labels[subarachnoid_mask] = 1  # CSF
    gt_labels[cortical_gm_mask] = 2   # GM
    gt_labels[wm_mask] = 3            # WM
    gt_labels[deep_gm] = 2            # Deep GM
    gt_labels[sulci | fissure | ventricles] = 1  # CSF

    if include_lesion:
        # Focal hyperintense lesion in right parietal white/gray matter
        lesion_mask = ((xx - 0.33) / 0.11) ** 2 + ((yy - 0.22) / 0.095) ** 2 <= 1.0
        gt_labels[lesion_mask] = 4

    # Assign clean tissue intensities
    intensity_map = {
        0: 0.03,  # Background
        1: 0.22,  # CSF
        2: 0.53,  # Gray Matter
        3: 0.76,  # White Matter
        4: 0.94,  # Hyperintense lesion
    }
    mri_clean = np.zeros((size, size), dtype=np.float64)
    for lbl, val in intensity_map.items():
        mri_clean[gt_labels == lbl] = val

    # Slight partial-volume smoothing at anatomical boundaries (sigma=0.6 px)
    mri_clean = ndimage.gaussian_filter(mri_clean, sigma=0.6)

    # Multiplicative RF Bias Field (Intensity Non-Uniformity - INU)
    bias_field = 1.0 + bias_strength * (0.6 * (xx ** 2 - yy ** 2) + 0.4 * xx * yy + 0.3 * xx)
    mri_biased = mri_clean * bias_field

    # Rician Noise (magnitude of complex Gaussian noise in MRI k-space reconstruction)
    if noise_sigma > 0:
        n_real = rng.normal(0.0, noise_sigma, size=(size, size))
        n_imag = rng.normal(0.0, noise_sigma, size=(size, size))
        mri_noisy = np.sqrt((mri_biased + n_real) ** 2 + n_imag ** 2)
    else:
        mri_noisy = mri_biased.copy()

    mri_noisy = np.clip(mri_noisy, 0.0, 1.0)
    mri_clean = np.clip(mri_clean, 0.0, 1.0)
    return mri_noisy, mri_clean, gt_labels


def load_real_brain_mri() -> np.ndarray:
    """
    Load the 3D public Brain MRI dataset from scikit-image (`data.brain()`),
    returning a normalized float64 array of shape (10, 256, 256) in [0, 1].
    """
    vol = data.brain().astype(np.float64)
    p1, p99 = np.percentile(vol, (0.5, 99.5))
    vol_norm = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    return vol_norm


def generate_pseudo_gt_for_real_mri(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a reference 4-class tissue segmentation (0: Background, 1: CSF/Dark tissue,
    2: Gray Matter, 3: White Matter) on a real 2D brain MRI slice using skull masking,
    bilateral denoising, and Multi-Otsu thresholding for objective benchmark comparison.
    """
    smoothed = cv2.bilateralFilter(image.astype(np.float32), d=7, sigmaColor=0.08, sigmaSpace=5).astype(np.float64)
    fg_mask = smoothed > 0.06
    fg_mask = ndimage.binary_fill_holes(fg_mask)
    fg_mask = ndimage.binary_opening(fg_mask, structure=np.ones((3, 3)))

    fg_pixels = smoothed[fg_mask]
    thresholds = filters.threshold_multiotsu(fg_pixels, classes=3)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[fg_mask & (smoothed <= thresholds[0])] = 1  # CSF / Sulci
    labels[fg_mask & (smoothed > thresholds[0]) & (smoothed <= thresholds[1])] = 2  # Gray Matter
    labels[fg_mask & (smoothed > thresholds[1])] = 3  # White Matter
    return labels, fg_mask


class SplitAndMergeSegmenter:
    """
    Split & Merge Image Segmentation using a Quadtree decomposition followed by
    Region Adjacency Graph (RAG) hierarchical merging.

    Parameters:
        split_threshold (float): Homogeneity threshold tau_split.
        merge_threshold (float): Similarity threshold tau_merge for adjacent regions.
        min_block_size (int): Minimum quadtree block dimension (power of 2, e.g., 2, 4, 8).
        predicate (str): Homogeneity criterion:
            - 'std': standard deviation <= split_threshold
            - 'range': (max - min) <= split_threshold
            - 'hybrid': std <= split_threshold AND (max - min) <= 2.8 * split_threshold
        adaptive_contrast (bool): Weight homogeneity by local contrast to preserve low-contrast tissues.
    """

    def __init__(
        self,
        split_threshold: float = 0.045,
        merge_threshold: float = 0.055,
        min_block_size: int = 4,
        predicate: str = "hybrid",
        adaptive_contrast: bool = True,
    ):
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        self.min_block_size = min_block_size
        self.predicate = predicate
        self.adaptive_contrast = adaptive_contrast

        # Stored state after fit_predict
        self.root: Optional[QuadNode] = None
        self.leaves: List[QuadNode] = []
        self.split_label_map: Optional[np.ndarray] = None
        self.split_reconstructed: Optional[np.ndarray] = None
        self.merged_label_map: Optional[np.ndarray] = None
        self.merged_reconstructed: Optional[np.ndarray] = None
        self.num_leaves: int = 0
        self.num_merged_regions: int = 0
        self.split_time_ms: float = 0.0
        self.merge_time_ms: float = 0.0

    def _is_homogeneous(self, patch: np.ndarray, w: int, h: int) -> bool:
        if w <= self.min_block_size or h <= self.min_block_size:
            return True

        std_val = float(np.std(patch))
        ptp_val = float(np.ptp(patch))
        mean_val = float(np.mean(patch))

        # Effective threshold can adapt slightly in mid-intensity brain parenchyma
        # where GM/WM contrast is lower than skull/background boundaries
        tau = self.split_threshold
        if self.adaptive_contrast and 0.18 < mean_val < 0.85:
            tau = self.split_threshold * 0.82

        if self.predicate == "std":
            return std_val <= tau
        elif self.predicate == "range":
            return ptp_val <= tau * 2.5
        elif self.predicate == "hybrid":
            return (std_val <= tau) and (ptp_val <= tau * 3.0)
        else:
            raise ValueError(f"Unknown predicate: {self.predicate}")

    def _build_quadtree(self, img: np.ndarray, x: int, y: int, w: int, h: int, depth: int) -> QuadNode:
        patch = img[y : y + h, x : x + w]
        node = QuadNode(
            x=x,
            y=y,
            w=w,
            h=h,
            depth=depth,
            mean=float(np.mean(patch)),
            std=float(np.std(patch)),
            min_val=float(np.min(patch)),
            max_val=float(np.max(patch)),
        )

        if not self._is_homogeneous(patch, w, h) and w > self.min_block_size and h > self.min_block_size:
            hw, hh = w // 2, h // 2
            node.children = [
                self._build_quadtree(img, x, y, hw, hh, depth + 1),
                self._build_quadtree(img, x + hw, y, w - hw, hh, depth + 1),
                self._build_quadtree(img, x, y + hh, hw, h - hh, depth + 1),
                self._build_quadtree(img, x + hw, y + hh, w - hw, h - hh, depth + 1),
            ]
        else:
            node.leaf_id = len(self.leaves)
            self.leaves.append(node)

        return node

    def _extract_rag_adjacency(self, label_map: np.ndarray) -> Dict[int, Set[int]]:
        """Build 4-connected Region Adjacency Graph (RAG) from leaf label map."""
        adj: Dict[int, Set[int]] = {i: set() for i in range(len(self.leaves))}

        # Horizontal adjacency
        diff_h = label_map[:, :-1] != label_map[:, 1:]
        u_h = label_map[:, :-1][diff_h]
        v_h = label_map[:, 1:][diff_h]
        for u, v in zip(u_h, v_h):
            adj[int(u)].add(int(v))
            adj[int(v)].add(int(u))

        # Vertical adjacency
        diff_v = label_map[:-1, :] != label_map[1:, :]
        u_v = label_map[:-1, :][diff_v]
        v_v = label_map[1:, :][diff_v]
        for u, v in zip(u_v, v_v):
            adj[int(u)].add(int(v))
            adj[int(v)].add(int(u))

        return adj

    def fit_predict(self, image: np.ndarray, n_tissue_classes: Optional[int] = 4) -> Dict[str, np.ndarray]:
        """
        Execute Split -> Merge segmentation on a 2D grayscale image in [0, 1].
        """
        H, W = image.shape
        self.leaves = []

        # --- PHASE 1: QUADTREE SPLIT ---
        t0 = time.perf_counter()
        self.root = self._build_quadtree(image, 0, 0, W, H, depth=0)
        self.num_leaves = len(self.leaves)

        split_labels = np.zeros((H, W), dtype=np.int32)
        split_recon = np.zeros((H, W), dtype=np.float64)
        for leaf in self.leaves:
            split_labels[leaf.y : leaf.y + leaf.h, leaf.x : leaf.x + leaf.w] = leaf.leaf_id
            split_recon[leaf.y : leaf.y + leaf.h, leaf.x : leaf.x + leaf.w] = leaf.mean

        self.split_label_map = split_labels
        self.split_reconstructed = split_recon
        self.split_time_ms = (time.perf_counter() - t0) * 1000.0

        # --- PHASE 2: REGION ADJACENCY GRAPH (RAG) MERGE ---
        t1 = time.perf_counter()
        adj = self._extract_rag_adjacency(split_labels)

        # Union-Find with region statistics tracking (sum of intensities, sum of squares, pixel count)
        parent = list(range(self.num_leaves))
        reg_area = np.array([leaf.area for leaf in self.leaves], dtype=np.float64)
        reg_mean = np.array([leaf.mean for leaf in self.leaves], dtype=np.float64)
        reg_var = np.array([leaf.std ** 2 for leaf in self.leaves], dtype=np.float64)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        # Iterative region merging: sort candidate neighbor pairs by mean similarity
        # and merge while combined region satisfies merge homogeneity predicate
        active_edges = []
        for u in range(self.num_leaves):
            for v in adj[u]:
                if u < v:
                    active_edges.append((abs(reg_mean[u] - reg_mean[v]), u, v))
        active_edges.sort(key=lambda x: x[0])

        # Multi-pass merge until no adjacent regions satisfy merge threshold
        changed = True
        pass_num = 0
        while changed and pass_num < 6:
            changed = False
            pass_num += 1
            for _, u, v in active_edges:
                ru, rv = find(u), find(v)
                if ru == rv:
                    continue
                mean_diff = abs(reg_mean[ru] - reg_mean[rv])
                if mean_diff <= self.merge_threshold:
                    # Compute combined variance via parallel axis theorem
                    new_area = reg_area[ru] + reg_area[rv]
                    new_mean = (reg_area[ru] * reg_mean[ru] + reg_area[rv] * reg_mean[rv]) / new_area
                    new_var = (
                        reg_area[ru] * (reg_var[ru] + (reg_mean[ru] - new_mean) ** 2)
                        + reg_area[rv] * (reg_var[rv] + (reg_mean[rv] - new_mean) ** 2)
                    ) / new_area

                    if np.sqrt(new_var) <= self.merge_threshold * 1.15:
                        # Merge rv into ru
                        if reg_area[ru] < reg_area[rv]:
                            ru, rv = rv, ru
                        parent[rv] = ru
                        reg_area[ru] = new_area
                        reg_mean[ru] = new_mean
                        reg_var[ru] = new_var
                        changed = True

            if changed:
                # Rebuild edges between current super-regions
                new_edges_set = set()
                for _, u, v in active_edges:
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        a, b = (ru, rv) if ru < rv else (rv, ru)
                        new_edges_set.add((a, b))
                active_edges = [(abs(reg_mean[a] - reg_mean[b]), a, b) for (a, b) in new_edges_set]
                active_edges.sort(key=lambda x: x[0])

        # Map leaf IDs to contiguous merged region IDs
        root_map: Dict[int, int] = {}
        leaf_to_merged = np.zeros(self.num_leaves, dtype=np.int32)
        leaf_to_mean = np.zeros(self.num_leaves, dtype=np.float64)

        for i in range(self.num_leaves):
            r = find(i)
            if r not in root_map:
                root_map[r] = len(root_map)
            leaf_to_merged[i] = root_map[r]
            leaf_to_mean[i] = reg_mean[r]

        self.num_merged_regions = len(root_map)
        self.merged_label_map = leaf_to_merged[split_labels]
        self.merged_reconstructed = leaf_to_mean[split_labels]
        self.merge_time_ms = (time.perf_counter() - t1) * 1000.0

        # Optional: Quantize merged region means into anatomical tissue classes (BG, CSF, GM, WM, [Lesion])
        tissue_seg = np.zeros_like(self.merged_label_map)
        if n_tissue_classes is not None and n_tissue_classes >= 2:
            tissue_seg = self.classify_tissues(self.merged_reconstructed, n_classes=n_tissue_classes)

        return {
            "split_labels": self.split_label_map,
            "split_reconstructed": self.split_reconstructed,
            "merged_labels": self.merged_label_map,
            "merged_reconstructed": self.merged_reconstructed,
            "tissue_segmentation": tissue_seg,
        }

    @staticmethod
    def classify_tissues(reconstructed_img: np.ndarray, n_classes: int = 4) -> np.ndarray:
        """
        Group merged region mean intensities into anatomical tissue classes using
        1D K-Means / Nearest-Prototype clustering initialized on anatomical intensity priors.
        For n_classes == 5 (Synthetic Brain Phantom with Lesion):
            0: Background (~0.03), 1: CSF (~0.22), 2: Gray Matter (~0.53),
            3: White Matter (~0.76), 4: Hyperintense Lesion (~0.94)
        For n_classes == 4 (Standard Brain MRI):
            0: Background (< 0.08), 1: CSF, 2: Gray Matter, 3: White Matter
        """
        labels = np.zeros(reconstructed_img.shape, dtype=np.int32)
        fg_mask = reconstructed_img > 0.09
        fg_vals = reconstructed_img[fg_mask]
        if len(fg_vals) == 0:
            return labels

        if n_classes == 5:
            # Initialize 4 foreground prototypes (CSF, GM, WM, Lesion) and refine via 1D K-Means (Lloyd's updates)
            centers = np.array([0.22, 0.53, 0.76, 0.94], dtype=np.float64)
            for _ in range(10):
                dists = np.abs(fg_vals[:, None] - centers[None, :])
                assign = np.argmin(dists, axis=1)
                for k in range(4):
                    mask_k = assign == k
                    if np.any(mask_k):
                        # Regularized update toward data mean while preserving anatomical ordering
                        centers[k] = 0.7 * float(np.mean(fg_vals[mask_k])) + 0.3 * centers[k]
                centers.sort()
            dists = np.abs(fg_vals[:, None] - centers[None, :])
            fg_labels = np.argmin(dists, axis=1) + 1
            labels[fg_mask] = fg_labels
        else:
            unique_vals = np.unique(fg_vals)
            if len(unique_vals) < n_classes - 1:
                thresholds = np.quantile(fg_vals, np.linspace(0, 1, n_classes)[1:-1])
            else:
                thresholds = filters.threshold_multiotsu(fg_vals, classes=n_classes - 1)
            labels[fg_mask] = 1
            for idx, th in enumerate(thresholds, start=2):
                labels[fg_mask & (reconstructed_img > th)] = idx

        return labels

    def draw_quadtree_grid(self, image: np.ndarray, color: Tuple[float, float, float] = (0.0, 0.85, 1.0)) -> np.ndarray:
        """Overlay the Quadtree leaf boundaries on top of the grayscale image."""
        rgb = np.stack([image, image, image], axis=-1).copy()
        H, W = image.shape
        for leaf in self.leaves:
            y0, y1 = leaf.y, min(leaf.y + leaf.h - 1, H - 1)
            x0, x1 = leaf.x, min(leaf.x + leaf.w - 1, W - 1)
            rgb[y0, x0:x1+1] = color
            rgb[y1, x0:x1+1] = color
            rgb[y0:y1+1, x0] = color
            rgb[y0:y1+1, x1] = color
        return rgb


def compute_segmentation_metrics(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    reconstructed: np.ndarray,
    clean_img: np.ndarray,
    class_names: Dict[int, str],
) -> Dict[str, float]:
    """
    Compute Dice Similarity Coefficient (DSC), IoU (Jaccard) per tissue class,
    macro-average Dice/IoU over foreground tissues, PSNR, and SSIM.
    """
    results: Dict[str, float] = {}
    dice_list = []
    iou_list = []

    for cid, cname in class_names.items():
        pred_mask = pred_labels == cid
        gt_mask = gt_labels == cid
        inter = float(np.logical_and(pred_mask, gt_mask).sum())
        pred_sum = float(pred_mask.sum())
        gt_sum = float(gt_mask.sum())
        union = float(np.logical_or(pred_mask, gt_mask).sum())

        dice = (2.0 * inter) / (pred_sum + gt_sum + 1e-8) if (pred_sum + gt_sum) > 0 else 1.0
        iou = inter / (union + 1e-8) if union > 0 else 1.0

        results[f"Dice_{cname}"] = dice
        results[f"IoU_{cname}"] = iou
        if cid > 0:  # Foreground brain tissues
            dice_list.append(dice)
            iou_list.append(iou)

    results["Macro_Dice"] = float(np.mean(dice_list))
    results["Macro_IoU"] = float(np.mean(iou_list))
    results["PSNR"] = float(metrics.peak_signal_noise_ratio(clean_img, reconstructed, data_range=1.0))
    results["SSIM"] = float(metrics.structural_similarity(clean_img, reconstructed, data_range=1.0))
    return results
