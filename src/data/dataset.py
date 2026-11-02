import json
import os
import glob
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Any
import csv

from . import preprocessing as preprocessing_utils
from .preprocessing import ImageIntensityStats

FLAIR_INDEX = 2

IMAGE_STATS_FILE_VERSION = 1


def patient_id_from_image_path(img_path: str) -> str:
    return os.path.basename(img_path).replace("_image.npy", "")


def n4_kwargs_from_dc(dc: dict) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if "n4_shrink_factor" in dc:
        kwargs["shrink_factor"] = int(dc["n4_shrink_factor"])
    if "n4_max_iterations" in dc:
        kwargs["max_iterations"] = tuple(dc["n4_max_iterations"])
    if "n4_convergence_threshold" in dc:
        kwargs["convergence_threshold"] = float(dc["n4_convergence_threshold"])
    return kwargs


def collect_raw_genomics_for_patients(data_dir: str, patient_ids: List[str]) -> np.ndarray:
    rows = []
    allowed = set(patient_ids)
    image_files = sorted(glob.glob(os.path.join(data_dir, "*_image.npy")))
    for img_path in image_files:
        pid = patient_id_from_image_path(img_path)
        if pid not in allowed:
            continue
        geno_path = img_path.replace("_image.npy", "_geno.npy")
        if os.path.exists(geno_path):
            rows.append(np.asarray(np.load(geno_path), dtype=np.float32).reshape(-1))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(rows)


def _read_split_csv(csv_path: str) -> Tuple[List[str], List[str], List[str]]:
    train_ids, val_ids, test_ids = [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid, split = row["patient_id"], row["split"]
            if split == "train":
                train_ids.append(pid)
            elif split == "val":
                val_ids.append(pid)
            elif split == "test":
                test_ids.append(pid)
    return train_ids, val_ids, test_ids


def split_patients_by_id(
    data_dir: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    save_csv_path: Optional[str] = None,
    require_existing: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    if require_existing and (save_csv_path is None or not os.path.exists(save_csv_path)):
        raise FileNotFoundError(
            f"[split_patients_by_id] configured split_csv_path='{save_csv_path}' does not "
            "exist. This path was provided explicitly to select which patient split to "
            "train on, so it must already exist — refusing to silently generate a new "
            "(different) split in its place. Create it first, or point split_csv_path at "
            "an existing split CSV."
        )
    if save_csv_path is not None and os.path.exists(save_csv_path):
        train_ids, val_ids, test_ids = _read_split_csv(save_csv_path)

        image_files = sorted(glob.glob(os.path.join(data_dir, "*_image.npy")))
        current_ids = {patient_id_from_image_path(p) for p in image_files}
        split_ids = set(train_ids) | set(val_ids) | set(test_ids)

        overlap = current_ids & split_ids
        if not overlap:
            raise ValueError(
                f"[split_patients_by_id] Existing split CSV '{save_csv_path}' shares ZERO "
                f"patients with data_dir '{data_dir}' (csv={len(split_ids)} patients, "
                f"data_dir={len(current_ids)} patients). This almost always means the CSV "
                "belongs to a different dataset/experiment than the one currently configured "
                "— refusing to silently build an empty train/val/test split from it. Point "
                "data.split_csv_path at the correct CSV, or delete/rename this one if you "
                "intend to generate a fresh split for this data_dir."
            )
        if current_ids != split_ids:
            warnings.warn(
                f"[split_patients_by_id] Existing split CSV '{save_csv_path}' does not exactly "
                f"match the patient set currently in '{data_dir}' "
                f"(csv={len(split_ids)} patients, data_dir={len(current_ids)} patients, "
                f"overlap={len(overlap)}). Reusing the existing split as-is (not regenerating) "
                "— delete the CSV explicitly if you intend to produce a new split.",
                stacklevel=2,
            )
        return train_ids, val_ids, test_ids

    image_files = sorted(glob.glob(os.path.join(data_dir, "*_image.npy")))
    patient_ids = sorted({patient_id_from_image_path(p) for p in image_files})

    rng = np.random.default_rng(seed)
    patient_ids = list(patient_ids)
    rng.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))
    n_test = n_total - n_train - n_val

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]

    if save_csv_path is not None:
        with open(save_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "split"])
            for pid in train_ids:
                writer.writerow([pid, "train"])
            for pid in val_ids:
                writer.writerow([pid, "val"])
            for pid in test_ids:
                writer.writerow([pid, "test"])

    return train_ids, val_ids, test_ids


def assert_disjoint_splits(train_ids, val_ids, test_ids) -> None:
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set
    assert not overlap_tv, f"[assert_disjoint_splits] train/val patient overlap: {overlap_tv}"
    assert not overlap_tt, f"[assert_disjoint_splits] train/test patient overlap: {overlap_tt}"
    assert not overlap_vt, f"[assert_disjoint_splits] val/test patient overlap: {overlap_vt}"


def _iter_train_flair_volumes(data_dir: str, train_patient_ids: List[str], flair_index: int):
    allowed = set(train_patient_ids)
    image_files = sorted(glob.glob(os.path.join(data_dir, "*_image.npy")))
    for img_path in image_files:
        pid = patient_id_from_image_path(img_path)
        if pid not in allowed:
            continue
        img_all = np.load(img_path, mmap_mode="r")
        for s in range(img_all.shape[0]):
            yield np.asarray(img_all[s, flair_index, :, :, :])


def _n4_cache_keys_for_train_patients(
    data_dir: str, train_patient_ids: List[str], flair_index: int, n4_kwargs: Optional[Dict[str, Any]] = None,
):
    allowed = set(train_patient_ids)
    image_files = sorted(glob.glob(os.path.join(data_dir, "*_image.npy")))
    for img_path in image_files:
        pid = patient_id_from_image_path(img_path)
        if pid not in allowed:
            continue
        img_all = np.load(img_path, mmap_mode="r")
        for s in range(img_all.shape[0]):
            yield preprocessing_utils.compute_n4_cache_key(pid, s, flair_index, img_path, n4_kwargs)


def fit_train_image_stats(
    data_dir: str,
    train_patient_ids: List[str],
    flair_index: int = FLAIR_INDEX,
    clip_pct: Tuple[float, float] = (0.5, 99.5),
    reservoir_size: int = 2_000_000,
    seed: int = 0,
    apply_n4: bool = False,
    n4_cache_dir: Optional[str] = None,
    n4_kwargs: Optional[Dict[str, Any]] = None,
    require_n4_cache: bool = True,
) -> Dict[int, ImageIntensityStats]:
    stats = preprocessing_utils.fit_image_intensity_stats(
        volume_iter_factory=lambda: _iter_train_flair_volumes(data_dir, train_patient_ids, flair_index),
        clip_pct=clip_pct,
        reservoir_size=reservoir_size,
        seed=seed,
        apply_n4=apply_n4,
        n4_cache_dir=n4_cache_dir,
        n4_kwargs=n4_kwargs,
        n4_cache_keys=(
            _n4_cache_keys_for_train_patients(data_dir, train_patient_ids, flair_index, n4_kwargs)
            if apply_n4 else None
        ),
        require_n4_cache=require_n4_cache,
    )
    return {flair_index: stats}


def save_image_stats(
    path: str,
    stats_by_channel: Dict[int, ImageIntensityStats],
    train_patient_ids: List[str],
    data_dir: str,
    apply_n4: bool = False,
) -> None:
    payload = {
        "version": IMAGE_STATS_FILE_VERSION,
        "data_dir": data_dir,
        "apply_n4": apply_n4,
        "train_patient_ids": sorted(train_patient_ids),
        "channels": {str(k): v.to_dict() for k, v in stats_by_channel.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp_path = path + f".tmp{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def load_image_stats(path: str) -> Tuple[Dict[int, ImageIntensityStats], Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[load_image_stats] '{path}' does not exist. Run scripts/fit_image_stats.py first — "
            "frozen image statistics are fit once, on disk, and never silently regenerated "
            "(same convention as split_patients_by_id's split_csv_path)."
        )
    with open(path) as f:
        payload = json.load(f)
    stats_by_channel = {
        int(k): ImageIntensityStats.from_dict(v) for k, v in payload.get("channels", {}).items()
    }
    if not stats_by_channel:
        raise ValueError(f"[load_image_stats] '{path}' contains no fitted channel statistics.")
    return stats_by_channel, payload


def collate_variable_length(batch):
    max_k = max(item["num_inputs"].item() for item in batch)
    B = len(batch)

    has_images = "input_images" in batch[0]
    has_latents = "input_latents" in batch[0]

    D, H, W = batch[0]["input_labels"].shape[1:]
    input_labels = torch.zeros(B, max_k, D, H, W)
    input_days = torch.zeros(B, max_k)
    input_treatments = torch.zeros(B, max_k, dtype=torch.long)
    input_mask = torch.zeros(B, max_k, dtype=torch.bool)

    if has_images:
        C = batch[0]["input_images"].shape[1]
        Di, Hi, Wi = batch[0]["input_images"].shape[2:]
        input_images = torch.zeros(B, max_k, C, Di, Hi, Wi)
        target_images = []

    if has_latents:
        Cz = batch[0]["input_latents"].shape[1]
        dl, hl, wl = batch[0]["input_latents"].shape[2:]
        input_latents = torch.zeros(B, max_k, Cz, dl, hl, wl)
        target_latents = []

    target_labels = []
    target_days = []
    target_treatments = []
    num_inputs = []

    has_geno = "geno" in batch[0]
    geno_list = []
    geno_mask_list = []

    has_tumor_tag = "is_tumor_positive_target" in batch[0]
    tumor_tag_list = []

    for i, item in enumerate(batch):
        k = item["num_inputs"].item()

        input_labels[i, :k] = item["input_labels"]
        input_days[i, :k] = item["input_days"]
        input_treatments[i, :k] = item["input_treatments"]
        input_mask[i, :k] = True

        if has_images:
            input_images[i, :k] = item["input_images"]
            target_images.append(item["target_image"])

        if has_latents:
            input_latents[i, :k] = item["input_latents"]
            target_latents.append(item["target_latent"])

        target_labels.append(item["target_label"])
        target_days.append(item["target_day"])
        target_treatments.append(item["target_treatment"])
        num_inputs.append(item["num_inputs"])

        if has_geno:
            geno_list.append(item["geno"])
            geno_mask_list.append(item["geno_mask"])

        if has_tumor_tag:
            tumor_tag_list.append(item["is_tumor_positive_target"])

    batch_out = {
        "input_visit_keys": [item.get("input_visit_keys") for item in batch],
        "input_labels": input_labels,
        "input_days": input_days,
        "input_treatments": input_treatments,
        "input_mask": input_mask,
        "target_label": torch.stack(target_labels),
        "target_day": torch.stack(target_days),
        "target_treatment": torch.stack(target_treatments),
        "num_inputs": torch.stack(num_inputs),
    }

    if has_images:
        batch_out["input_images"] = input_images
        batch_out["target_image"] = torch.stack(target_images)

    if has_latents:
        batch_out["input_latents"] = input_latents
        batch_out["target_latent"] = torch.stack(target_latents)

    if has_geno:
        batch_out["geno"] = torch.stack(geno_list)
        batch_out["geno_mask"] = torch.stack(geno_mask_list)

    if has_tumor_tag:
        batch_out["is_tumor_positive_target"] = torch.stack(tumor_tag_list)

    return batch_out


class PatientDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        dc: dict[str, Any],
        patient_ids: Optional[List[str]] = None,
        volume_size: Tuple[int, int, int] = (160, 240, 240),
        merge_labels: bool = True,
        geno_mean: Optional[np.ndarray] = None,
        geno_std: Optional[np.ndarray] = None,
        rng_seed: int = 0,
        mmap: bool = True,
        normalization_policy: Optional[str] = None,
        normalization_clip_pct: Optional[Tuple[float, float]] = None,
        image_stats: Optional[Dict[int, ImageIntensityStats]] = None,
        apply_n4: Optional[bool] = None,
        n4_cache_dir: Optional[str] = None,
        n4_kwargs: Optional[Dict[str, Any]] = None,
        context_sampling: Optional[str] = None,
        volume_cache_size: Optional[int] = None,
        normalized_volume_cache_dir: Optional[str] = None,
        latent_dir: Optional[str] = None,
        require_images: bool = True,
        representation: Optional[str] = None,
        latent_context_in_voxel: bool = False,
        voxel_shape: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.volume_size = tuple(volume_size)
        self.merge_labels = merge_labels
        self.geno_mean = geno_mean
        self.geno_std = geno_std
        self.rng = np.random.default_rng(rng_seed)
        self.mmap = mmap
        self.dc = dc

        self.latent_dir = latent_dir
        self.require_images = require_images
        if not require_images and latent_dir is None:
            raise ValueError(
                "PatientDataset(require_images=False) requires latent_dir to be set — "
                "there would be no image data of any kind to return."
            )

        self.representation = representation or dc.get("representation", "latent")
        if self.representation not in preprocessing_utils.REPRESENTATION_MODES:
            raise ValueError(
                f"Unknown data.representation={self.representation!r}, "
                f"expected one of {preprocessing_utils.REPRESENTATION_MODES}"
            )
        if self.representation == "voxel":
            self.model_shape = tuple(voxel_shape or dc.get("voxel_shape", (64, 96, 96)))
            if latent_dir is not None and latent_context_in_voxel:
                pass
            elif latent_dir is not None:
                warnings.warn(
                    "[PatientDataset] data.representation='voxel' but latent_dir is set — "
                    "the precomputed frozen-VAE latent cache is meaningless in voxel mode "
                    "(there is no VAE step) and will be ignored.",
                    stacklevel=2,
                )
                latent_dir = None
                self.latent_dir = None
        else:
            self.model_shape = self.volume_size

        self.normalization_policy = normalization_policy or dc.get(
            "normalization_policy", "whole_volume"
        )
        if self.normalization_policy not in preprocessing_utils.NORMALIZATION_POLICIES:
            raise ValueError(
                f"Unknown data.normalization_policy={self.normalization_policy!r}, "
                f"expected one of {preprocessing_utils.NORMALIZATION_POLICIES}"
            )
        self.normalization_clip_pct = tuple(
            normalization_clip_pct or dc.get("normalization_clip_pct", (0.5, 99.5))
        )
        self.volume_cache_size = int(
            volume_cache_size if volume_cache_size is not None else dc.get("volume_cache_size", 64)
        )
        self._volume_cache = preprocessing_utils.VolumeNormalizationCache(maxsize=self.volume_cache_size)
        self.normalized_volume_cache_dir = normalized_volume_cache_dir or dc.get("normalized_volume_cache_dir")

        self.image_stats = image_stats
        self.apply_n4 = bool(apply_n4) if apply_n4 is not None else bool(dc.get("apply_n4_bias_correction", False))
        self.n4_cache_dir = n4_cache_dir or dc.get("n4_cache_dir")
        self.n4_kwargs = n4_kwargs if n4_kwargs is not None else n4_kwargs_from_dc(dc)
        if self.apply_n4 and not self.n4_cache_dir:
            raise ValueError(
                "[PatientDataset] data.apply_n4_bias_correction=True requires data.n4_cache_dir — "
                "N4 is only ever computed by `scripts/precompute_n4.py` into a persistent cache; "
                "PatientDataset never computes it inline (see preprocessing.n4_correct_with_cache's "
                "require_cache=True)."
            )
        if self.normalization_policy == "frozen_train_stats":
            if not self.image_stats or FLAIR_INDEX not in self.image_stats or not self.image_stats[FLAIR_INDEX].is_fit:
                raise ValueError(
                    "[PatientDataset] normalization_policy='frozen_train_stats' requires a fitted "
                    "`image_stats={FLAIR_INDEX: ImageIntensityStats}` (see "
                    "dataset.fit_train_image_stats/scripts/fit_image_stats.py). This dataset never "
                    "fits/refits normalization statistics itself — pass the SAME frozen stats used "
                    "for every other split to avoid silently drifting from the train-fit transform."
                )

        self.context_sampling = context_sampling or dc.get("context_sampling", "all_windows")
        if self.context_sampling not in ("all_windows", "full_history_only"):
            raise ValueError(
                f"Unknown data.context_sampling={self.context_sampling!r}, "
                "expected 'all_windows' or 'full_history_only'"
            )

        self.patients = self._discover_patients(data_dir, patient_ids)

        self.index = self._build_index()

        tumor_pos = self.tumor_positive_fraction
        print(
            f"[Dataset3D-WholeVolume] Patients={len(self.patients)} | Samples={len(self.index)} "
            f"| representation={self.representation} | model_shape={self.model_shape} "
            f"| normalization_policy={self.normalization_policy} "
            f"| context_sampling={self.context_sampling} "
            f"| tumor_positive_target_fraction={tumor_pos:.3f} "
            f"| context_length_histogram={self.context_length_histogram}"
        )

    def _np_load(self, path: str):
        try:
            return np.load(path, mmap_mode="r") if self.mmap else np.load(path)
        except Exception as e:
            print("Failed loading:", path)
            raise e

    def _discover_patients(self, data_dir: str, patient_ids: Optional[List[str]]) -> List[Dict[str, str]]:
        allowed = set(patient_ids) if patient_ids is not None else None
        image_files = sorted(glob.glob(os.path.join(data_dir, "*_image.npy")))

        patients = []
        n_missing_latent = 0
        for img_path in image_files:
            pid = patient_id_from_image_path(img_path)
            if allowed is not None and pid not in allowed:
                continue

            prefix = img_path.replace("_image.npy", "")
            p = {
                "id": pid,
                "image": img_path,
                "label": prefix + "_label.npy",
                "days": prefix + "_days.npy",
                "treatment": prefix + "_treatment.npy",
                "geno": prefix + "_geno.npy",
            }
            required = ["label", "days", "treatment"]
            if self.latent_dir is not None:
                p["latent"] = os.path.join(self.latent_dir, f"{pid}_latent.npy")
                if not os.path.exists(p["latent"]):
                    n_missing_latent += 1
                    continue
                required.append("latent")
            if all(os.path.exists(p[k]) for k in required):
                patients.append(p)

        if self.latent_dir is not None and n_missing_latent > 0:
            warnings.warn(
                f"[PatientDataset] latent_dir='{self.latent_dir}' set but {n_missing_latent} "
                "patient(s) have no cached latent file — excluded from this split. Run "
                "scripts/precompute_latents.py to (re)build the cache.",
                stacklevel=2,
            )
        return patients

    @staticmethod
    def _validate_image_shape(arr: np.ndarray, patient_id: str) -> None:
        if arr.ndim != 5:
            raise ValueError(
                f"[PatientDataset] Patient '{patient_id}': expected image array of shape "
                f"(S, C, D, H, W) [S=sessions, C=modalities, D/H/W=volume dims], got "
                f"ndim={arr.ndim} with shape {arr.shape}. Whole-volume 3D processing requires "
                "a full 5D array — a 2D-slice-shaped array (e.g. (S, C, H, W)) is not supported."
            )

    @staticmethod
    def _validate_label_shape(arr: np.ndarray, patient_id: str) -> None:
        if arr.ndim != 4:
            raise ValueError(
                f"[PatientDataset] Patient '{patient_id}': expected label array of shape "
                f"(S, D, H, W), got ndim={arr.ndim} with shape {arr.shape}."
            )

    def _normalize_volume(self, img_all: np.ndarray, session_idx: int, patient_idx: int) -> np.ndarray:
        cache_key = (patient_idx, session_idx)
        pfiles = self.patients[patient_idx]
        patient_id = pfiles["id"]
        normalized_cache_key = (
            preprocessing_utils.compute_normalized_volume_cache_key(
                patient_id, session_idx, FLAIR_INDEX, pfiles["image"],
                self.normalization_policy, self.normalization_clip_pct,
                self.image_stats[FLAIR_INDEX] if self.image_stats else None,
                self.apply_n4, self.n4_kwargs,
            )
            if self.normalized_volume_cache_dir else None
        )

        def _load_volume():
            return np.asarray(img_all[session_idx, FLAIR_INDEX, :, :, :])

        if self.normalization_policy == "frozen_train_stats":
            n4_cache_key = (
                preprocessing_utils.compute_n4_cache_key(
                    patient_id, session_idx, FLAIR_INDEX, pfiles["image"], self.n4_kwargs
                )
                if self.apply_n4 else None
            )
            return preprocessing_utils.apply_normalization_policy(
                self.normalization_policy,
                raw_volume_fn=_load_volume,
                cache=self._volume_cache,
                cache_key=cache_key,
                stats=self.image_stats[FLAIR_INDEX],
                apply_n4=self.apply_n4,
                n4_cache_dir=self.n4_cache_dir,
                n4_cache_key=n4_cache_key,
                n4_kwargs=self.n4_kwargs,
                require_n4_cache=True,
                normalized_cache_dir=self.normalized_volume_cache_dir,
                normalized_cache_key=normalized_cache_key,
                require_normalized_cache=bool(self.normalized_volume_cache_dir),
            )

        return preprocessing_utils.apply_normalization_policy(
            self.normalization_policy,
            raw_volume_fn=_load_volume,
            cache=self._volume_cache,
            cache_key=cache_key,
            clip_pct=self.normalization_clip_pct,
            normalized_cache_dir=self.normalized_volume_cache_dir,
            normalized_cache_key=normalized_cache_key,
            require_normalized_cache=bool(self.normalized_volume_cache_dir),
        )

    def _crop_pad_3d(self, arr: np.ndarray) -> np.ndarray:
        return preprocessing_utils.crop_pad_volume_3d(arr, self.volume_size)

    def _to_model_shape(self, arr: np.ndarray, is_label: bool = False) -> np.ndarray:
        return preprocessing_utils.to_model_shape_3d(
            arr, self.model_shape, representation=self.representation, is_label=is_label
        )

    def _build_index(self) -> List[Tuple[int, int, int, bool]]:
        index = []
        context_length_counts: Dict[int, int] = {}
        tumor_positive_n = 0
        is_full_history_flags: List[bool] = []

        for p_idx, p in enumerate(self.patients):
            lbl = self._np_load(p["label"])
            self._validate_label_shape(lbl, p["id"])
            days = self._np_load(p["days"])
            treat = self._np_load(p["treatment"])

            S = int(lbl.shape[0])
            S_meta = min(len(days), len(treat), S)
            if S_meta < 2:
                continue

            if self.context_sampling == "full_history_only":
                windows = [(0, S_meta)]
            else:
                windows = [
                    (start, N)
                    for N in range(2, S_meta + 1)
                    for start in range(0, S_meta - N + 1)
                ]

            for start, N in windows:
                target_idx = start + N - 1
                k = N - 1
                is_full_history = (start == 0 and N == S_meta)
                is_tumor_positive = bool((lbl[target_idx] > 0).any())
                index.append((p_idx, start, N, is_tumor_positive))
                is_full_history_flags.append(is_full_history)
                context_length_counts[k] = context_length_counts.get(k, 0) + 1
                tumor_positive_n += int(is_tumor_positive)

        self.context_length_histogram = context_length_counts
        self.tumor_positive_fraction = (tumor_positive_n / len(index)) if index else float("nan")
        self.is_full_history_flags = is_full_history_flags
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient_idx, start, N, is_tumor_positive_target = self.index[idx]
        pfiles = self.patients[patient_idx]

        lbl_raw = self._np_load(pfiles["label"])
        days_raw = self._np_load(pfiles["days"])
        treat_raw = self._np_load(pfiles["treatment"])

        self._validate_label_shape(lbl_raw, pfiles["id"])

        geno = None
        real_geno = self.dc["real_geno"]
        geno_file_exists = os.path.exists(pfiles["geno"])
        if real_geno and geno_file_exists:
            genomics = self._np_load(pfiles["geno"])
            if self.geno_mean is not None and self.geno_std is not None:
                genomics = (genomics - self.geno_mean) / (self.geno_std + 1e-8)

            geno_mask = True

        else:
            genomics = np.zeros((self.dc["genomic_dim"],), dtype=np.float32)
            geno_mask = False

        S = int(lbl_raw.shape[0])

        S_meta = min(len(days_raw), len(treat_raw), S)
        days_arr = np.asarray(days_raw[:S_meta], dtype=np.float32)
        treat_arr = np.asarray(treat_raw[:S_meta], dtype=np.int64)

        window_sessions = list(range(start, start + N))
        input_indices = window_sessions[:-1]
        target_idx = window_sessions[-1]
        k = len(input_indices)

        sample: Dict[str, Any] = {}

        if self.require_images:
            img_raw = self._np_load(pfiles["image"])
            self._validate_image_shape(img_raw, pfiles["id"])

            input_images = []
            for i in input_indices:
                vol = self._normalize_volume(img_raw, i, patient_idx)
                vol = self._to_model_shape(vol, is_label=False)[None, ...]
                input_images.append(vol)

            target_vol = self._normalize_volume(img_raw, target_idx, patient_idx)
            target_vol = self._to_model_shape(target_vol, is_label=False)[None, ...]

            sample["input_images"] = torch.from_numpy(np.stack(input_images)).float()
            sample["target_image"] = torch.from_numpy(target_vol).float()

        if self.latent_dir is not None:
            latent_raw = self._np_load(pfiles["latent"])
            input_latents = [np.asarray(latent_raw[i]) for i in input_indices]
            target_latent = np.array(latent_raw[target_idx])

            sample["input_latents"] = torch.from_numpy(np.stack(input_latents)).float()
            sample["target_latent"] = torch.from_numpy(target_latent).float()

        input_labels = []
        for i in input_indices:
            seg = np.asarray(lbl_raw[i])
            if self.merge_labels:
                seg = (seg > 0).astype(np.float32)
            input_labels.append(self._to_model_shape(seg, is_label=True))

        target_seg = np.asarray(lbl_raw[target_idx])
        if self.merge_labels:
            target_seg = (target_seg > 0).astype(np.float32)
        target_seg = self._to_model_shape(target_seg, is_label=True)

        target_day_abs = days_arr[target_idx]
        input_days_rel = np.array([days_arr[i] - target_day_abs for i in input_indices], dtype=np.float32)
        last_input_day = days_arr[input_indices[-1]]
        target_day_rel = np.float32(target_day_abs - last_input_day)

        input_treats = np.array([treat_arr[i] for i in input_indices], dtype=np.int64)
        target_treat = np.int64(treat_arr[target_idx])

        sample.update({
            "input_visit_keys": [f"{pfiles['id']}_s{i}" for i in input_indices],
            "input_labels": torch.from_numpy(np.stack(input_labels)).float(),
            "input_days": torch.from_numpy(input_days_rel).float(),
            "input_treatments": torch.from_numpy(input_treats).long(),
            "target_label": torch.from_numpy(target_seg).float(),
            "target_day": torch.tensor(target_day_rel).float(),
            "target_treatment": torch.tensor(target_treat).long(),
            "num_inputs": torch.tensor(k).long(),
            "patient_idx": int(patient_idx),
            "window_N": int(N),
            "window_start": int(start),
            "is_tumor_positive_target": torch.tensor(is_tumor_positive_target, dtype=torch.bool),
            "geno_mask": torch.tensor(geno_mask, dtype=torch.bool),
        })
        geno = np.asarray(genomics, dtype=np.float32).reshape(-1)
        sample["geno"] = torch.from_numpy(geno.copy())

        return sample


class PatientLocalityBatchSampler(torch.utils.data.Sampler[List[int]]):

    def __init__(self, dataset: "PatientDataset", group_size: int, drop_last: bool = True, seed: int = 0):
        self.dataset = dataset
        self.group_size = max(1, int(group_size))
        self.drop_last = drop_last
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.dataset.index)
        return n // self.group_size if self.drop_last else -(-n // self.group_size)

    def _epoch_order(self) -> List[int]:
        self._epoch += 1
        rng = np.random.default_rng(self.seed + self._epoch)

        by_patient: Dict[int, List[int]] = {}
        for sample_idx, entry in enumerate(self.dataset.index):
            by_patient.setdefault(entry[0], []).append(sample_idx)

        patient_ids = list(by_patient.keys())
        rng.shuffle(patient_ids)
        for pid in patient_ids:
            rng.shuffle(by_patient[pid])

        order: List[int] = []
        for chunk_start in range(0, len(patient_ids), self.group_size):
            chunk_lists = [by_patient[pid] for pid in patient_ids[chunk_start: chunk_start + self.group_size]]
            max_len = max(len(lst) for lst in chunk_lists)
            for pos in range(max_len):
                for lst in chunk_lists:
                    if pos < len(lst):
                        order.append(lst[pos])
        return order

    def __iter__(self):
        order = self._epoch_order()
        stop = len(order) - (len(order) % self.group_size) if self.drop_last else len(order)
        for i in range(0, stop, self.group_size):
            yield order[i:i + self.group_size]
