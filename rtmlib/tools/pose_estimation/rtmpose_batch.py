from typing import List, Sequence, Tuple, Union
import numpy as np
from ..base import BaseTool
from .post_processings import convert_coco_to_openpose, get_simcc_maximum
from .pre_processings import bbox_xyxy2cs, top_down_affine

class RTMPose(BaseTool):
    """
    Vectorised implementation that supports

        keypoints, scores = model(images, bboxes_list)

    Parameters
    ----------
    images : np.ndarray | Sequence[np.ndarray]
        • single image  → shape (H, W, 3) or list of length==1  
        • batch of N images (same HxW) → iterable/array of shape (N, H, W, 3)

    bboxes_list : list[list[list[float]]]
        `bboxes_list[i]` is the list of xyxy-boxes detected on `images[i]`.
        Pass `None`/`[]` to fall back to a full-frame box for each image.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple = (288, 384),
        mean: tuple = (123.675, 116.28, 103.53),
        std: tuple = (58.395, 57.12, 57.375),
        to_openpose: bool = False,
        backend: str = "onnxruntime",
        device: str = "cpu",
    ):
        super().__init__(onnx_model, model_input_size, mean, std, backend, device)
        self.to_openpose = to_openpose
        self.mean = np.asarray(mean, dtype=np.float32)[None, None, :]
        self.std = np.asarray(std, dtype=np.float32)[None, None, :]

    # ------------------------------------------------------------- public API
    def __call__(
        self,
        images: Union[np.ndarray, Sequence[np.ndarray]],
        bboxes_list: Sequence[Sequence[List[float]]] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        # ➊  normalise input types ------------------------------------------
        if isinstance(images, np.ndarray) and images.ndim == 3:
            # single frame                                                   │
            images = [images]                                                #│  N = 1
        images = list(images)                                                #│ ensure indexable
        N = len(images)
        if bboxes_list is None:
            bboxes_list = [[] for _ in range(N)]
        assert len(bboxes_list) == N, "`bboxes_list` length must equal number of images"
        # ➋  build the crop batch ------------------------------------------
        crops, centers, scales, owners = [], [], [], []            # owners = which image a crop came from
        for img_idx, (img, boxes) in enumerate(zip(images, bboxes_list)):
            if len(boxes) == 0:                                    # full-frame fallback
                boxes = [[0, 0, img.shape[1], img.shape[0]]]

            for b in boxes:                                        # (one affine per bbox)
                crop, c, s = self._preprocess_single(img, b)
                crops.append(crop)                                 # (h,w,3)
                centers.append(c)                                  # (2,)
                scales.append(s)                                   # (2,)
                owners.append(img_idx)

        crops   = np.stack(crops,   axis=0)                        # (M, H, W, 3)
        centers = np.asarray(centers, dtype=np.float32)            # (M, 2)
        scales  = np.asarray(scales,  dtype=np.float32)            # (M, 2)
        owners  = np.asarray(owners,  dtype=np.int32)              # (M,)

        simcc_x, simcc_y = self._infer_batch(crops)                # list of two tensors

        kpts, scores = self._postprocess_batch(
            (simcc_x, simcc_y), centers, scales
        )                                                          # (M,K,2) , (M,K)
        
        if self.to_openpose:
            kpts, scores = convert_coco_to_openpose(kpts, scores)

        # ➎  regroup results by original frame ----------------------------
        out_per_image = [[] for _ in range(N)]
        for i, owner in enumerate(owners):
            out_per_image[owner].append((kpts[i], scores[i]))

        # Concatenate detections *inside each* frame so that the caller gets
        # [frame0_kpts, frame1_kpts, …], but stay 1-D if the user passed only
        # one image (compatible with the original API).
        keypoints_final, scores_final = [], []
        for k_s_list in out_per_image:
            if len(k_s_list) == 0:           # should not happen
                print(f"Warning: no detections for image {len(keypoints_final)}")
                keypoints_final.append(np.empty((0, 17, 2), np.float32))
                scores_final.append(np.empty((0, 17),   np.float32))
            else:
                k, s = zip(*k_s_list)        # unzip
                keypoints_final.append(np.stack(k, axis=0))
                scores_final.append(np.stack(s, axis=0))

        if len(keypoints_final) == 1:
            # maintain the original return type for single-image input
            return keypoints_final[0], scores_final[0]
        else:
            return keypoints_final, scores_final

    # ---------------------------------------------------------------- utils
    # single-crop preprocessing (loops are allowed here)
    def _preprocess_single(self, img: np.ndarray, bbox_xyxy: list):
        bbox = np.asarray(bbox_xyxy, dtype=np.float32)
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)
        crop, scale = top_down_affine(self.model_input_size, scale, center, img)
        crop = crop.astype(np.float32)
        crop = (crop - self.mean) / self.std                        # colour-norm
        return crop, center, scale

    # batched ONNXRuntime execution
    def _infer_batch(self, imgs_hwc: np.ndarray) -> List[np.ndarray]:
        # imgs_hwc shape = (M, H, W, 3)  →  (M, 3, H, W)
        imgs_chw = imgs_hwc.transpose(0, 3, 1, 2).astype(np.float32)

        if self.backend != "onnxruntime":
            raise NotImplementedError(f"Backend {self.backend} is not supported.")
        input_name   = self.session.get_inputs()[0].name
        output_names = [o.name for o in self.session.get_outputs()]
        simcc_x, simcc_y = self.session.run(output_names, {input_name: imgs_chw})
        return simcc_x, simcc_y                                     # 2-item list

    # vectorised post-processing (no python loops)
    def _postprocess_batch(
        self,
        outputs: List[np.ndarray],        # [simcc_x, simcc_y]
        centers: np.ndarray,              # (M, 2)
        scales:  np.ndarray,              # (M, 2)
        simcc_split_ratio: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        simcc_x, simcc_y = outputs        # (M, K, Wx), (M, K, Wy)

        # ➊ peak search (already vectorised inside helper)
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)          # (M,K,2), (M,K)
        kpts = locs / simcc_split_ratio                             # SIMCC → heat-map coords

        # ➋ map back to input pixel space
        model_size = np.asarray(self.model_input_size, np.float32)  # (2,)
        kpts = (kpts / model_size) * scales[:, None, :]             # scale   (broadcast)
        kpts = kpts + centers[:, None, :] - scales[:, None, :] / 2  # shift
        return kpts, scores
