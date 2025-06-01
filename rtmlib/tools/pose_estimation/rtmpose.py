from typing import List, Tuple

import numpy as np

from ..base import BaseTool
from .post_processings import convert_coco_to_openpose, get_simcc_maximum
from .pre_processings import bbox_xyxy2cs, top_down_affine
from pytictoc import TicToc

t = TicToc() #create instance of class

PROC_DEBUG = False
MAIN_DEBUG = False
INFER_DEBUG = False

class RTMPose(BaseTool):

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (288, 384),
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        super().__init__(onnx_model, model_input_size, mean, std, backend,
                         device)
        self.to_openpose = to_openpose

    def __call__(self, image: np.ndarray, bboxes: list = [], use_batch: bool = True):

        if use_batch:
            return self.call_batch(image, bboxes)
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        keypoints, scores = [], []
        for bbox in bboxes:
            img, center, scale = self.preprocess(image, bbox)
            outputs = self.inference(img)
            kpts, score = self.postprocess(outputs, center, scale)

            keypoints.append(kpts)
            scores.append(score)

        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)

        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        return keypoints, scores

    def call_batch(self, image: np.ndarray, bboxes: list = []):

        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        if MAIN_DEBUG: print("Batching RTMPose ...")
        if MAIN_DEBUG: t.tic()
        imgs, centers, scales = [], [], []
        for bbox in bboxes:
            img, c, s = self.preprocess(image, bbox)
            imgs.append(img)
            centers.append(c)
            scales.append(s)
        if MAIN_DEBUG: t.toc(); t.tic()

        # batched inference (helper defined outside the class)
        outputs = self.inference_batch(imgs)

        if MAIN_DEBUG: t.toc(); t.tic()

        # vectorised post-processing
        keypoints, scores = self.postprocess_batch(outputs,
                                                   np.asarray(centers),
                                                   np.asarray(scales))
        
        if MAIN_DEBUG: t.toc()

        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        return keypoints, scores

    def preprocess(self, img: np.ndarray, bbox: list):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            bbox (list):  xyxy-format bounding box of target.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        bbox = np.array(bbox)

        if PROC_DEBUG: print("Preprocessing RTMPose inputs...")

        if PROC_DEBUG: t.tic()

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        if PROC_DEBUG: t.toc(); t.tic()

        # do affine transformation
        resized_img, scale = top_down_affine(self.model_input_size, scale,
                                             center, img)
        
        if PROC_DEBUG: t.toc(); t.tic()

        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            resized_img = (resized_img - self.mean) / self.std
        
        if PROC_DEBUG: t.toc()

        return resized_img, center, scale

    def postprocess(
            self,
            outputs: List[np.ndarray],
            center: Tuple[int, int],
            scale: Tuple[int, int],
            simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        if PROC_DEBUG: print("Postprocessing RTMPose outputs...")
        if PROC_DEBUG: t.tic()

        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio
        if PROC_DEBUG: t.toc(); t.tic()

        # rescale keypoints
        keypoints = keypoints / self.model_input_size * scale
        keypoints = keypoints + center - scale / 2
        if PROC_DEBUG: t.toc(); 

        return keypoints, scores

    def inference(self, img: np.ndarray):
        """Inference model.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            outputs (np.ndarray): Output of RTMPose model.
        """
        
        # build input to (1, 3, H, W)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img[None, :, :, :]

        # run model
        if self.backend == 'onnxruntime':
            sess_input = {self.session.get_inputs()[0].name: input}
            sess_output = []
            for out in self.session.get_outputs():
                sess_output.append(out.name)

            outputs = self.session.run(sess_output, sess_input)
        else:
            raise NotImplementedError(f"Backend {self.backend} is not supported.")

        return outputs

    def inference_batch_batch_padding(self, imgs, target_batch_size=18):
        """
        Batched inference for RTMPose, padding batch to 'target_batch_size' for stable ONNXRuntime performance.

        Args
        ----
        imgs : Sequence[np.ndarray]
            List (or array) of images, each in HxWx3 format and already resized to the same HxW.

        Returns
        -------
        outputs : List[np.ndarray]
            A list of output tensors, each with shape (N, ...), corresponding to the original input count.
        """
        if INFER_DEBUG:
            t.tic()

        # Stack into (N, H, W, 3)
        batch_hwc = np.stack(imgs, axis=0).astype(np.float32)
        orig_batch_size, H, W, C = batch_hwc.shape

        if orig_batch_size < target_batch_size:
            pad_count = target_batch_size - orig_batch_size
            pad_imgs = np.zeros((pad_count, H, W, C), dtype=np.float32)
            batch_hwc = np.concatenate([batch_hwc, pad_imgs], axis=0)
        elif orig_batch_size > target_batch_size:
            raise ValueError(
                f"Got {orig_batch_size} images, but fixed batch size is {target_batch_size}. "
                "Modify code to handle more if needed."
            )

        if INFER_DEBUG:
            t.toc(); t.tic()

        # Convert to (N, 3, H, W)
        batch_chw = batch_hwc.transpose(0, 3, 1, 2)

        if INFER_DEBUG:
            t.toc(); t.tic()

        if self.backend == "onnxruntime":
            input_name = self.session.get_inputs()[0].name
            output_names = [o.name for o in self.session.get_outputs()]

            ort_inputs = {input_name: batch_chw}
            if INFER_DEBUG:
                t.toc(); t.tic()

            outputs = self.session.run(output_names, ort_inputs)

            if INFER_DEBUG:
                t.toc(); t.tic()

            # Trim outputs back to original batch size
            outputs = [o[:orig_batch_size] for o in outputs]

        else:
            raise NotImplementedError(
                f"Backend {self.backend} is not supported for batch inference."
            )

        return outputs


    def inference_batch(self, imgs):
        """
        Batched inference for RTMPose, without an explicit per-image loop for CHW conversion.

        Args
        ----
        imgs : Sequence[np.ndarray]
            List (or array) of images, each in HxWx3 format and already resized to the same HxW.

        Returns
        -------
        outputs : List[np.ndarray]
            A list of output tensors (one per model output), each having shape (N, …).
        """
        if INFER_DEBUG: t.tic()

        batch_hwc = np.stack(imgs, axis=0, dtype=np.float32)                  # shape = (N, H, W, 3)

        if INFER_DEBUG: t.toc(); t.tic()

        batch_chw = batch_hwc.transpose(0, 3, 1, 2)

        if INFER_DEBUG: t.toc(); t.tic()

        if self.backend == "onnxruntime":
            input_name   = self.session.get_inputs()[0].name
            output_names = [o.name for o in self.session.get_outputs()]
            ort_inputs = {input_name: batch_chw}
            if INFER_DEBUG: t.toc(); t.tic()
            outputs = self.session.run(output_names, ort_inputs)

            if INFER_DEBUG: t.toc(); t.tic()
        else:
            raise NotImplementedError(
                f"Backend {self.backend} is not supported for batch inference."
            )

        return outputs

    def postprocess_batch(
        self,
        outputs: List[np.ndarray],          # [simcc_x, simcc_y]
        centers: np.ndarray,                # (N, 2)  – cx, cy
        scales:  np.ndarray,                # (N, 2)  – w , h
        simcc_split_ratio: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode all poses in one shot.  No Python loop is executed after
        `get_simcc_maximum`.
        """
        # unpack ONNX tensors ------------------------------------------------
        simcc_x, simcc_y = outputs          # (N, K, Wx), (N, K, Wy)

        # ➊  get peak locations & confidences  (already batched)
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)   # (N, K, 2), (N,K)
        keypoints = locs / simcc_split_ratio                 # SIMCC → heat-map

        # ➋  map keypoints back to the original image ------------------------
        #     Broadcast over (N, K, 2) ←→ (N,1,2) and (1,1,2)
        model_size = np.asarray(self.model_input_size, dtype=np.float32)  # (2,)
        scales  = scales[:,  None, :]         # (N,1,2)
        centers = centers[:, None, :]         # (N,1,2)

        keypoints = (keypoints / model_size) * scales
        keypoints = keypoints + centers - scales / 2.0     # final coords

        return keypoints, scores