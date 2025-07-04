import numpy as np

from pytictoc import TicToc

t = TicToc()
class Body:
    MODE = {
        'performance': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',  # noqa
            'pose_input_size': (288, 384),
        },
        'lightweight': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip',  # noqa
            'det_input_size': (416, 416),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip',  # noqa
            'pose_input_size': (192, 256),
        },
        'balanced': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',  # noqa
            'pose_input_size': (192, 256),
        }
    }

    RTMO_MODE = {
        'performance': {
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip',  # noqa
            'pose_input_size': (640, 640),
        },
        'lightweight': {
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip',  # noqa
            'pose_input_size': (640, 640),
        },
        'balanced': {
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip',  # noqa
            'pose_input_size': (640, 640),
        }
    }

    def __init__(self,
                 det: str = None,
                 det_input_size: tuple = (640, 640),
                 pose: str = None,
                 pose_input_size: tuple = (288, 384),
                 mode: str = 'balanced',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):

        from rtmlib.tools.object_detection.yolox import YOLOX
        from rtmlib.tools.pose_estimation.rtmpose_batch import RTMPose

        self.one_stage = False

        if pose is None:
            pose = self.MODE[mode]['pose']
            pose_input_size = self.MODE[mode]['pose_input_size']

        if det is None:
            det = self.MODE[mode]['det']
            det_input_size = self.MODE[mode]['det_input_size']

        self.det_model = YOLOX(det,
                                model_input_size=det_input_size,
                                backend=backend,
                                device=device)
        self.pose_model = RTMPose(pose,
                                    model_input_size=pose_input_size,
                                    to_openpose=to_openpose,
                                    backend=backend,
                                    device=device)

    def __call__(self, images):

        # t.tic()
        bboxes_list = []
        for image in images:
            bboxes_list.append(self.det_model(image))

        # t.toc(); t.tic()
        keypoints_list, scores_list = self.pose_model(images, bboxes_list=bboxes_list)

        # t.toc()
        return keypoints_list, scores_list
