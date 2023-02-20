import asyncio

import torch
import kornia as K
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.boxes import Boxes

from limbus.core import Component, InputParams, ComponentState 


def box_iou(box1: Boxes, box2: Boxes) -> Tensor:
    """Compute the intersection over union of two set of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        box1: boxes, sized [N,4,2].
        box2: boxes, sized [M,4,2].

    Return:
        iou, sized [N,M].
    """
    KORNIA_CHECK_SHAPE(box1, ["N", "4", "2"])
    KORNIA_CHECK_SHAPE(box2, ["M", "4", "2"])

    area1 = box1.area()  # [N,]
    area2 = box2.area()  # [M,]

    br = torch.min(box1.bottom_right()[:, None], box2.bottom_right())  # [N, M, 2]
    tl = torch.max(box1.top_left()[:, None], box2.top_left())  # [N, M, 2]

    inter = (br - tl).clamp(min=0).prod(2)  # [N,M]

    return inter / (area1[:, None] + area2 - inter) # [N,M]


class FaceDetectorMetric(Component):
    def __init__(self, name: str):
        super().__init__(name)
    
    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        inputs.declare("detections", K.contrib.FaceDetectorResult)
        inputs.declare("boxes", Boxes)
    
    async def forward(self):
        batched_detections, batched_boxes_gt = await asyncio.gather(
            self.inputs.detections.receive(),
            self.inputs.boxes.receive()
        )
        # convert detections to Boxes
        # TODO: handle batch properly
        batched_boxes_gt.data.unsqueeze_(0)

        for dets, boxes_gt in zip(batched_detections, batched_boxes_gt):
            boxes = [d.top_left.tolist() + d.bottom_right.tolist() for d in dets]
            boxes = Boxes.from_tensor(torch.tensor(boxes), mode="xyxy")

            # compute iou
            iou = box_iou(boxes, boxes_gt)

            # compute confusion matrix
            # TODO: this is not correct, just for debug
            iou_thresh = 0.45
            print(iou.mean())
            pass
        return ComponentState.OK
