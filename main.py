from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import asyncio
import logging
import multiprocessing as mp
import aioprocessing as amp
import time

import torch
import cv2
import kornia as K
import kornia_rs as KRS
from limbus.core import Component, InputParams, OutputParams, ComponentState 
from limbus.core.app import App

from metrics import FaceDetectorMetric

@dataclass
class DataSample:
    image: K.core.Tensor = None
    image_path: Path = None
    label: K.core.Tensor = None


class DataGenerator(Component):
    def __init__(self, name: str, root: Path, images: Path, labels: Path, num_processes: int = 1) -> None:
        super().__init__(name)
        # load images
        #self.image_files: list[Path] = list((root / "images").rglob("*.jpg"))
        
        # load labels
        self.data = []
        labels = open(labels).readlines()
        labels_iter = iter(labels)
        while True:
            try:
                line = next(labels_iter)
            except StopIteration:
                break
            if ".jpg" in line:
                image_path = Path(line.strip())
                num_boxes = int(next(labels_iter).strip())
                boxes = []
                for _ in range(num_boxes):
                    box = [int(x) for x in next(labels_iter).strip().split(" ")]
                    boxes.append(box)
                self.data.append(DataSample(
                    image_path=(root / "images" / image_path),
                    label=K.geometry.boxes.Boxes.from_tensor(torch.tensor(boxes)[..., :4], mode="xywh", validate_boxes=False)))

        # split data
        num_splits = len(self.data) // num_processes
        self.data_split = [
            self.data[i * num_splits:(i + 1) * num_splits]
            for i in range(num_processes)]

        self.queue = amp.AioQueue()
        self.process = [
            amp.AioProcess(target=self._run, args=(data_split, self.queue))
            for data_split in self.data_split]
        for p in self.process:
            p.start()

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        outputs.declare("image")
        outputs.declare("boxes")
        outputs.declare("t0")
    
    @staticmethod
    def _run(data, queue):
        for sample in data:
            assert sample.image_path.exists(), sample.image_path
            img = cv2.imread(str(sample.image_path))
            img_t = K.utils.image_to_tensor(img).float()
            img_t = K.color.bgr_to_rgb(img_t)
            sample.image = img_t
            queue.put(sample)

    async def forward(self):
        t0 = time.time()
        sample: DataSample = await self.queue.coro_get()
        await asyncio.gather(
            self.outputs.image.send(sample.image),
            self.outputs.boxes.send(sample.label)
        )
        await self.outputs.t0.send(t0)
        return ComponentState.OK


class AutoBatcher(Component):
    def __init__(self, name: str, batch_size: int):
        super().__init__(name)
        self.batch_size = batch_size
        self.batch = []

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        inputs.declare("data")

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        outputs.declare("data")

    async def forward(self):
        data = await self.inputs.data.receive()
        self.batch.append(data)
        if len(self.batch) == self.batch_size:
            batch = K.core.stack(self.batch, dim=0)
            await self.outputs.data.send(batch)
            self.batch = []
        return ComponentState.OK


class FaceDetection(Component):
    def __init__(self, name: str):
        super().__init__(name)
        self.model = K.contrib.FaceDetector()

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        inputs.declare("data")

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        outputs.declare("detections")

    async def forward(self):
        img: K.core.Tensor = await self.inputs.data.receive()
        out: K.core.Tensor = self.model(img)
        dets: list[list[K.contrib.FaceDetectorResult]] = []
        for o in out:
            dets.append([K.contrib.FaceDetectorResult(d) for d in o])
        await self.outputs.detections.send(dets)
        return ComponentState.OK


class FaceDetectionViz(Component):
    def __init__(self, name: str):
        super().__init__(name)

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        inputs.declare("images")
        inputs.declare("detections")
        inputs.declare("t0")

    async def forward(self):
        imgs, detections, t0 = await asyncio.gather(
            self.inputs.images.receive(),
            self.inputs.detections.receive(),
            self.inputs.t0.receive()
        )
        assert len(imgs.shape) == 4, imgs.shape

        for img_t, dets in zip(imgs, detections):
            img_vis = K.color.rgb_to_bgr(img_t)
            img_vis = K.utils.tensor_to_image(img_vis.byte()).copy()
            for d in dets:
                if d.score < 0.5:
                    continue
                x1, y1 = d.top_left.int().tolist()
                x2, y2 = d.bottom_right.int().tolist()
                img_vis = cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        t1 = time.time()
        print(f"FPS: {1 / (t1 - t0)}")
        #cv2.imshow(self.name, img_vis)
        #cv2.waitKey(1000)
        return ComponentState.OK


class FaceDetectionApp(App):
    def create_components(self):
        self.dataset = DataGenerator(
            "data",
            root=Path("/home/edgar/data/WIDER_val"),
            images=Path("/home/edgar/data/WIDER_val"),
            labels=Path("/home/edgar/Downloads/wider_face_split/wider_face_val_bbx_gt.txt"))
        self.batcher = AutoBatcher("batcher", batch_size=1)
        self.detector = FaceDetection("detector")
        self.viz = FaceDetectionViz("viz")
        self.metric = FaceDetectorMetric("metric")
    
    def connect_components(self):
        self.dataset.outputs.image >> self.batcher.inputs.data
        self.dataset.outputs.boxes >> self.metric.inputs.boxes
        self.batcher.outputs.data >> self.detector.inputs.data
        self.detector.outputs.detections >> self.viz.inputs.detections
        self.detector.outputs.detections >> self.metric.inputs.detections
        self.batcher.outputs.data >> self.viz.inputs.images
        self.dataset.outputs.t0 >> self.viz.inputs.t0


if __name__ == "__main__":
    asyncio.run(FaceDetectionApp().run())