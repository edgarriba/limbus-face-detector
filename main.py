from __future__ import annotations

from pathlib import Path
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


class DataGenerator(Component):
    def __init__(self, name: str, root: Path, num_processes: int = 1) -> None:
        super().__init__(name)
        self.files: list[Path] = list(root.rglob("*.jpg"))
        num_splits = len(self.files) // num_processes
        self.split_files = [
            self.files[i * num_splits:(i + 1) * num_splits]
            for i in range(num_processes)]

        self.queue = amp.AioQueue()
        self.process = [
            amp.AioProcess(target=self._run, args=(split_files, self.queue))
            for split_files in self.split_files]
        for p in self.process:
            p.start()

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        outputs.declare("data")
        outputs.declare("t0")
    
    def _run(self, files, queue):
        for file in files:
            img = cv2.imread(str(file))
            img_t = K.utils.image_to_tensor(img).float()
            img_t = K.color.bgr_to_rgb(img_t)
            #img_t = K.geometry.rescale(img_t, 0.5)
            # queue.put(torch.to_dlpack(img_t))
            queue.put(img_t)

    async def forward(self):
        t0 = time.time()
        data = await self.queue.coro_get()
        await self.outputs.data.send(data.cuda())
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
        self.model = K.contrib.FaceDetector().cuda()

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
        self.dataset = DataGenerator("data", Path("/home/edgar/data/WIDER_val"))
        self.batcher = AutoBatcher("batcher", batch_size=1)
        self.detector = FaceDetection("detector")
        self.viz = FaceDetectionViz("viz")
    
    def connect_components(self):
        self.dataset.outputs.data >> self.batcher.inputs.data
        self.batcher.outputs.data >> self.detector.inputs.data
        self.detector.outputs.detections >> self.viz.inputs.detections
        self.batcher.outputs.data >> self.viz.inputs.images
        self.dataset.outputs.t0 >> self.viz.inputs.t0


if __name__ == "__main__":
    asyncio.run(FaceDetectionApp().run())