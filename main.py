from __future__ import annotations

from pathlib import Path
import asyncio
import logging

import cv2
import kornia as K
import kornia_rs as KRS
from limbus.core import Component, InputParams, OutputParams, ComponentState 
from limbus.core.app import App


class DataGenerator(Component):
    def __init__(self, name: str, root: Path):
        super().__init__(name)
        self.root = root
        self.files = list(self.root.rglob("*.jpg"))
        self.index = 0

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        outputs.declare("data")

    async def forward(self):
        if self.index >= len(self.files):
            return ComponentState.FINISHED
        file = self.files[self.index].absolute()
        #img = torch.from_dlpack(KRS.read_image_rs(str(file)))
        img = cv2.imread(str(file))
        img_t = K.utils.image_to_tensor(img).float()
        img_t = K.color.bgr_to_rgb(img_t)
        await self.outputs.data.send(img_t)
        self.index += 1
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


class FaceDetectionComponent(Component):
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
        # TODO: handle image batch
        dets = []
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

    async def forward(self):
        imgs, detections = await asyncio.gather(
            self.inputs.images.receive(), self.inputs.detections.receive())
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
        #await self.outputs.img.send(img)
        cv2.imshow(self.name, img_vis)
        cv2.waitKey(1000)
        return ComponentState.OK


class Logger(Component):
    def __init__(self, name: str):
        super().__init__(name)
        self.log = logging.getLogger(name)

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        inputs.declare("data")

    async def forward(self):
        data = await self.inputs.data.receive()
        self.log.info(data.shape)
        return ComponentState.OK


class FaceDetectionApp(App):
    def create_components(self):
        self.dataset = DataGenerator("data", Path("/home/edgar/data/WIDER_val"))
        self.batcher = AutoBatcher("batcher", batch_size=1)
        self.detector = FaceDetectionComponent("detector")
        #self.viz = ImageShow("viz")
        self.viz = FaceDetectionViz("viz")
        self.logger = Logger("logger")
    
    def connect_components(self):
        self.dataset.outputs.data >> self.logger.inputs.data
        self.dataset.outputs.data >> self.batcher.inputs.data
        self.batcher.outputs.data >> self.detector.inputs.data
        self.detector.outputs.detections >> self.viz.inputs.detections
        self.batcher.outputs.data >> self.viz.inputs.images


if __name__ == "__main__":
    app = FaceDetectionApp()
    asyncio.run(app.run())