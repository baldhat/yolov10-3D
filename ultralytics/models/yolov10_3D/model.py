from ultralytics.engine.model import Model
from ultralytics.nn.tasks import YOLOv10_3DDetectionModel
from .val import YOLOv10_3DDetectionValidator
from .predict import YOLOv10_3DDetectionPredictor
from .train import YOLOv10_3DDetectionTrainer

class YOLOv10_3D(Model):

    def __init__(self, model="yolov10n_3D.pt", task=None, verbose=False,
                 names=None):
        super().__init__(model=model, task=task, verbose=verbose)
        if names is not None:
            setattr(self.model, 'names', names)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10_3DDetectionModel,
                "trainer": YOLOv10_3DDetectionTrainer,
                "validator": YOLOv10_3DDetectionValidator,
                "predictor": YOLOv10_3DDetectionPredictor,
            },
        }