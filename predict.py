import numpy as np
import config
from onnx_runtime import create_session

session_bar = create_session(
    config.deepcreampy_bar_model,
    providers=["CPUExecutionProvider"],
)
session_mosaic = create_session(
    config.deepcreampy_mosaic_model,
    providers=["CPUExecutionProvider"],
)


def predict(censored, mask, is_mosaic=bool):
    censored = np.float32([censored])
    mask = np.float32([mask])

    session = session_mosaic if is_mosaic else session_bar
    return list(session.run(["add:0"], {
        "Placeholder:0": censored,
        "Placeholder_1:0": censored,
        "Placeholder_2:0": mask,
    })[0])[0]
