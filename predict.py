import numpy as np
import onnxruntime
import config

session_options = onnxruntime.SessionOptions()
session_bar = onnxruntime.InferenceSession(config.deepcreampy_bar_model, session_options)
session_mosaic = onnxruntime.InferenceSession(config.deepcreampy_mosaic_model, session_options)


def predict(censored, mask, is_mosaic=bool):
    censored = np.float32([censored])
    mask = np.float32([mask])

    session = session_mosaic if is_mosaic else session_bar
    return list(session.run(["add:0"], {
        "Placeholder:0": censored,
        "Placeholder_1:0": censored,
        "Placeholder_2:0": mask,
    })[0])[0]
