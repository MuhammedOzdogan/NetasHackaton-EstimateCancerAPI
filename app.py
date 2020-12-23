from flask import Flask
import numpy as np
from flask import request
import ml

app = Flask(__name__)

model = ml.train_model()


@app.route('/')
def main():
    if request.args.__len__() == 0:
        return app.send_static_file("error.html")
    if not is_args_valid():
        return "Apart from age attribute feature values must be between 1 and 10"

    x = read_feature_matrix_from_query_params()
    result = model.predict(x)

    return {
        "cancer_predict": result[0]
    }


def is_args_valid():
    for key in request.args:
        v = int(request.args.get(key))
        if  key != "age" and v > 10 or v < 1:
            return False

    return True


def read_feature_matrix_from_query_params() -> np.array:
    return np.array([[
        arg("age"),
        arg("gender"),
        arg("air_pollution"),
        arg("alcohol"),
        arg("dusty_allergy"),
        arg("chronic_lung_disease"),
        arg("balanced_diet"),
        arg("obesity"),
        arg("smoking"),
        arg("chest_pain"),
        arg("coughing_of_blood"),
        arg("fatigue"),
        arg("weight_loss"),
        arg("shortness_of_breath"),
        arg("wheezing"),
        arg("swallowing_difficulty"),
        arg("clubbing_of_finger_nails"),
        arg("frequent_cold"),
        arg("dry_cough"),
        arg("snoring")
    ]])


def arg(key) -> int:
    return int(request.args.get(key))


if __name__ == '__main__':
    app.run()
