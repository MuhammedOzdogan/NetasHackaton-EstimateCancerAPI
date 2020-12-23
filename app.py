from flask import Flask
import numpy as np
from flask import request
import ml

app = Flask(__name__)

model = ml.train_model()


@app.route('/')
def hello_world():
    x = read_feature_matrix_from_query_params()
    result = model.predict(x)

    return {
        "cancer_predict": result[0]
    }


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
