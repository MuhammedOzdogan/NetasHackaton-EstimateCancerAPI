import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def get_data() -> pd.DataFrame:
    df = pd.read_excel('static/data/cancer-no.xls')
    return df


def get_features() -> []:
    return [
        'Age',
        'Gender',
        'Air Pollution',
        'Alcohol use',
        'Dust Allergy',
        # 'OccuPational Hazards',
        # 'Genetic Risk',
        'chronic Lung Disease',
        'Balanced Diet',
        'Obesity',
        'Smoking',
        # 'Passive Smoker',
        'Chest Pain',
        'Coughing of Blood',
        'Fatigue',
        'Weight Loss',
        'Shortness of Breath',
        'Wheezing',
        'Swallowing Difficulty',
        'Clubbing of Finger Nails',
        'Frequent Cold',
        'Dry Cough',
        'Snoring',
    ]


def train_model() -> LinearRegression:
    df = get_data()
    features = get_features()
    feature_matrix = df.loc[:, features].values
    target_vector = df.loc[:, 'Level'].values
    print("Feture matrix shape:", feature_matrix.shape)
    print("Target matrix shape:", target_vector.shape)

    x_train, x_test, y_train, y_test = train_test_split(feature_matrix, target_vector, random_state=3)

    reg = LinearRegression(fit_intercept=True)
    reg.fit(x_train, y_train)
    score = reg.score(x_test, y_test)
    print("With intercept,score {:.2f}".format(score))

    reg_no_inter = LinearRegression(fit_intercept=False)
    reg_no_inter.fit(x_train, y_train)
    score_no_inter = reg_no_inter.score(x_test, y_test)
    print("Without intercept,score {:.2f}".format(score_no_inter))

    # print("Test")
    # test_data = feature_matrix[2].reshape(1, -1)
    # result = reg.predict(test_data)
    # print(result)
    return reg


if __name__ == '__main__':
    train_model()
