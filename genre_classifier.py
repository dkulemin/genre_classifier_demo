import joblib
import numpy as np


class GenreClassifier(object):
    def __init__(self):
        self.model = joblib.load("./src/Tfidf_OneVsRest_LogReg.pkl")
        self.const_probs = [0.43635526, 0.17050567, 0.00175783, 0.02208745, 0.14861302,
                            0.10409633, 0.20918487, 0.00666155, 0.04036518, 0.00671773,
                            0.03233133, 0.00636946, 0.00100639, 0.08053753, 0.07814176,
                            0.13075581, 0.00292913, 0.5005538 , 0.01290418, 0.00491794]
        self.classes_dict = {
                            -1: 'PREDICTION FAILED!',
                            0: 'ACTION',
                            1: 'ADVENTURE',
                            2: 'ANIMATION',
                            3: 'BIOGRAPHY',
                            4: 'COMEDY',
                            5: 'CRIME',
                            6: 'DRAMA',
                            7: 'FAMILY',
                            8: 'FANTASY',
                            9: 'HISTORY',
                            10: 'HORROR',
                            11: 'MUSIC',
                            12: 'MUSICAL',
                            13: 'MYSTERY',
                            14: 'ROMANCE',
                            15: 'SCI-FI',
                            16: 'SPORT',
                            17: 'THRILLER',
                            18: 'WAR',
                            19: 'WESTERN'
                            }

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.5:
            return "not sure"
        if probability < 0.7:
            return "probably"
        if probability > 0.95:
            return "definitely"
        else:
            return "accuratly"

    def get_most_probable_genres(self, probs_genres_list, threshold = 0.25):
        result = []
        for pair in probs_genres_list:
            if pair[0] >= threshold:
                result.append((pair[1], self.get_probability_words(pair[0]), f'{int(pair[0] * 100)}%'))
        if result:
            return result
        else:
            return [('probably', 'DRAMA', '50%')]

    def predict_text(self, text):
        try:
            return self.model.predict_proba([text])[0]
        except:
            print('PREDICTION FAILED!')
            return -1

    def get_prediction_message(self, text):
        prediction_probability = self.predict_text(text)
        check_for_const = abs(prediction_probability - self.const_probs)
        if np.all(check_for_const < 1e-08):
            return [('', self.classes_dict[-1], '')]
        probs_genres_list = list(zip(prediction_probability, list(self.classes_dict.values())[1:]))
        return self.get_most_probable_genres(sorted(probs_genres_list, reverse=True))