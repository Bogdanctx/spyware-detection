import pandas as pd
import torch

from src.feature_scheme import PE
from src.judge_random_forest import JudgeRandomForest
from src.judge_logistic_regression import JudgeLogisticRegression
from src.judge_svm import JudgeSVM
from src.judge_xgboost import JudgeXGBoost
from src.judge_bytecnn import JudgeByteCNN

class Council:
    def __init__(self):
        self.judges = {
            'logistic_regression': JudgeLogisticRegression(),
            'random_forest': JudgeRandomForest(),
            'svm': JudgeSVM(),
            'xgboost': JudgeXGBoost(),
            'cnn': JudgeByteCNN()
        }

        self.weights = {
            'logistic_regression': 95.42,
            'random_forest': 97.91,
            'svm': 95.39,
            'xgboost': 98.78,
            'cnn': 95.17
        }

        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total

        self.judges['logistic_regression'].load_model('./models/judge_logistic_regression.joblib')
        self.judges['random_forest'].load_model('./models/judge_random_forest.joblib')
        self.judges['svm'].load_model('./models/judge_svm.joblib')
        self.judges['xgboost'].load_model('./models/judge_xgboost.joblib')
        self.judges['cnn'].load_model('./models/judge_cnn.pth')

        self.cache_y_true = []
        self.cache_y_pred = []
        self.misclassified = []

    def judge(self, path):
        results = {}
        for name, judge in self.judges.items():
            feature = None
            if name != 'cnn':
                pe = PE(path)
                feature = pe.get_features()
                feature = pd.DataFrame([feature])
                feature = feature[judge.name_of_features].to_numpy()
            else:
                view_len = JudgeByteCNN.CONFIG["VIEW_LEN"]
                with open(path, 'rb') as f:
                    bytes_data = f.read(view_len)
                    if len(bytes_data) < view_len:
                        bytes_data += b'\x00' * (view_len - len(bytes_data))

                feature = torch.frombuffer(bytes_data, dtype=torch.uint8).unsqueeze(0)
                feature = feature.to(JudgeByteCNN.CONFIG["DEVICE"])

            results[name] = judge.predict_proba(feature)[0][1] # probability of being malware

        weighted_vote = sum(results[j] * self.weights[j] for j in self.judges)
        verdict = 1 if weighted_vote >= 0.35 else 0

        return {"verdict": verdict, "confidence_score": weighted_vote, "details": results}