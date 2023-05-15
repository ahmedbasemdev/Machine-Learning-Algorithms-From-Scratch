import numpy as np


class NaiveBayes():

    def __init__(self):
        self.prior_probability = {}
        self.marginal_probability = {}
        self.likelihood = {}

    def cal_prior_probability(self):
        for target in self.classes:
            target_count = sum(self.y_train == target)
            self.prior_probability[target] = target_count / len(self.y_train)

    def cal_marginal_probability(self):
        for feature in self.features:
            self.marginal_probability[feature] = {}
            feature_values = self.x_train[feature].value_counts().to_dict()
            for feat, count in feature_values.items():
                self.marginal_probability[feature][feat] = count / self.num_samples

    def cal_likelihood(self):
        for feature in self.features:
            self.likelihood[feature] = {}
            for target in self.classes:

                target_count = sum(self.y_train == target)

                feature_likelihood = self.x_train[feature][
                    self.y_train[self.y_train == target].index.values.tolist()].value_counts().to_dict()
                # print(self.x_train[feature][self.y_train[self.y_train == target].index.values].value_counts().to_dict())

                for feat, count in feature_likelihood.items():
                    self.likelihood[feature][feat + ":" + target] = count / target_count

    def fit(self, X, Y):

        self.y_train = Y
        self.x_train = X
        self.features = X.columns
        self.num_samples = X.shape[0]
        self.num_features = X.shape[1]
        self.classes = np.unique(Y)

        self.cal_prior_probability()
        self.cal_marginal_probability()
        self.cal_likelihood()

    def predict(self, X):

        results = []

        X = np.array(X)

        for sample in X:
            probs_target = {}
            for target in self.classes:
                prior_prob = self.prior_probability[target]
                total_likelihood = 1
                total_marginal = 1

                for feat, feat_value in zip(self.features, sample):

                    total_likelihood *= self.likelihood[feat].get(feat_value + ":" + target, 0)
                    total_marginal *= self.marginal_probability[feat][feat_value]
                posterior_prob = (total_likelihood * prior_prob) / total_marginal
                probs_target[target] = posterior_prob


            result = max(probs_target, key=lambda x: probs_target[x])
            results.append(result)
        return np.array(results)
