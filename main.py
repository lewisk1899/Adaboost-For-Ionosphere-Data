import numpy as np
from sklearn import tree

class Adaboost:
    def weaklearn(self, training_data, labels, weights, round):
        # randomly sample the indices of the distribution
        sampled_indices = sample(len(weights), weights)[0].astype(int)
        x_t, y_t = training_data[sampled_indices], labels[sampled_indices]
        ht = tree.DecisionTreeClassifier(max_depth=3).fit(x_t, y_t, weights)
        return ht

    def train(self, training_data, labels, training_rounds):
        weights = np.array([1 / training_data.shape[0]] * training_data.shape[0])  # initialize the difficulty weights to be 1/N N being the number of samples.
        round = 0
        while round < training_rounds:
            # weak learn
            hypothesis_class = self.weaklearn(training_data, labels, weights, round)
            prediction = hypothesis_class.predict(training_data) # make a prediction on the data after generating the hypothesis class
            prediction = np.reshape(prediction, (prediction.shape[0], 1))
            # find the error
            error = 0
            for i in range(prediction.shape[0]):
                wrong_prediction_indexes = []
                if not prediction[i] == labels[i]:
                    error = error + weights[i]
                    wrong_prediction_indexes.append([i, ['weight at ', i, weights[i]], ["error:" , error]])
            print(wrong_prediction_indexes)
                # else don't change the weights
            alpha_t = .5 * np.log((1 - error) / error)  # calculate the alpha t
            # update the weights
            for i in range(prediction.shape[0]):
                weights[i] = (weights[i] / (2 * np.sqrt(error * (1 - error)))) * np.exp(-1 * alpha_t * labels[i] * prediction[i])
            round += 1
        return hypothesis_class

    def predict(self, hypothesis_class, data_test, labels_test):
        prediction = hypothesis_class.predict(data_test) # predict with the hypothesis class we found
        number_wrong = 0
        for i in range(prediction.shape[0]):
            # count number wrong
            if prediction[i] != labels_test[i]:
                number_wrong += 1

        return number_wrong/prediction.shape[0]


def sample(N, p):
    random_sample = np.zeros(N)
    p_estimate = np.zeros(len(p))
    p_cdf = np.cumsum(p)
    counts = np.zeros(len(p))

    for n in range(N):
        # generate a random number on [0,1]
        x = np.random.rand()
        random_sample[n] = np.where(((p_cdf > x) * 1.0) == 1.)[0][0]
        counts[int(random_sample[n])] += 1

    p_estimate = counts / counts.sum()
    return random_sample, p_estimate

def read_from_file(filename):
    path = open('ionosphere_test.csv', 'rb')
    data = np.genfromtxt(path, delimiter=',')
    y_test = data[:, -1]
    y_test = np.reshape(y_test, (y_test.shape[0], 1))  # should be 351 by 1
    x_test = np.delete(data, np.s_[-1:], axis=1)  # remove the last column as it is already saved in the y_test var

    return x_test, y_test


def main():
    ada = Adaboost()
    data_training, labels_training = read_from_file('ionosphere_train.csv')
    hypothesis_class = ada.train(data_training, labels_training, training_rounds=10)
    data_testing, labels_testing = read_from_file('ionosphere_test.csv')
    accuracy = ada.predict(hypothesis_class, data_testing, labels_testing)
    print("Accuracy:")
    print(1 - accuracy)


main()
