import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        evidences = []
        labels = []

        months = ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]

        for row in reader:
            evidence_for_row = []
            # make evidence by columns if that row, and add to evidence_for_row, then add evidence_for_row to evidences
            for i in range(16):
                if i == 10:
                    evidence_for_row.append(months.index(row[i]))
                elif i == 15:
                    if row[i] == "Returning_Visitor":
                        evidence_for_row.append(1)
                    else:
                        evidence_for_row.append(0)
                elif i == 16:
                    if row[i] == "TRUE":
                        evidence_for_row.append(1)
                    else:
                        evidence_for_row.append(0)
                else:
                    evidence_for_row.append(float(row[i]))

            evidences.append(evidence_for_row)

            if row[-1] == "TRUE":
                labels.append(1)
            else:
                labels.append(0)

    return evidences, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    x_training = evidence
    y_training = labels
    return model.fit(x_training, y_training)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity_num = 0
    sensitivity_denom = 0

    specifity_num = 0
    specifity_denom = 0
    total = 0
    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            sensitivity_denom += 1
            if predicted == 1:
                sensitivity_num += 1

        if actual == 0:
            specifity_denom += 1
            if predicted == 0:
                specifity_num += 1

    sensitivity = sensitivity_num/sensitivity_denom
    specificity = specifity_num/specifity_denom

    return sensitivity, specificity


if __name__ == "__main__":
    main()
