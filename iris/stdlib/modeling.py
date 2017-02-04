from .. import IrisCommand
from .. import state_types as t
from .. import state_machine as sm
from .. import util as util

class MakeModel(IrisCommand):
    title = "create a new classification model"
    examples = [ "build a new classification model",
                 "make a new classification model" ]
    argument_types = { "x_features": t.DataframeSelector("What dataframe do you want to use to select the features?"), #t.ArgList(question="Please give me a comma-separated list of features"),
                       "y_classes": t.DataframeSelector("What dataframe holds the values to be predicted?"), }#t.ArgList(question="What would you like to predict?") }
    help_text = [
        "A classification model is designed to predict a categorical variable, like eye color or gender.",
        "This command takes a list of input features, as arrays, that will be used to predict the category in question.",
        "It then takes the name of an array that corresponds to the category to be predicted."
    ]
    store_result = t.VarName(question="What should I call the model?")
    def command(self, x_features, y_classes):
        model = LogisticRegression()
        y_classes = y_classes.reshape(y_classes.shape[0])
        model.fit(x_features, y_classes)
        if "names" in self.context:
            name = self.context["names"][0].name
        else:
            name = None
        # we use IrisModel here because it retains a link to X, y data
        # this can be useful for cross-validation, etc.
        return iris_objects.IrisModel(model, x_features, y_classes, name=name)
    def explanation(self, results):
        return [] # do nothing

makeModel = MakeModel()

class TrainTestSplit(IrisCommand):
    title = "create training and test data splits"
    examples = [ "create train test data",
                 "split data into train and test" ]
    store_result = [ t.VarName(question="Where to store training data?"),
                     t.VarName(question="Where to store testing data?") ]
    help_text = [
        "This command takes a dataset and splits it into training and testing data.",
        "By convention, training data is used to train a model, and testing data is used to evaluate a model's performance."
    ]
    def command(self, x_features : t.ArgList(), y_classes : t.ArgList()):
        from sklearn.model_selection import train_test_split
        xvals = np.array(x_features).T
        yvals = np.array(y_classes).T
        yvals = yvals.reshape(yvals.shape[0])
        x_train, x_test, y_train, y_test = train_test_split(xvals, yvals, train_size=0.25)
        train_data = iris_objects.IrisData(x_train, y_train)
        test_data = iris_objects.IrisData(x_test, y_test)
        return train_data, test_data

trainTestSplit = TrainTestSplit()

class TrainModel(IrisCommand):
    title = "train {model} on {data}"
    examples = [ "train {model} {data}",
                 "train model {model} on data {data}" ]
    help_text = [
        "This command trains a model on a dataset. The model will be fit on the provided data."
    ]
    def command(self, iris_model : t.EnvVar(), iris_data : t.EnvVar()):
        iris_model.model.fit(iris_data.X, iris_data.y)
    def explanation(self, *args):
        return "I fit the model."

trainModel = TrainModel()

class TestModelF1(IrisCommand):
    title = "test {iris_model} on {iris_data} using f1 score with {weighting}"
    examples = [ "test {iris_model} {iris_data} with {weighting}",
                 "test model {iris_model} on data {data} with {weighting}" ]
    help_text = [
        "This command evaluates a model on a dataset using the F1 metric.",
        "F1 is a common metric that balances precision and recall."
        "When you evaluate a model, you should not use data the model has seen during training."
    ]
    def __init__(self):
        metrics = { "Binary: report results for the class specified by pos_label. Data must be binary.": "binary",
                    "Micro: calculate metrics globally by counting the total true positives, false negatives and false positives.": "micro",
                    "Macro: calculate metrics for each label, and find their unweighted mean (does not take label imbalance into account).": "macro" }
        select_classifier = t.Select(options=metrics, default="binary")
        self.argument_types = { "iris_model": t.EnvVar(),
                                "iris_data": t.EnvVar(),
                                "weighting": select_classifier }
        super().__init__()
    def command(self, iris_model, iris_data, weighting):
        from sklearn.metrics import f1_score
        pred_y = iris_model.model.predict(iris_data.X)
        score = f1_score(iris_data.y, pred_y, average=weighting)
        return score
    def explanation(self, score):
        score = round(score, 4)
        return "F1 score of {}".format(score)

testModelF1 = TestModelF1()

class CrossValidateModel(IrisCommand):
    title = "cross-validate {model} with {score} and {n} folds"
    examples = [ "cross-validate {model} {score} {n}", "evaluate model performance" ]
    help_text = [
        "This command evaluates a model through cross-validation, using either accuracy or F1 score.",
        "Cross-validation is a common way to evaluate a model.",
        "For each fold in n fold cross-validation, this command will train on n-1 folds and evaluate the model on the held out fold.",
        "The resulting scores will be averaged together as a measure of overall performance."
    ]
    argument_types = {
        "model": t.EnvVar(),
        "score": t.Select(options={
            "Accuracy: correct predictions / incorrect predictions": "accuracy",
            "F1 macro: f1 score computed with average across classes": "f1_macro",
            "F1 binary: f1 score computed on the positive class": "f1"
        }, default="accuracy"),
        "n": t.Int()
    }
    argument_help = {
        "score": t.YesNo(["I'm happy to help.",
                          "Are the classes balanced in the data (does each class have the same number of examples)?"],
                    yes=sm.DoAll([sm.Print(["Great, let's use accuracy"]), sm.ValueState("accuracy")]),
                    no=t.YesNo("Do you have more than two classes?",
                            yes=sm.DoAll([sm.Print(["Great, let's use f1_macro.",
                                                    "That's a standard metric for mult-class analysis"]),
                                              sm.ValueState("f1_macro")]),
                            no=sm.DoAll([sm.Print(["Great, let's use f1 (defaults to binary).",
                                                   "That's the conventional metric for binary data."]),
                                             sm.ValueState("f1_binary")])))
    }
    def command(self, model, score, n):
        from sklearn.cross_validation import cross_val_score
        return cross_val_score(model.model, model.X, model.y, scoring = score, cv=n)
    def explanation(self, score):
        import numpy as np
        score = round(np.average(score), 4)
        return "Average performance of {} across the folds".format(score)

crossValidateModel = CrossValidateModel()

class CompareModels(IrisCommand):
    title = "compare {model1} and {model2} using {metric}"
    examples = [ "compare {model1} {model2} using {metric}",
                 "which model is better under {metric}, {model1} or {model2}" ]
    def __init__(self):
        metrics = { "Accuracy: correct predictions / incorrect predictions": "accuracy",
                    "F1 macro: f1 score computed with average across classes": "f1_macro",
                    "F1 micro: f1 score computed with weighted average": "f1_micro" }
        select_metric = t.Select(options=metrics, default="accuracy")
        self.argument_types = { "model1": t.EnvVar(),
                                "model2": t.EnvVar(),
                                "metric": select_metric }
        super().__init__()
    help_text = [
        "This command takes two models, and determines which performs better under a given metric."
    ]
    def command(self, model1, model2, metric):
        import numpy as np
        m1_scores = np.average(crossValidateModel(model1, metric, 10))
        m2_scores = np.average(crossValidateModel(model2, metric, 10))
        if m1_scores > m2_scores:
            higher_m, lower_m = model1, model2
            higher_s, lower_s = m1_scores, m2_scores
        else:
            higher_m, lower_m = model2, model1
            higher_s, lower_s = m2_scores, m1_scores
        return (higher_m.name, higher_s), (lower_m.name, lower_s)
    def explanation(self, results):
        higher_tuple, lower_tuple = results
        higher_name, lower_name = [x[0] for x in [higher_tuple, lower_tuple]]
        higher_score, lower_score = [round(x[1],4) for x in [higher_tuple, lower_tuple]]
        return "I'd say \"{}\" is better than \"{}\", with {} vs. {}".format(higher_name, lower_name, higher_score, lower_score)

compareModels = CompareModels()


class FindRegularization(IrisCommand):
    title = "find the best l2 regularization parameter for {model} with {metric}"
    examples = [ "best regularization for {model} {metric}",
                 "best l2 parameter for {model} under {metric}" ]
    def __init__(self):
        metrics = { "Accuracy: correct predictions / incorrect predictions": "accuracy",
                    "F1 macro: f1 score computed with average across classes": "f1_macro",
                    "F1 micro: f1 score computed with weighted average": "f1_micro" }
        select_metric = t.Select(options=metrics, default="accuracy")
        self.argument_types = { "model": t.EnvVar(),
                                "metric": select_metric }
        super().__init__()
    help_text = [
        "This command finds the best L2 parameter for a model, under accuracy or F1.",
        "Regularization parameters are useful to prevent a model from overfitting.",
        "Finding a good parameter can improve model performance."
    ]
    def command(self, model, metric):
        from sklearn.cross_validation import cross_val_score
        import numpy as np
        best_score = 0
        best_c = None
        for c in [0.01, 0.1, 1, 10, 100]:
            score = np.average(crossValidateModel(model, metric, 5))
            if score > best_score:
                best_score = score
                best_c = c
        return best_c, best_score, metric
    def explanation(self, results):
        best_c, best_score, metric = results
        best_score = round(best_score, 4)
        return "Best L2 of {} with {} {}".format(best_c, best_score, metric)

findRegularization = FindRegularization()
