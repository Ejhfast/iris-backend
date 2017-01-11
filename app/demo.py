import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from iris import IRIS, IrisCommand
from iris import iris_types as t
from iris import dynamic_state as dt
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import fileinput
import numpy as np
import math

class SaveEnv(IrisCommand):
    title = "save environment to {name}"
    examples = [ "save environment {name}",
                 "save env to {name}" ]
    help_text = [
        "This command saves the current environment (all data in the left pane).",
        "This data can be loaded later using the 'load environment' command."
    ]
    def command(self, name : t.String(question="What filename to save under?")):
        import pickle
        with open(name, 'wb') as f:
            pickle.dump({"env":self.iris.env, "env_order":self.iris.env_order}, f)
            return "Saved to {}.".format(name)

saveEnv = SaveEnv()

class LoadEnv(IrisCommand):
    title = "load environment from {name}"
    examples = [ "load environment {name}",
                 "load env from {name}" ]
    help_text = [
        "This command loads an enviornment previously saved by Iris."
    ]
    def command(self, name : t.String(question="What filename to load?")):
        import pickle
        with open(name, 'rb') as f:
            data = pickle.load(f)
            self.iris.env = data["env"]
            self.iris.env_order = data["env_order"]
            return "Loaded environment from \"{}\".".format(name)

loadEnv = LoadEnv()

class AddTwoNumbers(IrisCommand):
    title = "add two numbers: {x} and {y}"
    examples = [ "add {x} and {y}",
                 "add {x} {y}" ]
    argument_types = {
        "x": dt.Int("Please enter a number for x:"),
        "y": dt.Int("Please enter a number for y:")
    }
    help_text = [
        "This command performs addition on two numbers, e.g., 'add 3 and 2' will return 5"
    ]
    def command(self, x, y):
        return x + y

addTwoNumbers = AddTwoNumbers()

class DivideTwoNumbers(IrisCommand):
    title = "divide two numbers: {x} and {y}"
    examples = [ "divide {x} and {y}",
                 "divide {x} {y}" ]
    help_text = [
        "This command performs division on two numbers, e.g., 'divide 6 and 3' will return 2"
    ]
    def command(self, x : t.Int(), y : t.Int()):
        return x / y

divideTwoNumbers = DivideTwoNumbers()

class PearsonCorrelation(IrisCommand):
    title = "compute pearson correlation: {x} and {y}"
    examples = [ "pearson correlation between {x} and {y}",
                 "pearson correlation {x} {y}",
                 "how are {x} and {y} correlated" ]
    help_text = [
        "A pearson correlation coefficient is a measure of linear dependence between two variables.",
        "A coefficient greater than 0 indicates a positive linear relationship",
        "A coefficient less than 0 indicates a negative relationship",
        "And a coefficient near 0 indicates the absence of any relationship.",
        "This command returns a coefficient and a p-value that measures the degree of confidence in its significance."
    ]
    def command(self, x : t.Array(), y : t.Array()):
        from scipy.stats import pearsonr
        return pearsonr(x,y)
    def explanation(self, corr_pval):
        corr = round(corr_pval[0],4)
        pval = round(corr_pval[1],4)
        return "Correlation of {} with p-value of {}".format(corr, pval)

pearsonCorrelation = PearsonCorrelation()

class MakeModel(IrisCommand):
    title = "create a new classification model"
    examples = [ "build a new classification model",
                 "make a new classification model" ]
    argument_types = { "x_features": t.ArgList(question="Please give me a comma-separated list of features"),
                       "y_classes": t.ArgList(question="What would you like to predict?") }
    help_text = [
        "A classification model is designed to predict a categorical variable, like eye color or gender.",
        "This command takes a list of input features, as arrays, that will be used to predict the category in question.",
        "It then takes the name of an array that corresponds to the category to be predicted."
    ]
    store_result = t.StoreName(question="What should I call the model?")
    def command(self, x_features, y_classes):
        model = LogisticRegression()
        X = np.array(x_features).T
        y = np.array(y_classes).T
        y = y.reshape(y.shape[0])
        model.fit(X,y)
        if "names" in self.context:
            name = self.context["names"][0].name
        else:
            name = None
        # we use IrisModel here because it retains a link to X, y data
        # this can be useful for cross-validation, etc.
        return t.IrisModel(model, X, y, name=name)
    def explanation(self, results):
        return [] # do nothing

makeModel = MakeModel()

class TrainTestSplit(IrisCommand):
    title = "create training and test data splits"
    examples = [ "create train test data",
                 "split data into train and test" ]
    store_result = [ t.StoreName(question="Where to store training data?"),
                     t.StoreName(question="Where to store testing data?") ]
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
        train_data = t.IrisData(x_train, y_train)
        test_data = t.IrisData(x_test, y_test)
        return train_data, test_data

trainTestSplit = TrainTestSplit()

class TrainModel(IrisCommand):
    title = "train {model} on {data}"
    examples = [ "train {model} {data}",
                 "train model {model} on data {data}" ]
    help_text = [
        "This command trains a model on a dataset. The model will be fit on the provided data."
    ]
    def command(self, iris_model : t.Any(), iris_data : t.Any()):
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
        select_classifier = t.Select(metrics, default="binary")
        self.argument_types = { "iris_model": t.Any(),
                                "iris_data": t.Any(),
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
    def __init__(self):
        metrics = { "Accuracy: correct predictions / incorrect predictions": "accuracy",
                    "F1 macro: f1 score computed with average across classes": "f1_macro",
                    "F1 micro: f1 score computed with weighted average": "f1_micro" }
        select_metric = t.Select(metrics, default="accuracy")
        self.argument_types = { "model": t.Any(),
                                "score": select_metric,
                                "n": t.Int() }
        super().__init__()
    def command(self, model, score, n):
        from sklearn.cross_validation import cross_val_score
        return cross_val_score(model.model, model.X, model.y, scoring = score, cv=n)
    def explanation(self, score):
        import numpy as np
        score = round(np.average(score), 4)
        return "Average performance of {} across the folds".format(score)

crossValidateModel = CrossValidateModel()

class ComputeAUC(IrisCommand):
    title = "compute auc curve data for {model}"
    examples = [ "auc curve {model}",
                 "auc data for {model}" ]
    help_text = [
        "This command will compute the necessary data to plot an AUROC curve for a model."
    ]
    store_result = t.StoreName(question="Where do you want to save the auc data?")
    def command(self, model : t.Any()):
        from sklearn.metrics import roc_curve, auc
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import label_binarize
        from scipy import interp
        classes = set(model.y)
        n_classes = len(classes)
        X_train, X_test, y_train, y_test = train_test_split(model.X, model.y, test_size=0.1, random_state=0)
        y_score = model.model.fit(X_train, y_train).decision_function(X_test)
        fpr = {}
        tpr = {}
        roc_auc = {}
        binary_ytest = label_binarize(y_test, classes=list(classes))
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(binary_ytest[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(binary_ytest.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        package_data = {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc, "n_classes": n_classes}
        return package_data
    def explanation(self, data):
        return "Computed the auc curve data."

computeAUC = ComputeAUC()

class PlotAUCFromData(IrisCommand):
    title = "plot auc curve from {data}"
    examples = [ "plot auc data {data}",
                 "plot {data} auc" ]
    argument_types = { "data": t.Any(question="Where is the auc curve data?") }
    store_result = t.StoreName(question="What would you like to name the plot?")
    help_text = [
        "This command takes pre-computed AUROC data and makes a plot."
    ]
    def command(self, data):
        import matplotlib
        matplotlib.use('AGG')
        import matplotlib.pyplot as plt
        fpr, tpr, roc_auc, n_classes = data["fpr"], data["tpr"], data["roc_auc"], data["n_classes"]
        # this is annoyingly magical, we want to pull the user-specified 'StoreName' to label the figure
        name = self.context["names"][0]
        # Plot all ROC curves
        f = plt.figure(name.id)
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i],
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        return t.IrisImage(f, name.name)

plotAUCFromData = PlotAUCFromData()

# NEW

class PlotAUC(IrisCommand):
    title = "plot auc curve for {model}"
    examples = [ "plot auc curve for model {model}" ]
    store_result = t.StoreName(question="What would you like to name the plot?")
    help_text = [
        "This command will plot an AUROC curve for a model.",
        "An AUROC curve shows the tradeoff between true positive and false positive rates for a model."
    ]
    def command(self, model : t.Any()):
        import matplotlib
        matplotlib.use('AGG')
        import matplotlib.pyplot as plt
        data = computeAUC(model)
        # thread through the 'name' from self
        return plotAUCFromData.with_context(self.context)(data)

plotAUC = PlotAUC()

class CompareModels(IrisCommand):
    title = "compare {model1} and {model2} using {metric}"
    examples = [ "compare {model1} {model2} using {metric}",
                 "which model is better under {metric}, {model1} or {model2}" ]
    def __init__(self):
        metrics = { "Accuracy: correct predictions / incorrect predictions": "accuracy",
                    "F1 macro: f1 score computed with average across classes": "f1_macro",
                    "F1 micro: f1 score computed with weighted average": "f1_micro" }
        select_metric = t.Select(metrics, default="accuracy")
        self.argument_types = { "model1": t.Any(),
                                "model2": t.Any(),
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

class PlotHistogram(IrisCommand):
    title = "plot a histogram on {data}"
    examples = [ "plot histogram {data}",
                 "histogram {data}" ]
    store_result = t.StoreName(question="Where would you like to save the plot?")
    help_text = [
        "This command plots a histogram on the provided data.",
        "A histogram counts the number of datapoints that hold certain values."
    ]
    def command(self, data : t.Any()):
        import matplotlib
        matplotlib.use('AGG')
        import matplotlib.pyplot as plt
        name = self.context["names"][0]
        f = plt.figure(name.id)
        plt.hist(data)
        return t.IrisImage(f, name.name)

plotHistogram = PlotHistogram()

class FindRegularization(IrisCommand):
    title = "find the best l2 regularization parameter for {model} with {metric}"
    examples = [ "best regularization for {model} {metric}",
                 "best l2 parameter for {model} under {metric}" ]
    def __init__(self):
        metrics = { "Accuracy: correct predictions / incorrect predictions": "accuracy",
                    "F1 macro: f1 score computed with average across classes": "f1_macro",
                    "F1 micro: f1 score computed with weighted average": "f1_micro" }
        select_metric = t.Select(metrics, default="accuracy")
        self.argument_types = { "model": t.Any(),
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

class PrintValue(IrisCommand):
    title = "print {value}"
    examples = [ "display {value}", "{}"]
    help_text = [
        "This command will display the underlying data for environment variables."
    ]
    def command(self, value : t.Any()):
        return value

printValue = PrintValue()

# statistical tests

class StudentTTest(IrisCommand):
    title = "calculate two sample Student t-test on {x} and {y}"
    examples = [
        "Student t-test on {x} {y}",
        "statistical test",
        "two sample statistical test",
        "test statistic"
    ]
    help_text = [
        "This test determines whether two independent samples are significantly different from one another.",
        "It assumes that both samples are normally distributed with equal variance."
    ]
    def command(self, x : t.Array(), y : t.Array()):
        from scipy.stats import ttest_ind
        return ttest_ind(x,y)
    def explanation(self, results):
        pval = round(results[1], 4)
        if pval < 0.05:
            return "These distributions are significantly different, with p-value of {}.".format(pval)
        else:
            return "These distributions are not significantly different, with p-value of {}.".format(pval)

studentTTest = StudentTTest()

class WelchTTest(IrisCommand):
    title = "calculate Welch t-test on {x} and {y}"
    examples = [
        "Welch t-test on {x} and {y}",
        "statistical test",
        "two sample statistical test",
        "statistical"
    ]
    help_text = [
        "This test determines whether two independent samples are significantly different from one another.",
        "It assumes that both samples are normally distributed, but does not assume they have equal variance."
    ]
    def command(self, x : t.Array(), y : t.Array()):
        from scipy.stats import ttest_ind
        return ttest_ind(x,y, equal_var=False)
    def explanation(self, results):
        pval = round(results[1], 4)
        if pval < 0.05:
            return "These distributions are significantly different, with p-value of {}.".format(pval)
        else:
            return "These distributions are not significantly different, with p-value of {}.".format(pval)

welchTTest = WelchTTest()

class MannWhitney(IrisCommand):
    title = "calculate Mann-Whitney U test on {x} and {y}"
    examples = [
        "Mann-Whitney U test on {x} and {y}",
        "statistical test",
        "two sample statistical test"
    ]
    help_text = [
        "This test determines whether two independent samples are significantly different from one another.",
        "It does not assume that both samples are normally distributed."
    ]
    def command(self, x : t.Array(), y : t.Array()):
        from scipy.stats import mannwhitneyu
        return mannwhitneyu(x,y)
    def explanation(self, results):
        pval = round(results[1], 4)
        if pval < 0.05:
            return "These distributions are significantly different, with p-value of {}.".format(pval)
        else:
            return "These distributions are not significantly different, with p-value of {}.".format(pval)

mannWhitney = MannWhitney()

class TTestHelp(IrisCommand):
    title = "help me run a t-test"
    example = [
        "which t-test should I use?"
    ]
    help_text = [
        "This command walks you through choosing a t-test"
    ]
    argument_types = {
        "choice": dt.YesNo("Is your data normally distributed?",
            yes=dt.YesNo("Do your samples have equal variance?",
                yes="use student's t-test",
                no="use welch's t-test"
            ),
            no="use mann-whitney u test"
        )
    }
    def command(self, choice):
        return choice

tTestHelp = TTestHelp()

x_val = dt.Variable("X")
y_val = dt.Variable("Y")

class TestLoop(IrisCommand):
    title = "lets test the loop"
    examples = [ "test loop" ]
    argument_types = {
        "dummy": dt.Label("main_loop", dt.ListStates([
            dt.Assign(x_val, dt.Int("Choose the first int value")),
            dt.Print("Just printing something to show that I can!"),
            dt.Assign(y_val, dt.Int("Choose the second int value")),
            dt.PrintF(lambda: "The sum of your values is {}".format(x_val.get_value() + y_val.get_value())),
            dt.YesNo("Are you finished?",
                yes="Bye",
                no=dt.Jump("main_loop"))
        ]))
    }
    def command(self, dummy):
        return dummy

testLoop = TestLoop()

# x_val = dt.Variable("X")
# y_val = dt.Variable("Y")
#
# script = dt.SequentialMachine()
#
# script.add(dt.Assign(x_val, dt.Int("Please enter the first integer")))
#
# script.add(dt.Print("Thanks!"))
#
# script.add(dt.Assign(y_val, dt.Int("Please enter the second integer")))
#
# def confirm_print():
#     val1, val2 = x_val.get_value(), y_val.get_value()
#     return "Awesome, I got {} and {}".format(val1, val2)
#
# script.add(dt.PrintF(confirm_print))
#
# def add_numbers():
#     val1, val2 = x_val.get_value(), y_val.get_value()
#     return "Sum is {}".format(val1 + val2)
#
# script.add(dt.PrintF(add_numbers))
#
# script.add(dt.Print("See you later!"))
#
# machine = script.compile()
#
# class TestLoop2(IrisCommand):
#     title = "lets test the fancy loop"
#     examples = [ "fancy loop" ]
#     argument_types = {
#         "dummy": machine
#     }
#     def command(self, dummy):
#         return dummy
#
# testLoop = TestLoop2()

# class ChooseStatisticalTest(IrisQuestion):
#     title = "help me choose a statistical test"
#     examples = [
#         "i need to choose a statistical test",
#         "which statistical test should i pick"
#     ]
#     question_tree = t.YesNo("Is the data normally distributed? (If you're not sure, say 'no'.)")(
#         yes=t.YesNo("Do your samples have equal variance? (If you're not sure, say 'no'.)")(
#             yes=studentTTest,
#             no=welchTTest
#         )
#         no=mannWhitney
#     )
#
# chooseStatisticalTest = ChooseStatisticalTest()

# class TellMeAMetric(IrisQuestion):
#     title = "example for determining what metric to use"
#     question_tree = t.SelectQuestion("Which of the following metrics are you interested in", options={
#         "Accuracy: description of what accuracy is": "use the keyword accuracy",
#         "F1: description of what F1 is": t.SelectQuestion("What kind of F1?", options={
#             "Macro: description of what macro is": "use the keyword f1_macro",
#             "Micro: description of what micro is": "use the keyword f1_micro"
#         })
#     })
#
# tellMeAMetric = TellMeAMetric()

# student's t-test (normal, independent, equal variance)
# welch's t-test (normal, independent, not equal variance)
# Mann-Whitney U test (independent)


# @iris.register("list features")
# def list_features():
#     return iris.env.keys()
#
# @iris.register("find predictive value of {feature}")
# def get_predictive_value(feature : String()):
#     model = iris.env["data_model"]
#     x = iris.env["features"]
#     y = iris.env["classes"]
#     feature_table = iris.env["feature-table"]['X']
#     f2i = {i:f for f,i in feature_table.items()}
#     model.fit(x,y)
#     return model.coef_[0][f2i[feature]]
#
# @iris.register("predictive power of all features")
# def all_features():
#     model = iris.env["data_model"]
#     x = iris.env["features"]
#     y = iris.env["classes"]
#     features = list(iris.env["feature-table"]['X'].values())
#     feature_table = iris.env["feature-table"]['X']
#     f2i = {i:f for f,i in feature_table.items()}
#     model.fit(x,y)
#     return "\n".join(["{} of {}".format(f,model.coef_[0][f2i[f]]) for f in features])
#
# # so here we add a new named variable to enviornment context that
# # holds the result
# @iris.register("add {n1:Int} and {n2:Int} to var")
# def add_named(n1 : Int(), n2 : Int()):
#     return IrisValue(n1+n2, name="n1_and_n2")
#
# # demonstrate lookup of variable from environment
# @iris.register("sum {lst}")
# def sum1(lst : List()):
#     return sum(lst)
#
# @iris.register("count {lst}")
# def count1(lst : Any()):
#     counts = defaultdict(int)
#     for x in lst:
#         counts[x] += 1
#     return counts
#
# @iris.register("make indicator for {lst}")
# def make_indicator(lst : Any()):
#     keys = set(lst)
#     index2key = {i:k for i,k in enumerate(keys)}
#     key2index = {k:i for i,k in index2key.items()}
#     return [key2index[x] for x in lst]
#
# @iris.register("what vars")
# def what_vars():
#     return iris.env.keys()
#
# @iris.register("last values")
# def last_values():
#     return iris.env["results"]
#
# @iris.register("program enviornment")
# def env():
#     return iris.env
#
# @iris.register("print data", examples=["print data {x}", "{x}"])
# def info(x : Any()):
#     return x
#
# @iris.register("list commands")
# def list_cmds():
#     for k in iris.mappings.keys():
#         print(k)
#
# iris.train_model()
