import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from iris import IRIS, IrisCommand
from iris import iris_types as t
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import fileinput
import numpy as np
import math

class SaveEnv(IrisCommand):
    title = "save environment to {name}"
    examples = [ "save environment {name}",
                 "save env to {name}" ]
    def command(self, name : t.String(question="What filename to save under?")):
        import pickle
        with open(name, 'wb') as f:
            pickle.dump({"env":self.iris.env, "env_order":IRIS.env_order}, f)
            return "Saved to {}.".format(name)

saveEnv = SaveEnv()

class LoadEnv(IrisCommand):
    title = "load environment from {name}"
    examples = [ "load environment {name}",
                 "load env from {name}" ]
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
                 "add {x} and {y}" ]
    def command(self, x : t.Int(), y : t.Int()):
        return x + y

addTwoNumbers = AddTwoNumbers()

class AddAndStoreTwoNumbers(IrisCommand):
    title = "add and store two numbers: {x} and {y}"
    examples = [ "add and store {x} and {y}",
                 "add and store {x} {y}" ]
    store_result = t.StoreName(question="Where would you like to store x+y?")
    def command(self, x : t.Int(), y : t.Int()):
        return x + y

addAndStoreTwoNumbers = AddAndStoreTwoNumbers()

class PearsonCorrelation(IrisCommand):
    title = "pearson correlation: {x} and {y}"
    examples = [ "pearson correlation between {x} and {y}",
                 "pearson correlation {x} {y}" ]
    def command(self, x : t.Array(), y : t.Array()):
        from scipy.stats import pearsonr
        return pearsonr(x,y)
    def explanation(corr_pval):
        corr = round(corr_pval[0],4)
        pval = round(corr_pval[1],4)
        return "Correlation of {} with p-value of {}".format(corr, pval)

pearsonCorrelation = PearsonCorrelation()

class StorePearsonCorrelation(PearsonCorrelation):
    title = "store pearson correlation: {x} and {y}"
    examples = [ "store pearson {x} {y}" ]
    store_result = [ t.StoreName(question="Where to store t-statistic?"),
                     t.StoreName(question="Where to store the pval?") ]

storePearson = StorePearsonCorrelation()

class MakeModel(IrisCommand):
    title = "new classification model"
    examples = [ "build a new classification model",
                 "make a new classification model" ]
    argument_types = { "x_features": t.ArgList(),
                       "y_classes": t.ArgList() }
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

makeModel = MakeModel()

class TrainTestSplit(IrisCommand):
    title = "create training and test data splits"
    examples = [ "create train test data",
                 "split data into train and test" ]
    store_result = [ t.StoreName(question="Where to store training data?"),
                     t.StoreName(question="Where to store testing data?") ]
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
    def command(self, iris_model : t.Any(), iris_data : t.Any()):
        iris_model.model.fit(iris_data.X, iris_data.y)
    def explanation(*args):
        return "I fit the model."

trainModel = TrainModel()

class TestModel(IrisCommand):
    title = "test {iris_model} on {iris_data} using f1 score with {weighting}"
    examples = [ "test {iris_model} {iris_data} with {weighting}",
                 "test model {iris_model} on data {data} with {weighting}" ]
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
    def explanation(score):
        score = round(score, 4)
        return "F1 score of {}".format(score)

testModel = TestModel()

class CrossValidateModel(IrisCommand):
    title = "cross-validate {model} with {score} and {n} folds"
    examples = [ "cross-validate {model} {score} {n}" ]
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
    def explanation(score):
        import numpy as np
        score = round(np.average(score), 4)
        return "Average performance of {} across the folds".format(score)

crossValidateModel = CrossValidateModel()

class ComputeAUC(IrisCommand):
    title = "compute auc curve data for {model}"
    examples = [ "auc curve {model}",
                 "auc data for {model}" ]
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
    def explanation(data):
        return "Computed the auc curve data."

computeAUC = ComputeAUC()

class PlotAUCFromData(IrisCommand):
    title = "plot auc curve from {data}"
    examples = [ "plot auc data {data}",
                 "plot {data} auc" ]
    argument_types = { "data": t.Any(question="Where is the auc curve data?") }
    store_result = t.StoreName(question="What would you like to name the plot?")
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
    def explanation(results):
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
    def explanation(results):
        best_c, best_score, metric = results
        return "Best L2 of {} with {} {}".format(best_c, best_score, metric)

findRegularization = FindRegularization()


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
