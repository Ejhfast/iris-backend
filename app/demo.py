import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from iris import Iris, IrisImage, IrisValue, Int, IrisType, Any, List, String, ArgList, Name, IrisModel, Array, Select, IrisValues, IrisData
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from iris import primitives as iris_api
import fileinput
import numpy as np
import math
iris = Iris()


# MAGIC COMMANDS

@iris.register("save environment to {name}")
def save_env(name : String(question="What filename to save under?")):
    import pickle
    with open(name, 'wb') as f:
        pickle.dump({"env":iris.env, "env_order":iris.env_order}, f)
        return "Saved to {}.".format(name)

@iris.register("load environment from {name}")
def load_env(name : String(question="What filename to save under?")):
    import pickle
    with open(name, 'rb') as f:
        data = pickle.load(f)
        iris.env = data["env"]
        iris.env_order = data["env_order"]
        return "Loaded environment from \"{}\".".format(name)

# OPTION 1
examples = [ "add {n1} and {n2}",
             "add {n1} {n2}" ]

@iris.register("add two numbers", examples=examples)
def add(n1 : Int(), n2 : Int()):
    return n1+n2

# OPTION 2
# class AddTwoNumbers(IrisCommand):
#     title = "add two numbers"
#     examples = [ "add {n1} and {n2}",
#                  "add {n1} {n2}" ]
#     def command(n1 : Int(), n2: Int()):
#         return n1+n2

# ADD AND STORE
examples = [ "add store {n1} and {n2}",
                   "add store {n1} {n2}" ]

@iris.register("add and store two numbers", examples=examples)
def add_store(n1 : Int(), n2 : Int(), name : Name()):
    return IrisValue(n1+n2, name=name.name)

# PEARSON CORRELATION

# OPTION 1
examples = [ "pearson correlation between {x} and {y}",
             "pearson correlation {x} {y}" ]

explain = lambda result: "Correlation of {} with p-value of {}".format(round(result[0],4), round(result[1],4))

@iris.register("pearson correlation", examples=examples, format_out=explain)
def pearsonr(x : Array(), y : Array()):
    from scipy.stats import pearsonr
    return pearsonr(x,y)

# OPTION 2

# class PearsonCorrelation(IrisCommand):
#
#     title = "pearson correlation: {x} and {y}"
#     examples = [ "pearson correlation between {x} and {y}",
#                  "pearson correlation {x} {y}" ]
#
#     def command(x : Array(), y : Array()):
#         from scipy.stats import pearsonr
#         return pearsonr(x,y)
#
#     def explanation(result):
#         corr, pval = round(result[0],4), round(result[1],4)
#         return "Correlation of {} with p-value of {}".format(corr, pval)

# CLASSIFICATION MODEL

examples = [ "build a new classification model",
             "make a new classification model" ]

@iris.register("new classification model")
def make_model(x_features : ArgList(), y_classes : ArgList(), name : Name()):
    model = LogisticRegression()
    X = np.array(x_features).T
    y = np.array(y_classes).T
    y = y.reshape(y.shape[0])
    model.fit(X,y)
    return IrisModel(model, X, y, name=name.value)

# MAKE TEST/TRAIN SPLITS

# OPTION 1

q_txt = "What would you like to name the {} data?"
@iris.register("create training and test data splits")
def split_data(x_features : ArgList(), y_classes : ArgList(), train_name : Name(question=q_txt.format("training")), test_name : Name(question=q_txt.format("test"))):
    from sklearn.model_selection import train_test_split
    xvals = np.array(x_features).T
    yvals = np.array(y_classes).T
    yvals = yvals.reshape(yvals.shape[0])
    x_train, x_test, y_train, y_test = train_test_split(xvals, yvals, train_size=0.25)
    train_data = IrisData(x_train, y_train)
    test_data = IrisData(x_test, y_test)
    return IrisValues(values=[train_data, test_data], names=[train_name.name, test_name.name])

# OPTION 2

# class SplitData(IrisCommand):
#     title = "create training and test data splits"
#     examples = [ "split data train test",
#                  "randomly select data for training and testing"]
#
#     argument_types = { "x_features": ArgList(),
#                        "y_feartures": ArgList()
#                        "training_name": Name(question="What would you like to name the training data")
#                        "testing_name": Name(question="What would you like to name the testing data") }
#
#     def command(x_features, y_features, training_name, testing_name):
#             from sklearn.model_selection import train_test_split
#             xvals = np.array(x_features).T
#             yvals = np.array(y_classes).T
#             yvals = yvals.reshape(yvals.shape[0])
#             x_train, x_test, y_train, y_test = train_test_split(xvals, yvals, train_size=0.25)
#             train_data = IrisData(x_train, y_train)
#             test_data = IrisData(x_test, y_test)
#             return train_data, train_name.name, test_data, test_name.name
#
#     def memory(*args):
#         train_data, train_name, test_data, test_name = args
#         return IrisValues(values=[train_data, test_data], names=[train_name.name, test_name.name])


# evaluate model on train test_name

@iris.register("train {model} on {data}")
def train_model(model : Any(), data : Any()):
    model.model.fit(data.X, data.y)
    return "I fit the model"

@iris.register("evaluate {model} on {data}")
def evaluate_model(model : Any(), data : Any()):
    from sklearn.metrics import f1_score
    test_y = data.y
    pred_y = model.model.predict(data.X)
    # need to support multiple metrics
    score = f1_score(test_y, pred_y, average="macro")
    return "F1 score of {}".format(score)

# Cross-validate

classifier_scores = { "Accuracy, correct predictions / incorrect predictions": "accuracy",
                      "F1 macro, f1 score computed with average across classes": "f1_macro",
                      "F1 micro, f1 score computed with weighted average": "f1_micro" }

select_classifier = Select(classifier_scores, default="accuracy")

explain = lambda scores: "Average performance of {} across the folds".format(round(np.average(scores),4))

@iris.register("cross-validate {model} with {score} and {n} folds", format_out=explain)
def cross_validate_model(model : Any(), score : select_classifier, n : Int()):
    from sklearn.cross_validation import cross_val_score
    return cross_val_score(model.model, model.X, model.y, scoring = score, cv=n)

@iris.register("compute auc curve data for {model}")
def compute_auc(model : Any(), name : Name()):
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
    # are we in the iris environment, or calling this function in code?
    if isinstance(name, IrisValue):
        return IrisValue(package_data, name=name.name)
    else:
        return package_data

@iris.register("plot auc curve from {data}")
def plot_auc_data(data : Any(question="Where is the auc plot data?"), name : Name()):
    import matplotlib
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt
    fpr, tpr, roc_auc, n_classes = data["fpr"], data["tpr"], data["roc_auc"], data["n_classes"]
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
    return iris_api.send_plot(f, name)

@iris.register("plot auc curve for {model}")
def plot_auc(model : Any(), name : Name()):
    import matplotlib
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt
    data = compute_auc(model, None)
    return plot_auc_data(data, name)

# compare models

@iris.register("compare {model1} and {model2}")
def make_model(model1 : Any(), model2 : Any()):
    import numpy as np
    m1_scores = np.average(cross_validate_model(model1, "f1_macro", 10))
    m2_scores = np.average(cross_validate_model(model2, "f1_macro", 10))
    if m1_scores > m2_scores:
        higher_m, lower_m = model1, model2
        higher_s, lower_s = m1_scores, m2_scores
    else:
        higher_m, lower_m = model2, model1
        higher_s, lower_s = m2_scores, m1_scores
    return "I'd say \"{}\" is better than \"{}\", with {} vs. {} f1_macro".format(higher_m.name, lower_m.name, higher_s, lower_s)


@iris.register("plot a histogram on {data}")
def plot_histogram(data : Any(), name : Name()):
    import matplotlib
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt
    f = plt.figure(name.id)
    plt.hist(data)
    return iris_api.send_plot(f, name)

select_classifier = Select(classifier_scores, default="accuracy")

explain = lambda x: "The best l2 score is {} with cross-validation performance of {} {}".format(x[0], round(x[1],4), x[2])

@iris.register("find the best regularization parameter for {model} with {score}", format_out=explain)
def find_regularize(model : Any(), scoring : select_classifier):
    from sklearn.cross_validation import cross_val_score
    import numpy as np
    best_score = 0
    best_c = None
    for c in [0.01, 0.1, 1, 10, 100]:
        score = np.average(cross_val_score(model.model, model.X, model.y, scoring = scoring, cv=5))
        if score > best_score:
            best_score = score
            best_c = c
    return best_c, best_score, scoring

@iris.register("list features")
def list_features():
    return iris.env.keys()

@iris.register("find predictive value of {feature}")
def get_predictive_value(feature : String()):
    model = iris.env["data_model"]
    x = iris.env["features"]
    y = iris.env["classes"]
    feature_table = iris.env["feature-table"]['X']
    f2i = {i:f for f,i in feature_table.items()}
    model.fit(x,y)
    return model.coef_[0][f2i[feature]]

@iris.register("predictive power of all features")
def all_features():
    model = iris.env["data_model"]
    x = iris.env["features"]
    y = iris.env["classes"]
    features = list(iris.env["feature-table"]['X'].values())
    feature_table = iris.env["feature-table"]['X']
    f2i = {i:f for f,i in feature_table.items()}
    model.fit(x,y)
    return "\n".join(["{} of {}".format(f,model.coef_[0][f2i[f]]) for f in features])

# so here we add a new named variable to enviornment context that
# holds the result
@iris.register("add {n1:Int} and {n2:Int} to var")
def add_named(n1 : Int(), n2 : Int()):
    return IrisValue(n1+n2, name="n1_and_n2")

# demonstrate lookup of variable from environment
@iris.register("sum {lst}")
def sum1(lst : List()):
    return sum(lst)

@iris.register("count {lst}")
def count1(lst : Any()):
    counts = defaultdict(int)
    for x in lst:
        counts[x] += 1
    return counts

@iris.register("make indicator for {lst}")
def make_indicator(lst : Any()):
    keys = set(lst)
    index2key = {i:k for i,k in enumerate(keys)}
    key2index = {k:i for i,k in index2key.items()}
    return [key2index[x] for x in lst]

@iris.register("what vars")
def what_vars():
    return iris.env.keys()

@iris.register("last values")
def last_values():
    return iris.env["results"]

@iris.register("program enviornment")
def env():
    return iris.env

@iris.register("print data", examples=["print data {x}", "{x}"])
def info(x : Any()):
    return x

@iris.register("list commands")
def list_cmds():
    for k in iris.mappings.keys():
        print(k)

# iris.env["my_list"] = [1,2,3,4,5]
# iris.env["my_num"] = 3
# model = LogisticRegression()
# x_data = np.random.randint(0,10,size=(100,10))
# y_data = (np.sum(x_data,axis=1) > 40).astype(int)
# iris.env["data_model"] = model
# iris.env["features"] = x_data
# iris.env["classes"] = y_data


# data_cols = defaultdict(list)
# lookup = {}
# for i,line in enumerate(fileinput.input()):
#     if i == 0:
#         for j,col in enumerate(line.strip().split(",")):
#             lookup[j] = col
#     else:
#         for j,d in enumerate(line.strip().split(",")):
#             data_cols[lookup[j]].append(d)
#
# for k,vs in data_cols.items():
#     iris.env[k] = vs

iris.train_model()

# init_message = [{"text": "add 3 4", "origin": "user"}]
# state1 = iris.state_machine({ "state": "START", "messages":init_message})
# print(state1)
# state2 = iris.state_machine({"state": "CLASSIFICATION", "id": state1["id"], "messages":init_message+[{"text": "yes", "origin": "user"}]})
# print(state2)

# iris.env_loop()
