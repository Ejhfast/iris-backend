from .. import IrisCommand
from .. import state_types as t
from .. import state_machine as sm
from .. import util as util
from .. import iris_objects

class ComputeAUC(IrisCommand):
    title = "compute auc curve data for {model}"
    examples = [ "auc curve {model}",
                 "auc data for {model}" ]
    help_text = [
        "This command will compute the necessary data to plot an AUROC curve for a model."
    ]
    store_result = t.VarName(question="Where do you want to save the auc data?")
    def command(self, model : t.EnvVar()):
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
    argument_types = { "data": t.EnvVar(question="Where is the auc curve data?") }
    store_result = t.VarName(question="What would you like to name the plot?")
    help_text = [
        "This command takes pre-computed AUROC data and makes a plot."
    ]
    def command(self, data):
        import matplotlib
        matplotlib.use('AGG')
        import matplotlib.pyplot as plt
        fpr, tpr, roc_auc, n_classes = data["fpr"], data["tpr"], data["roc_auc"], data["n_classes"]
        # this is annoyingly magical, we want to pull the user-specified 'VarName' to label the figure
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
        return iris_objects.IrisImage(f, name.name)

plotAUCFromData = PlotAUCFromData()

# NEW

class PlotAUC(IrisCommand):
    title = "plot auc curve for {model}"
    examples = [ "plot auc curve for model {model}" ]
    store_result = t.VarName(question="What would you like to name the plot?")
    help_text = [
        "This command will plot an AUROC curve for a model.",
        "An AUROC curve shows the tradeoff between true positive and false positive rates for a model."
    ]
    def command(self, model : t.EnvVar()):
        import matplotlib
        matplotlib.use('AGG')
        import matplotlib.pyplot as plt
        data = computeAUC(model)
        # thread through the 'name' from self
        return plotAUCFromData.with_context(self.context)(data)

plotAUC = PlotAUC()


class PlotHistogram(IrisCommand):
    title = "plot a histogram on {data}"
    examples = [ "plot histogram {data}",
                 "histogram {data}" ]
    help_text = [
        "This command plots a histogram on the provided data.",
        "A histogram counts the number of datapoints that hold certain values."
    ]
    argument_types = {
        "data": t.EnvVar("What would you like to plot?"),
        "name": t.String("Where would you like to save the plot?")
    }
    def command(self, data, name):
        import matplotlib
        matplotlib.use('AGG')
        import matplotlib.pyplot as plt
        f = plt.figure(self.iris.gen_plot_id(name))
        plt.hist(data)
        plot_data = iris_objects.IrisImage(f, name)
        self.iris.add_to_env(name, plot_data)
        return plot_data

plotHistogram = PlotHistogram()
