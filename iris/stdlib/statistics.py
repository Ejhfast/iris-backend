from .. import IRIS, IrisCommand
from .. import state_types as t
from .. import state_machine as sm
from .. import util as util

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
    argument_help = {
        "x": "The x value should be an array from the current environment",
        "y": "The y value should be an array from the current environment",
    }
    def command(self, x : t.Array(), y : t.Array()):
        from scipy.stats import pearsonr
        return pearsonr(x,y)
    def explanation(self, corr_pval):
        corr = round(corr_pval[0],4)
        pval = round(corr_pval[1],4)
        return "Correlation of {} with p-value of {}".format(corr, pval)

pearsonCorrelation = PearsonCorrelation()

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
        "choice": t.YesNo("Is your data normally distributed?",
            yes=t.YesNo("Do your samples have equal variance?",
                yes="use student's t-test",
                no="use welch's t-test"
            ),
            no="use mann-whitney u test"
        )
    }
    def command(self, choice):
        return choice

tTestHelp = TTestHelp()
