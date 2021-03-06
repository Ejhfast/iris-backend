class PearsonCorrelation(IrisCommand):
    title = "compute pearson correlation: {x} and {y}"
    examples = [ "pearson correlation between {x} and {y}",
                 "pearson correlation {x} {y}",
                 "how are {x} and {y} correlated" ]
    help_text = [ "..." ]
    argument_types = {
        "x": Array("Where is the first array to analyze?"),
        "y": Array("Where is the second array?"),
    }
    def command(self, x, y):
        from scipy.stats import pearsonr
        return pearsonr(x,y)
    def explanation(self, corr_pval):
        corr = round(corr_pval[0],4)
        pval = round(corr_pval[1],4)
        return "Correlation of {} with p-value of {}".format(corr, pval)

pearsonCorrelation = PearsonCorrelation()

class GetName(StateMachine):
    output = "Hi, what is your name?"
    accepts_input = True
    def on_input(self, user_input):
        self.write("NAME", user_input)

class SayHello(StateMachine):
    def __init__(self):
        self.output = "Hello {}!".format(self.read("NAME"))

conversation = Loop(Sequence([
    GetName(),
    SayHello()
]))
conversation.run()
