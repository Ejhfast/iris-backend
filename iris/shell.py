import cmd

class IrisShell(cmd.Cmd):

    def __init__(self, main_loop):
        self.main_loop = main_loop
        super().__init__()

    intro = 'You are using the IrisML prototype.'
    prompt = '(iris)> '
    file = None

    def default(self, line):
        self.main_loop(line)
