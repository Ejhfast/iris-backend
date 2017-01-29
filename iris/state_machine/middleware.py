from .basic import Jump

class Middleware:
    def test(self, text):
        return True
    def transform(self):
        pass
    def hint(self):
        return []

class QuitMiddleware(Middleware):
    def test(self, text):
        if text:
            return "quit" in text
        return False
    def transform(self, caller, state, text):
        state.clear_error()
        return Jump("START")

class ExplainMiddleware(Middleware):
    def __init__(self, gen_state):
        self.gen_state = gen_state
    def test(self, text):
        if text:
            return any([x in text for x in ["explain", "help"]])
        return False
    def transform(self, caller, state, text):
        state.clear_error()
        return self.gen_state(caller)
