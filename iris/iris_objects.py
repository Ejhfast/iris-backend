import numpy as np
import io
import base64

class IrisValue:
    def __init__(self, value, name=None):
        self.value = value
        self.name = name

class IrisValues(IrisValue):
    def __init__(self, values, names):
        self.values = values
        self.names = names

class IrisId(IrisValue):
    def __init__(self, value, id, name=None):
        self.value = value
        self.id = id
        if not name:
            self.name = value
        else:
            self.name = name

class IrisImage(IrisId):
    type="Image"
    def __init__(self, plt, name):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.value = base64.b64encode(buf.read()).decode('utf-8')
        self.name = name

class IrisModel(IrisValue):
    type="Model"
    def __init__(self, model, X, y, name=None):
        self.X = X
        self.y = y
        self.model = model
        self.name = name

class IrisData(IrisValue):
    type="Data"
    def __init__(self, xvals, yvals):
        self.X = xvals
        self.y = yvals
