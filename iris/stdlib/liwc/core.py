from ... import IrisCommand
from ... import state_types as t
from ... import state_machine as sm
from ... import util as util
from ... import iris_objects

import dill
from collections import defaultdict

data = dill.load(open("../iris/stdlib/liwc/liwc_data/liwc-data.pkl","rb"))

cat2word = data["cat2word"]
word2cat = data["word2cat"]

liwc_keys = sorted(list(cat2word.keys()))

def order_liwc(d,keys=None):
  if not keys:
    keys = d.keys()
  order_liwc.s_keys = sorted(keys)
  return [d[k] for k in order_liwc.s_keys]

def analyze(doc,normalize=False,lex=word2cat,keys=liwc_keys):
  cats = defaultdict(float)
  words = 0.0
  for w in doc.lower().split():
    for c in lex[w]:
      cats[c] += 1.0
    words += 1.0
  if normalize:
    for k in keys:
      cats[k] = cats[k] / words
  return cats

class LiwcAnalysis(IrisCommand):
    title = "run liwc analysis on {documents}"
    examples = [ "liwc {documents}" ]
    argument_types = {
        "documents": t.Array("Where is the collection of documents?")
    }
    def command(self, documents):
        import numpy as np
        data = np.array([order_liwc(analyze(doc, normalize=True), liwc_keys) for doc in documents])
        liwc_types = ["Number" for _ in order_liwc.s_keys]
        print("LIWC", data.shape, len(order_liwc.s_keys))
        return iris_objects.IrisDataframe(column_names=order_liwc.s_keys, column_types=liwc_types, data=data, do_conversion=False)

liwcAnalysis = LiwcAnalysis()
