import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from functools import wraps
import inspect
from functools import wraps
def saveLast(func):
  @wraps(func)
  def with_logging(self,*args, **kwargs):
      self.lastlast=self.last
      self.lastlastlogs=self.lastlogs
      self.last=self.data
      self.lastlogs=self.logs

      rep=func(self,*args, **kwargs)
      return rep
  return with_logging
  
def addCustomFunc2(self,func):
  @wraps(func)
  def with_logging(*args, **kwargs):
      self.lastlast=self.last
      self.lastlastlogs=self.lastlogs
      self.last=self.data
      self.lastlogs=self.logs

      rep=func(self,*args, **kwargs)
      argss= inspect.getcallargs(func,self, *args, **kwargs)
      del argss["self"]
      argss=["{}={}".format(i,"\""+j+"\"" if isinstance(j,str) else j) for i,j in argss.items()]
      self._log( "self.{}({})".format( func.__name__, ", ".join(argss) ) )
      return rep
  return with_logging

class Dora:
  CUSTOMS={}
  
  @classmethod
  def addCustomFunction(cls,func,fn=None):
    fn = func.__name__ if fn is None else fn
    cls.CUSTOMS[fn]=func
    
  def __init__(self, data = None, output = None):
    self.init(data = data, output = output)

  def init(self,data,output):
    self.snapshots = {}
    self.logs = []
    self.last=None
    self.lastlogs=None
    self.lastlast=None
    self.lastlastlogs=None

    self.configure(data = data, output = output)

  def configure(self, data = None, output = None):
    if (type(output) is str or type(output) is int):
      self.output = output
    if (type(data) is str):
      self.initial_data = pd.read_csv(data)
      self.data = self.initial_data.copy()
      self.logs = []
    if (type(data) is pd.DataFrame):
      self.initial_data = data
      self.data = self.initial_data.copy()
      self.logs = []

  @saveLast
  def remove_feature(self, feature_name):
    del self.data[feature_name]
    self._log("self.remove_feature('{0}')".format(feature_name))

  @saveLast
  def extract_feature(self, old_feat, new_feat, mapper):
    new_feature_column = map(mapper, self.data[old_feat])
    self.data[new_feat] = list(new_feature_column)
    self._log("self.extract_feature({0}, {1}, {2})".format(old_feat, new_feat, mapper))

  @saveLast
  def impute_missing_values(self):
    column_names = self.input_columns()
    imp = preprocessing.Imputer()
    imp.fit(self.data[column_names])
    self.data[column_names] = imp.transform(self.data[column_names])
    self._log("self.impute_missing_values()")

  @saveLast
  def scale_input_values(self):
    column_names = self.input_columns()
    self.data[column_names] = preprocessing.scale(self.data[column_names])
    self._log("self.scale_input_values()")

  @saveLast
  def extract_ordinal_feature(self, feature_name):
    feature = self.data[feature_name]
    feature_dictionaries = map(
      lambda x: { str(feature_name): str(x) },
      feature
    )
    vec = DictVectorizer()
    one_hot_matrix = vec.fit_transform(feature_dictionaries).toarray()
    one_hot_matrix = pd.DataFrame(one_hot_matrix)
    one_hot_matrix.columns = vec.get_feature_names()
    self.data = pd.concat(
      [
        self.data,
        one_hot_matrix
      ],
      axis = 1
    )
    del self.data[feature_name]
    self._log("self.extract_ordinal_feature('{0}')".format(feature_name))

  def set_training_and_validation(self):
    training_rows = np.random.rand(len(self.data)) < 0.8
    self.training_data = self.data[training_rows]
    self.validation_data = self.data[~training_rows]

  def plot_feature(self, feature_name):
    x = self.data[feature_name]
    y = self.data[self.output]
    fit = np.polyfit(x, y, deg = 1)
    fig, ax = plt.subplots()
    ax.plot(x, fit[1] + fit[0] * x)
    ax.scatter(x, y)
    ax.set_title("{0} vs. {1}".format(feature_name, self.output))
    fig.show()

  def input_columns(self):
    column_names = list(self.data.columns)
    column_names.remove(self.output)
    return column_names

  def explore(self):
    features = self.input_columns()
    row_count = math.floor(math.sqrt(len(features)))
    col_count = math.ceil(len(features) / row_count)
    figure = plt.figure(1)

    for index, feature in enumerate(features):
      figure.add_subplot(row_count, col_count, index + 1)
      x = self.data[feature]
      y = self.data[self.output]
      fit = np.polyfit(x, y, deg = 1)
      plt.plot(x, fit[0] * x + fit[1])
      plt.scatter(x, y)
      plt.title("{0} vs. {1}".format(feature, self.output))
    plt.show()

  def snapshot(self, name):
    snapshot = {
      "data": self.data.copy(),
      "logs": self.logs.copy()
    }
    self.snapshots[name] = snapshot

  def use_snapshot(self, name):
    self.data = self.snapshots[name]["data"]
    self.logs = self.snapshots[name]["logs"]

  def _log(self, string):
    self.logs.append(string)

  def __getattr__(self,g):
    if g in self.CUSTOMS:
      return addCustomFunc2(self,self.CUSTOMS[g])
    return object.__getattribute__(self,g)
  
  def _ipython_display_(self, **kwargs):
    from IPython.display import HTML,display
    display(HTML(self.data._repr_html_()))

  def back_initial_data(self):
    self.init(self.initial_data,self.output)

  def back_one(self):
    self.data=self.last
    self.logs=self.lastlogs
    self.last=self.lastlast
    self.lastlogs=self.lastlastlogs
    # self.snapshots=self.lastsnapshots

