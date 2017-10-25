from pyspark import since, keyword_only
from pyspark.rdd import ignore_unicode_prefix
from pyspark.ml.linalg import _convert_to_vector
from pyspark.ml.param.shared import *
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaTransformer, _jvm
from pyspark.ml.common import inherit_doc

class ColumnPruner(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable):

    keep = Param(Params._dummy(), "keep", "keep the specified columns in the frame")

    columns = Param(Params._dummy(), "columns", "specified columns")

    @keyword_only
    def __init__(self, keep=False, columns=[]):
       super(ColumnPruner, self).__init__()
       self._java_obj = self._new_java_obj("org.apache.spark.ml.h2o.features.ColumnPruner", self.uid)
       self._setDefault(keep=False, columns=[])
       kwargs = self._input_kwargs
       self.setParams(**kwargs)

    @keyword_only
    def setParams(self, keep=False, columns=[]):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setKeep(self, value):
        return self._set(keep=value)

    def setColumns(self, value):
        return self._set(columns=value)

    def getKeep(self):
        self.getOrDefault(self.keep)

    def getColumns(self):
        self.getOrDefault(self.columns)