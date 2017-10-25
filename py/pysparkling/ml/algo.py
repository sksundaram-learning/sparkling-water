from pyspark import since, keyword_only
from pyspark.rdd import ignore_unicode_prefix
from pyspark.ml.linalg import _convert_to_vector
from pyspark.ml.param.shared import *
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaTransformer, _jvm
from pyspark.ml.common import inherit_doc
from pyspark import SparkContext
from pysparkling import *
class H2OGBM(JavaEstimator, JavaMLReadable, JavaMLWritable):

    featuresCols = Param(Params._dummy(), "featuresCols", "columns used as features")

    predictionsCol = Param(Params._dummy(), "predictionsCol", "label")

    @keyword_only
    def __init__(self, featuresCols=[], predictionsCol=None):
        super(H2OGBM, self).__init__()
        self._java_obj = self._new_java_obj("org.apache.spark.ml.h2o.algos.H2OGBM",
                                            self.uid,
                                            H2OContext.getOrCreate(SparkContext._active_spark_context)._jhc.h2oContext(),
                                            H2OContext.getOrCreate(SparkContext._active_spark_context)._jsql_context
                                            )
        self._setDefault(featuresCols=[], predictionsCol=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, featuresCols=[], predictionsCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setFeaturesCols(self, value):
        return self._set(featuresCols=value)

    def setPredictionsCol(self, value):
        return self._set(predictionsCol=value)

    def getFeaturesCols(self):
        self.getOrDefault(self.featuresCols)

    def getPredictionsCol(self):
        self.getOrDefault(self.predictionsCol)

    def _create_model(self, java_model):
        return H2OGBMModel(java_model)


class H2OGBMModel(JavaModel, JavaMLWritable, JavaMLReadable):

    @property
    @since("2.2.0")
    def coefficients(self):
        """
        Model coefficients of Linear SVM Classifier.
        """
        return self._call_java("coefficients")

    @property
    @since("2.2.0")
    def intercept(self):
        """
        Model intercept of Linear SVM Classifier.
        """
        return self._call_java("intercept")