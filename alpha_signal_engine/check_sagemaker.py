import sagemaker

print("sagemaker version:", sagemaker.__version__)

from sagemaker.sklearn.estimator import SKLearn

print("SKLearn import OK")
