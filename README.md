==============================
GMR
==============================

Gaussian Model Regression class extended from scikit-learn. 

Extends the GMM class to include a predict_gmr method. This method uses gaussian model regression to predict a Y value for each independent variable X along with a covarance estimate. The method can be summed up as follows:

1. Each gaussian model component provides an estimate for Y
2. Each estimate is weighted by the likelihood that the component was the correct model representation.
3. Eeach component provides a covariance estimate as is similarly weighted
4. Weighted estimates are summed and retured.

Pros:
- Provides a probability for each estimated Y

Cons:
- Computationally expensive for large numbers of gaussian model components. This can be reduced without much penalty in most cases by only considering only the k-closest models (by their mean).


