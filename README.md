# Feature Scaling

Feature Scaling | Example code and own notes while taking the course "Intro to Machine Learning" on Udacity.

## Definition
It is a method for rescaling features like height and weight.

![formula](resources/formula.png)

## Example
Assume that the old weights are like this `[115, 140, 175]`.

`X'` (for 140): new (rescaled) feature

Right side of the equation consists of info taken from old feature(s):

|Variable|Value|
|---|---|
|`X`|140|
|`Xmin`|115|
|`Xmax`|175|

And the scale result would be: `0.41`

Scale should be represented between 0 to 1 as we see above.

![scale](resources/scale.png)

## Important highlights

- One of the distadvantages is that if you have outliers in your input features then thet can kind of mess up your rescaling because your `Xmin` and `Xmax` might have really extreme values.
- SVM with RBF kernel and K-means clustering would be affected by feature rescaling, but decision tree or linear regression not.
- If there are huge difference on your data ad scale, rescaling is crucial.

## scikit-learn implementation
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(finance_features)
finance_features = scaler.transform(finance_features)

```