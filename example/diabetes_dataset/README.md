This dataset was pulled from `scikit-learn` via the following code:

```py
import json

from sklearn import datasets

X, y = datasets.load_diabetes(return_X_y=True)
export_json = {"Features": X.tolist(), "Label": y.tolist()}
with open("diabetes_dataset.json", "w") as f:
    f.write(json.dumps(export_json))
```

This follows the schema presented in `tinymlkit`