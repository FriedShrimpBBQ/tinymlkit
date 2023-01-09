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

Running this example is as simple as:

```sh
$ go run . -json_data diabetes_dataset.json -penalty elasticnet -max_iter 10000
```

Although its not packaged as proper idiomatic go, this is indeed a start!