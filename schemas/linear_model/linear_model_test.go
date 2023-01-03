package linear_model_schema


import (
"testing"
)

func TestLinearModelJson(t *testing.T) {
	sampleLinearModel := &LinearModel{}
	sampleLinearModel.UnmarshalJSON([]byte(`{"Coefficient": [1,2,3], "Intercept": 4, "Simplex": [[5,6,7]], "SimplexLabels": [8], "Options": {"Penalty": "l2", "Alpha": 0.0001, "L1Ratio": 0.15}}`))
	if sampleLinearModel.Intercept != 4.0 {
        t.Fatalf("sampleLinearModel.Intercept - got %f, wanted %f", sampleLinearModel.Intercept, 4.0)
	}
	if sampleLinearModel.Options.Penalty != "l2" {
        t.Fatalf("sampleLinearModel.Options.Penalty - got %s, wanted %s", sampleLinearModel.Options.Penalty, "l2")
    }
}