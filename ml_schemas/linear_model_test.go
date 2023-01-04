package ml_schemas

import (
	"testing"
)

func TestLinearModelJson(t *testing.T) {
	sampleLinearModel := &LinearModel{}
	sampleLinearModel.UnmarshalJSON([]byte(`{"Coefficient": [1,2,3,4], "Intercept": 5, "Penalty": "l2", "Alpha": 6, "L1Ratio": 7}`))

	if sampleLinearModel.Intercept != 5.0 {
		t.Fatalf("sampleLinearModel.Intercept - got %f, wanted %f", sampleLinearModel.Intercept, 5.0)
	}
	if sampleLinearModel.Coefficient[0] != 1.0 {
		t.Fatalf("Failed to parse coefficients")
	}
}
