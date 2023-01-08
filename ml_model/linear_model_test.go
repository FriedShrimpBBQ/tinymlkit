package ml_model

import (
	"ml_schemas"
	"testing"
)

func TestLinearModelMapper(t *testing.T) {
	sampleLinearModel := &ml_schemas.LinearModel{}
	sampleLinearModel.UnmarshalJSON([]byte(`{"Coefficient": [1,2,3,4], "Intercept": 5, "Penalty": "l2", "Alpha": 6, "L1Ratio": 7}`))

	x := LinearModelMapper(*sampleLinearModel)
	if x[0] != sampleLinearModel.Intercept {
		t.Fatalf("intercept should 0th element")
	}
	if x[1] != sampleLinearModel.Coefficient[0] {
		t.Fatalf("intercept should 0th element")
	}
}

func TestLinearModelPrediction(t *testing.T) {
	sampleLinearModel := &ml_schemas.LinearModel{}
	sampleLinearModel.UnmarshalJSON([]byte(`{"Coefficient": [1,2,3,4], "Intercept": 5, "Penalty": "l2", "Alpha": 6, "L1Ratio": 7}`))

	dataset := [][]float64{
		{1.0, 1.0, 1.0, 1.0},
	}
	output := LinearModel{*sampleLinearModel}.Predict(ml_schemas.Dataset{dataset})
	expected := 15.0 // 1+2+3+4+5
	if output[0] != expected {
		t.Fatalf("Incorrect inference")
	}
}

func TestLinearModelLoss(t *testing.T) {
	sampleLinearModel := &ml_schemas.LinearModel{}
	sampleLinearModel.UnmarshalJSON([]byte(`{"Coefficient": [1,2,3,4], "Intercept": 5, "Penalty": "l2", "Alpha": 0.0, "L1Ratio": 1.0}`))

	dataset := [][]float64{
		{1.0, 1.0, 1.0, 1.0},
	}
	label := []float64{14.0}

	lm := LinearModel{*sampleLinearModel}
	loss := lm.MSELoss(ml_schemas.LabelledDataset{dataset, label})
	expected := 1.0
	if loss != expected {
		t.Fatalf("Incorrect mse loss value")
	}

}

func TestOptimFuncGenerator(t *testing.T) {
	sampleLinearModel := &ml_schemas.LinearModel{}
	sampleLinearModel.UnmarshalJSON([]byte(`{"Coefficient": [1,2,3,4], "Intercept": 5, "Penalty": "l2", "Alpha": 0.0, "L1Ratio": 1.0}`))

	dataset := [][]float64{
		{1.0, 1.0, 1.0, 1.0},
	}
	label := []float64{14.0}

	lm := LinearModel{*sampleLinearModel}
	labelledDataset := ml_schemas.LabelledDataset{dataset, label}

	x0 := LinearModelMSEObjective(lm.lm, labelledDataset)([]float64{1.0, 2.0, 3.0, 4.0})
	x1 := LinearModelMSEObjective(lm.lm, labelledDataset)([]float64{4.0, 3.0, 2.0, 1.0})

	if x0 != x1 {
		t.Fatalf("Incorrect mse loss value when using mse objective")
	}

}
