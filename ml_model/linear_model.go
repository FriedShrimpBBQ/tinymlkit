package ml_model

// map model setup to something compatible with optim schema

import (
	"ml_schemas"
)

type LinearModel struct {
	lm ml_schemas.LinearModel
}

func (linear_model LinearModel) Predict(dataset ml_schemas.Dataset) []float64 {
	predictions := []float64{}
	for i := 0; i < len(dataset.Features); i++ {
		var prediction float64 = linear_model.lm.Intercept
		for j := 0; j < len(linear_model.lm.Coefficient); j++ {
			prediction = prediction + linear_model.lm.Coefficient[j]*dataset.Features[i][j]
		}
		predictions = append(predictions, prediction)
	}
	return predictions
}

func LinearModelMapper(linearModelObj *ml_schemas.LinearModel) []float64 {
	x := []float64{}
	x = append(x, linearModelObj.Intercept)
	for i := 0; i < len(linearModelObj.Coefficient); i++ {
		x = append(x, linearModelObj.Coefficient[i])
	}
	return x
}
