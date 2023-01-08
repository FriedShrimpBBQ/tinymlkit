package ml_model

// map model setup to something compatible with optim schema

import (
	"math"
	"ml_schemas"
	"optim"
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

func (linear_model LinearModel) MSELoss(dataset ml_schemas.LabelledDataset) float64 {
	var score float64 = 0.0
	for i := 0; i < len(dataset.Features); i++ {
		var prediction float64 = linear_model.lm.Intercept
		for j := 0; j < len(linear_model.lm.Coefficient); j++ {
			prediction = prediction + linear_model.lm.Coefficient[j]*dataset.Features[i][j]
		}
		actual := dataset.Label[i]
		score = score + (prediction-actual)*(prediction-actual)
	}

	// add regularization loss
	if linear_model.lm.Penalty == "l1" {
		for j := 0; j < len(linear_model.lm.Coefficient); j++ {
			score = score + linear_model.lm.Alpha*math.Abs(linear_model.lm.Coefficient[j])
		}
	} else if linear_model.lm.Penalty == "l2" {
		for j := 0; j < len(linear_model.lm.Coefficient); j++ {
			score = score + linear_model.lm.Alpha*linear_model.lm.Coefficient[j]*linear_model.lm.Coefficient[j]
		}
	} else if linear_model.lm.Penalty == "elasticnet" {
		for j := 0; j < len(linear_model.lm.Coefficient); j++ {
			score = score + linear_model.lm.Alpha*(1-linear_model.lm.L1Ratio)*linear_model.lm.Coefficient[j]*linear_model.lm.Coefficient[j] + linear_model.lm.Alpha*linear_model.lm.L1Ratio*math.Abs(linear_model.lm.Coefficient[j])
		}
	}
	return score
}

func LinearModelMapper(linearModelObj ml_schemas.LinearModel) []float64 {
	x := []float64{}
	x = append(x, linearModelObj.Intercept)
	for i := 0; i < len(linearModelObj.Coefficient); i++ {
		x = append(x, linearModelObj.Coefficient[i])
	}
	return x
}

func LinearModelMSEObjective(linearModelObj ml_schemas.LinearModel, data ml_schemas.LabelledDataset) optim.ObjectiveFunction {
	// initial value is created through LinearModelMapper(...)
	return func(x []float64) float64 {
		var lm LinearModel = LinearModel{linearModelObj}

		// set weights to be x
		for i := 0; i < len(x); i++ {
			if i == 0 {
				lm.lm.Intercept = x[i]
			} else {
				lm.lm.Coefficient[i-1] = x[i]
			}
		}
		return lm.MSELoss(data)
	}
}
