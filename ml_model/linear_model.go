package ml_model

// map model setup to something compatible with optim schema

import (
	"math"
	"math/rand"
	"ml_schemas"
	"optim"
)

type LinearModel struct {
	Lm ml_schemas.LinearModel
}

func (linear_model LinearModel) Predict(dataset ml_schemas.Dataset) []float64 {
	predictions := []float64{}
	for i := 0; i < len(dataset.Features); i++ {
		var prediction float64 = linear_model.Lm.Intercept
		for j := 0; j < len(linear_model.Lm.Coefficient); j++ {
			prediction = prediction + linear_model.Lm.Coefficient[j]*dataset.Features[i][j]
		}
		predictions = append(predictions, prediction)
	}
	return predictions
}

func (linear_model LinearModel) MSELoss(dataset ml_schemas.LabelledDataset) float64 {
	var score float64 = 0.0
	for i := 0; i < len(dataset.Features); i++ {
		var prediction float64 = linear_model.Lm.Intercept
		for j := 0; j < len(linear_model.Lm.Coefficient); j++ {
			prediction = prediction + linear_model.Lm.Coefficient[j]*dataset.Features[i][j]
		}
		actual := dataset.Label[i]
		score = score + (prediction-actual)*(prediction-actual)
	}

	// add regularization loss
	if linear_model.Lm.Penalty == "l1" {
		for j := 0; j < len(linear_model.Lm.Coefficient); j++ {
			score = score + linear_model.Lm.Alpha*math.Abs(linear_model.Lm.Coefficient[j])
		}
	} else if linear_model.Lm.Penalty == "l2" {
		for j := 0; j < len(linear_model.Lm.Coefficient); j++ {
			score = score + linear_model.Lm.Alpha*linear_model.Lm.Coefficient[j]*linear_model.Lm.Coefficient[j]
		}
	} else if linear_model.Lm.Penalty == "elasticnet" {
		for j := 0; j < len(linear_model.Lm.Coefficient); j++ {
			score = score + linear_model.Lm.Alpha*(1-linear_model.Lm.L1Ratio)*linear_model.Lm.Coefficient[j]*linear_model.Lm.Coefficient[j] + linear_model.Lm.Alpha*linear_model.Lm.L1Ratio*math.Abs(linear_model.Lm.Coefficient[j])
		}
	}
	return score
}

func (linear_model LinearModel) LinearModelMapper() []float64 {
	x := []float64{}
	x = append(x, linear_model.Lm.Intercept)
	for i := 0; i < len(linear_model.Lm.Coefficient); i++ {
		x = append(x, linear_model.Lm.Coefficient[i])
	}
	return x
}

func (linear_model LinearModel) SetWeightsAndCopy(weights []float64) LinearModel {
	// set weights
	coefficient := []float64{}
	for i := 0; i < len(weights); i++ {
		if i == 0 {
			linear_model.Lm.Intercept = weights[i]
		} else {
			coefficient = append(coefficient, weights[i])
		}
	}
	linear_model.Lm.Coefficient = coefficient
	return linear_model
}

func sampleValue() float64 {
	return (rand.Float64() - 0.5) * 2
}

func LinearModelInitWeights(numFeatures int) []float64 {
	weights := []float64{sampleValue()}
	for i := 0; i < numFeatures; i++ {
		weights = append(weights, sampleValue())
	}
	return weights
}

func LinearModelMSEObjective(linearModelObj ml_schemas.LinearModel, data ml_schemas.LabelledDataset) optim.ObjectiveFunction {
	// initial value is created through LinearModelMapper(...)
	return func(x []float64) float64 {
		lm := LinearModel{linearModelObj}.SetWeightsAndCopy(x)
		return lm.MSELoss(data)
	}
}

func (linear_model LinearModel) FitAndCopy(data ml_schemas.LabelledDataset, max_iter int) LinearModel {
	x1 := optim.RandomOptim(LinearModelMSEObjective(linear_model.Lm, data), linear_model.LinearModelMapper(), max_iter)
	return linear_model.SetWeightsAndCopy(x1)
}
