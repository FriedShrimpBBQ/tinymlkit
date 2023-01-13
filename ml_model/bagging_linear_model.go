package ml_model

import (
	"fmt"
	"math"
	"math/rand"
	"ml_schemas"
	"optim"
)

type BaggingLinearModel struct {
	Bagging ml_schemas.BaggingLinearModel
}

func SamplePoisson(k float64) int {
	var p float64 = 0.0
	var counter int = 0
	for {
		p = p - math.Log(rand.Float64())
		counter = counter + 1
		if p > k {
			break
		}
	}
	return counter
}

func (baggingModel BaggingLinearModel) Predict(dataset ml_schemas.Dataset) []float64 {
	predictions := []float64{}
	predictionsAll := [][]float64{}
	numModels := float64(len(baggingModel.Bagging.Models))
	for _, model := range baggingModel.Bagging.Models {
		predictionsAll = append(predictionsAll, LinearModel{model}.Predict(dataset))
	}
	for i := 0; i < len(dataset.Features); i++ {
		var prediction float64 = 0
		for index, _ := range baggingModel.Bagging.Models {
			fmt.Println(i, index)
			prediction = prediction + predictionsAll[i][index]
		}
		predictions = append(predictions, prediction/numModels)
	}
	return predictions
}

func (baggingModel BaggingLinearModel) FitAndCopy(data ml_schemas.LabelledDataset, max_iter int) BaggingLinearModel {
	var weights []float64
	var models []ml_schemas.LinearModel
	for _, linearModel := range baggingModel.Bagging.Models {
		lm := LinearModel{linearModel}.SetWeightsAndCopy(LinearModelInitWeights(len(data.Features[0])))
		for k := 0; k < SamplePoisson(baggingModel.Bagging.Lambda); k++ {
			weights = optim.RandomOptim(LinearModelMSEObjective(lm.Lm, data), lm.LinearModelMapper(), max_iter)
			linearModel = lm.SetWeightsAndCopy(weights).Lm
		}
		models = append(models, LinearModel{linearModel}.SetWeightsAndCopy(weights).Lm)
	}
	baggingModel.Bagging.Models = models
	return baggingModel
}
