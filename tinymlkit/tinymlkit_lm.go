package main

import (
	"flag"
	"fmt"
	"ml_model"
	"ml_schemas"
	"optim"
)

func main() {
	coefficient := []float64{}
	intercept := 0.0
	penalty := "l2"
	alpha := 0.0001
	l1ratio := 0.15

	flag.StringVar(&penalty, "penalty", "l2", "Penalty for the linear model")
	flag.Float64Var(&alpha, "alpha", 0.0001, "Regularizer weight")
	flag.Float64Var(&l1ratio, "l1_ratio", 0.15, "l1 ratio if penalty is elasticnet")
	flag.Parse()

	baseLM := ml_schemas.LinearModel{coefficient, intercept, penalty, alpha, l1ratio}

	labelledDataset := &ml_schemas.LabelledDataset{[][]float64{{1.0, 1.0, 1.0, 1.0}}, []float64{14.0}}
	lm := ml_model.LinearModel{baseLM}.SetWeightsAndCopy(ml_model.LinearModelInitWeights(len(labelledDataset.Features[0])))
	fmt.Println("Initial Loss\t", lm.MSELoss(*labelledDataset))
	// train the model
	x1 := optim.RandomOptim(ml_model.LinearModelMSEObjective(lm.Lm, *labelledDataset), lm.LinearModelMapper(), 100)
	fmt.Println("After Optim\t", lm.SetWeightsAndCopy(x1).MSELoss(*labelledDataset))
}
