package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
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
	json_data := ""
	max_iter := 1000

	labelledDataset := &ml_schemas.LabelledDataset{[][]float64{{1.0, 1.0, 1.0, 1.0}}, []float64{14.0}}

	flag.StringVar(&penalty, "penalty", "l2", "Penalty for the linear model")
	flag.Float64Var(&alpha, "alpha", 0.0001, "Regularizer weight")
	flag.Float64Var(&l1ratio, "l1_ratio", 0.15, "l1 ratio if penalty is elasticnet")
	flag.StringVar(&json_data, "json_data", "", "Path to the json dataset")
	flag.IntVar(&max_iter, "max_iter", 1000, "Number of iterations the algorithm will run")
	flag.Parse()

	baseLM := ml_schemas.LinearModel{coefficient, intercept, penalty, alpha, l1ratio}

	if json_data != "" {
		// read in data
		fileContent, err := ioutil.ReadFile(json_data)
		if err != nil {
			log.Fatal(err)
		}
		labelledDataset.UnmarshalJSON(fileContent)
	}

	lm := ml_model.LinearModel{baseLM}.SetWeightsAndCopy(ml_model.LinearModelInitWeights(len(labelledDataset.Features[0])))
	fmt.Println("Initial Loss\t", lm.MSELoss(*labelledDataset))
	// train the model
	x1 := optim.RandomOptim(ml_model.LinearModelMSEObjective(lm.Lm, *labelledDataset), lm.LinearModelMapper(), max_iter)
	fmt.Println("After Optim\t", lm.SetWeightsAndCopy(x1).MSELoss(*labelledDataset))
	fmt.Println("Optim iter 2\t", lm.SetWeightsAndCopy(x1).FitAndCopy(*labelledDataset, 1000).MSELoss(*labelledDataset))

	// setup bagging example
	bagging_setup := ml_schemas.BaggingLinearModel{[]ml_schemas.LinearModel{baseLM, baseLM}, 1}
	bagging_lm := ml_model.BaggingLinearModel{bagging_setup}
	output := bagging_lm.FitAndCopy(*labelledDataset, 10)
	fmt.Println("Bagging Output", output)
}
