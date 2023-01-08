package optim

// Implement a really naive random optimization algorithm

import (
	"math/rand"
)

type ObjectiveFunction func(x0 []float64) float64

func randomOptimSingleIter(objFunc ObjectiveFunction, x0 []float64) []float64 {
	// run single iteration - returns a new updated x0
	x1 := []float64{}
	for i := 0; i < len(x0); i++ {
		x1 = append(x1, x0[i]+(rand.Float64()-0.5)*2)
	}

	if objFunc(x0) < objFunc(x1) {
		return x0
	} else {
		return x1
	}
}

func RandomOptim(objFunc ObjectiveFunction, x0 []float64, maxiter int) []float64 {
	for i := 0; i < maxiter; i++ {
		x0 = randomOptimSingleIter(objFunc, x0)
	}
	return x0
}
