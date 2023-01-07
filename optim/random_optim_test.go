package optim

import (
	"testing"
)

func sampleObjectiveFunction(x []float64) float64 {
	var sum float64 = 0
	for i := 0; i < len(x); i++ {
		sum = sum + x[i]
	}
	return sum
}

func TestRandomOptim(t *testing.T) {
	x0 := []float64{1, 2, 3, 4}
	x1 := RandomOptim(sampleObjectiveFunction, x0, 100)

	if sampleObjectiveFunction(x0) < sampleObjectiveFunction(x1) {
		t.Fatalf("the returned outcome should be the same or an improvement")
	}
}
