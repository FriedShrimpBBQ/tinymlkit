package ml_model

// map model setup to something compatible with optim schema

import (
	"ml_schemas"
)

func LinearModelMapper(linearModelObj *ml_schemas.LinearModel) []float64 {
	x := []float64{}
	x = append(x, linearModelObj.Intercept)
	for i := 0; i < len(linearModelObj.Coefficient); i++ {
		x = append(x, linearModelObj.Coefficient[i])
	}
	return x
}
