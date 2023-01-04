package ml_schemas

type LinearModel struct {
	Coefficient []float64
	Intercept   float64
	Penalty     string
	Alpha       float64
	L1Ratio     float64
}
