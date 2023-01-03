package linear_model_schema

type LinearModelOptions struct {
	Penalty string
	Alpha   float64
	L1Ratio float64
}

type LinearModel struct {
	Coefficient   []float64
	Intercept     float64
	Simplex       [][]float64
	SimplexLabels []float64
	Options       LinearModelOptions
}
