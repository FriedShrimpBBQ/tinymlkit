package ml_schemas

type BaggingLinearModel struct {
	Models []LinearModel
	Lambda float64
}
