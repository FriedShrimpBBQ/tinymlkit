package ml_schemas

type LabelledDataset struct {
	Features [][]float64
	Label    []float64
}

type Dataset struct {
	Features [][]float64
}
