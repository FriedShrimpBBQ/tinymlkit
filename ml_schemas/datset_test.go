package ml_schemas

import (
	"testing"
)

func TestDatasetJson(t *testing.T) {
	sampleDataset := &LabelledDataset{}
	sampleDataset.UnmarshalJSON([]byte(`{"Features": [[1,2,3,4]], "Label": [5]}`))

	if sampleDataset.Label[0] != 5.0 {
		t.Fatalf("sampleDataset.Label - got %f, wanted %f", sampleDataset.Label, 5.0)
	}
	if sampleDataset.Features[0][0] != 1.0 {
		t.Fatalf("Failed to parse features")
	}
}
