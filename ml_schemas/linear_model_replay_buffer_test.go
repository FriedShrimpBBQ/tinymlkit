package ml_schemas

import (
	"testing"
)

func TestLinearModelReplayBufferJson(t *testing.T) {
	sampleReplayBuffer := &LinearModelReplayBuffer{}
	sampleReplayBuffer.UnmarshalJSON([]byte(`{"Coefficient": [[1,2,3,4]], "Label": [5]}`))

	if sampleReplayBuffer.Coefficient[0][0] != 1.0 {
		t.Fatalf("sampleReplayBuffer.Coefficient - got %f, wanted %f", sampleReplayBuffer.Coefficient[0][0], 5.0)
	}
	if sampleReplayBuffer.Label[0] != 5.0 {
		t.Fatalf("Failed to parse label")
	}
}
