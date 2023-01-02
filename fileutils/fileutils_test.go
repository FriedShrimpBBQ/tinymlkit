package utils

import (
    "testing"
)

// TestHelloName calls utils.Hello with a name, checking
// for a valid return value.

// TestHelloEmpty calls utils.Hello with an empty string,
// checking for an error.
func TestParseStringAsVector(t *testing.T) {
	vector := ParseStringAsVector("1 2 3")
	
	if vector[0] != 1 {
		t.Fail()
	}
	if vector[1] != 2 {
		t.Fail()
	}
	if vector[2] != 3 {
		t.Fail()
	}
}

func TestParseStringAsRecord(t *testing.T) {
	record, err := ParseStringAsRecord("1 | 2 3")
	
	if err != nil {
	    t.Fail()
	}
	if record.label[0] != 1 {
		t.Fail()
	}
	if record.feature[0] != 2 {
		t.Fail()
	}
	if record.feature[1] != 3 {
		t.Fail()
	}
}