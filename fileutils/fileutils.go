package utils

import (
    "errors"
    "fmt"
	"io/ioutil"
	"strings"
	"strconv"
)

type Record struct {
   label []float64
   feature []float64
}

type Dataset struct {
	records []Record
}

func (dataset *Dataset) AddRecord(record Record) {
	dataset.records = append(dataset.records, record)
}


func ParseStringAsVector(record string) []float64 {
	vector := []float64{}
	stringArray := strings.Split(strings.TrimSpace(record), " ")
	for i:=0;i<len(stringArray);i++ {
		if n, err := strconv.ParseFloat(stringArray[i], 64); err == nil {
			vector = append(vector, n)
		} else {
			panic(err)
		}
	}
	return vector
}


func ParseStringAsRecord(recordString string) (Record, error)  {
	label, feature, found := strings.Cut(strings.TrimSpace(recordString), "|")
	if !found {
		return Record{}, errors.New("Invalid record")
	} else {
		return Record{ParseStringAsVector(label), ParseStringAsVector(feature)}, nil
	}
	
}

// read returns training/label, which are both 2d/1d array float64
func Read(filepath string) (Dataset, error) {
    readFile, err := ioutil.ReadFile(filepath)
	if err != nil {
		fmt.Println(err)
	}
    
	lines := strings.Split(string(readFile), "\n")
	
	dataset := Dataset{}
	
	for i:=0;i<len(lines);i++ {
	    if record, err := ParseStringAsRecord(lines[i]); err == nil {
			dataset.AddRecord(record)
		} else {
			fmt.Println("Unable to add record: ", err)
		}
		
	}
	
    return dataset, nil
}

