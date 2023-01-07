// Code generated by easyjson for marshaling/unmarshaling. DO NOT EDIT.

package ml_schemas

import (
	json "encoding/json"
	easyjson "github.com/mailru/easyjson"
	jlexer "github.com/mailru/easyjson/jlexer"
	jwriter "github.com/mailru/easyjson/jwriter"
)

// suppress unused package warning
var (
	_ *json.RawMessage
	_ *jlexer.Lexer
	_ *jwriter.Writer
	_ easyjson.Marshaler
)

func easyjsonEd27195aDecodeMlSchemas(in *jlexer.Lexer, out *LinearModelReplayBuffer) {
	isTopLevel := in.IsStart()
	if in.IsNull() {
		if isTopLevel {
			in.Consumed()
		}
		in.Skip()
		return
	}
	in.Delim('{')
	for !in.IsDelim('}') {
		key := in.UnsafeFieldName(false)
		in.WantColon()
		if in.IsNull() {
			in.Skip()
			in.WantComma()
			continue
		}
		switch key {
		case "Coefficient":
			if in.IsNull() {
				in.Skip()
				out.Coefficient = nil
			} else {
				in.Delim('[')
				if out.Coefficient == nil {
					if !in.IsDelim(']') {
						out.Coefficient = make([][]float64, 0, 2)
					} else {
						out.Coefficient = [][]float64{}
					}
				} else {
					out.Coefficient = (out.Coefficient)[:0]
				}
				for !in.IsDelim(']') {
					var v1 []float64
					if in.IsNull() {
						in.Skip()
						v1 = nil
					} else {
						in.Delim('[')
						if v1 == nil {
							if !in.IsDelim(']') {
								v1 = make([]float64, 0, 8)
							} else {
								v1 = []float64{}
							}
						} else {
							v1 = (v1)[:0]
						}
						for !in.IsDelim(']') {
							var v2 float64
							v2 = float64(in.Float64())
							v1 = append(v1, v2)
							in.WantComma()
						}
						in.Delim(']')
					}
					out.Coefficient = append(out.Coefficient, v1)
					in.WantComma()
				}
				in.Delim(']')
			}
		case "Intercept":
			if in.IsNull() {
				in.Skip()
				out.Intercept = nil
			} else {
				in.Delim('[')
				if out.Intercept == nil {
					if !in.IsDelim(']') {
						out.Intercept = make([]float64, 0, 8)
					} else {
						out.Intercept = []float64{}
					}
				} else {
					out.Intercept = (out.Intercept)[:0]
				}
				for !in.IsDelim(']') {
					var v3 float64
					v3 = float64(in.Float64())
					out.Intercept = append(out.Intercept, v3)
					in.WantComma()
				}
				in.Delim(']')
			}
		case "Label":
			if in.IsNull() {
				in.Skip()
				out.Label = nil
			} else {
				in.Delim('[')
				if out.Label == nil {
					if !in.IsDelim(']') {
						out.Label = make([]float64, 0, 8)
					} else {
						out.Label = []float64{}
					}
				} else {
					out.Label = (out.Label)[:0]
				}
				for !in.IsDelim(']') {
					var v4 float64
					v4 = float64(in.Float64())
					out.Label = append(out.Label, v4)
					in.WantComma()
				}
				in.Delim(']')
			}
		default:
			in.SkipRecursive()
		}
		in.WantComma()
	}
	in.Delim('}')
	if isTopLevel {
		in.Consumed()
	}
}
func easyjsonEd27195aEncodeMlSchemas(out *jwriter.Writer, in LinearModelReplayBuffer) {
	out.RawByte('{')
	first := true
	_ = first
	{
		const prefix string = ",\"Coefficient\":"
		out.RawString(prefix[1:])
		if in.Coefficient == nil && (out.Flags&jwriter.NilSliceAsEmpty) == 0 {
			out.RawString("null")
		} else {
			out.RawByte('[')
			for v5, v6 := range in.Coefficient {
				if v5 > 0 {
					out.RawByte(',')
				}
				if v6 == nil && (out.Flags&jwriter.NilSliceAsEmpty) == 0 {
					out.RawString("null")
				} else {
					out.RawByte('[')
					for v7, v8 := range v6 {
						if v7 > 0 {
							out.RawByte(',')
						}
						out.Float64(float64(v8))
					}
					out.RawByte(']')
				}
			}
			out.RawByte(']')
		}
	}
	{
		const prefix string = ",\"Intercept\":"
		out.RawString(prefix)
		if in.Intercept == nil && (out.Flags&jwriter.NilSliceAsEmpty) == 0 {
			out.RawString("null")
		} else {
			out.RawByte('[')
			for v9, v10 := range in.Intercept {
				if v9 > 0 {
					out.RawByte(',')
				}
				out.Float64(float64(v10))
			}
			out.RawByte(']')
		}
	}
	{
		const prefix string = ",\"Label\":"
		out.RawString(prefix)
		if in.Label == nil && (out.Flags&jwriter.NilSliceAsEmpty) == 0 {
			out.RawString("null")
		} else {
			out.RawByte('[')
			for v11, v12 := range in.Label {
				if v11 > 0 {
					out.RawByte(',')
				}
				out.Float64(float64(v12))
			}
			out.RawByte(']')
		}
	}
	out.RawByte('}')
}

// MarshalJSON supports json.Marshaler interface
func (v LinearModelReplayBuffer) MarshalJSON() ([]byte, error) {
	w := jwriter.Writer{}
	easyjsonEd27195aEncodeMlSchemas(&w, v)
	return w.Buffer.BuildBytes(), w.Error
}

// MarshalEasyJSON supports easyjson.Marshaler interface
func (v LinearModelReplayBuffer) MarshalEasyJSON(w *jwriter.Writer) {
	easyjsonEd27195aEncodeMlSchemas(w, v)
}

// UnmarshalJSON supports json.Unmarshaler interface
func (v *LinearModelReplayBuffer) UnmarshalJSON(data []byte) error {
	r := jlexer.Lexer{Data: data}
	easyjsonEd27195aDecodeMlSchemas(&r, v)
	return r.Error()
}

// UnmarshalEasyJSON supports easyjson.Unmarshaler interface
func (v *LinearModelReplayBuffer) UnmarshalEasyJSON(l *jlexer.Lexer) {
	easyjsonEd27195aDecodeMlSchemas(l, v)
}