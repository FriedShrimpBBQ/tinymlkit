module ml_model

go 1.19

replace ml_schemas => ../ml_schemas
replace optim => ../optim

require (
	github.com/josharian/intern v1.0.0 // indirect
	github.com/mailru/easyjson v0.7.7 // indirect
	ml_schemas v0.0.0-00010101000000-000000000000 // indirect
	optim v0.0.0-00010101000000-000000000000 // indirect
)
