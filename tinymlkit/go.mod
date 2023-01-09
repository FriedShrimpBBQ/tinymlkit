module tinymlkit

go 1.19

replace ml_model => ../ml_model

replace ml_schemas => ../ml_schemas

replace optim => ../optim

require (
	ml_model v0.0.0-00010101000000-000000000000
	ml_schemas v0.0.0-00010101000000-000000000000
)

require (
	github.com/josharian/intern v1.0.0 // indirect
	github.com/mailru/easyjson v0.7.7 // indirect
	optim v0.0.0-00010101000000-000000000000 // indirect
)
