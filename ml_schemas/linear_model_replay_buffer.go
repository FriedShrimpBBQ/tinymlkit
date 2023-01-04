// replay_buffer - I'm really bad at naming things, and clearly I'm stuck in reinforcement learning world
// will rename if there is a more appropriate name for this. It is the replay buffer used by the simplex
// and also the buffer that is generated from the federated learning agents

package ml_schemas

type LinearModelReplayBuffer struct {
	Weights [][]float64
	Label   []float64
}
