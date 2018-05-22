from gym.envs.registration import registry, register, make, spec

for k in [0.2,0.25,0.3,0.35,0.425,0.45,0.475,0.6,0.75,2]:
	register(
	    id='Pendulumlearned'+str(k)+'-v0',
	    entry_point='t.pendulum_learned:NewPendulumEnv',
	    max_episode_steps=200,
	    kwargs={
	        'model_name': 'pendulum_learned_k'+str(k)

	    }
	)