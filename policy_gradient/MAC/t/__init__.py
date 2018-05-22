from gym.envs.registration import registry, register, make, spec

for k in [0.1,0.2,0.3,0.5,0.75,1.5,2]:
	register(
	    id='CartPolelearned'+str(k)+'-v0',
	    entry_point='t.cartpole_learned:NewCartPoleEnv',
	    kwargs={
	        'model_name': 'cartpole_learned_k'+str(k)
	    }
	)