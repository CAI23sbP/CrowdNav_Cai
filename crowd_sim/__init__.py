import gymnasium.envs.registration as registration

registration.register(
    id='example_scan-v0',
    entry_point='crowd_sim.envs:ExampleSimScan',
)