"""City-wide routing and signal optimisation - the last stage of the pipeline (work package WP4).

    network         the road network: nodes, directed links, movements, phases; travel times from the cameras
    google_traffic  Google Routes API congestion as a numeric fill-in where no camera looks (requirement R-8)
    routing         shortest path, alternatives and incident-aware routes on the current travel times
    signals         fixed-time and max-pressure controllers, emergency-vehicle priority, timing recommendations
    sim             a deterministic mesoscopic queue simulator that the controllers are validated on

This is a simulation-validated controller and router. It has not controlled a real signal, it is not
connected to one, and it needs no GPU: max pressure (Varaiya, Transportation Research Part C, 2013) is the
training-free controller that MPLight's pressure reward is built on, and the reinforcement-learning layer the
report reviews (MPLight, CoLight, EMVLight) is deliberately not reproduced here.
"""
