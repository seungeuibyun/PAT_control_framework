"""Frozen-PINN predictive pointing control for HAPS free-space optical links.

spec          bench configuration (config/bench.json) and derived quantities
optics        receiver responses (bench direct beam, HAPS focal plane)
disturbance   rail-translation and injected-vibration profiles
plant         batched 0.5 ms closed-loop simulation of the bench
controllers   PID baseline, Frozen-PINN controller, oracle bound
predictor     Frozen-PINN disturbance predictor (sampled frozen features, linear readout)
command_layer Frozen-PINN in time with the FSM-dynamics residual (solves P1)
comm          HAPS-equivalent link metrics (BER, outage, adaptive rate)
empirical     loading and analysing measured bench logs
figures       analytical and empirical figures
"""
