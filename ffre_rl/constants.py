"""
Physical constants for FFRE simulations.

This module provides physical constants needed for simulating particle dynamics
in Fission Fragment Rocket Engines.
"""

import numpy as np

# Elementary charge
ELECTRON_CHARGE = 1.602176634e-19  # Coulombs

# Alpha particle properties
ALPHA_PARTICLE_CHARGE = 2 * ELECTRON_CHARGE  # Coulombs
ALPHA_PARTICLE_MASS = 6.644657230e-27  # kg

# Energies of alpha particles from different sources (in eV)
AM241_ENERGY = 5.5e6 * ELECTRON_CHARGE  # Joules (5.5 MeV)
THORIUM232_ENERGY = 4.0816e6 * ELECTRON_CHARGE  # Joules (4.0816 MeV)

# Uranium-235 fission fragment properties
# Simplified - in reality there's a distribution
# Average values for light and heavy fragments
U235_LIGHT_FRAGMENT_MASS = 9.5e-26  # kg
U235_LIGHT_FRAGMENT_CHARGE = 20 * ELECTRON_CHARGE  # Coulombs
U235_LIGHT_FRAGMENT_ENERGY = 100e6 * ELECTRON_CHARGE  # Joules (100 MeV)

U235_HEAVY_FRAGMENT_MASS = 1.4e-25  # kg
U235_HEAVY_FRAGMENT_CHARGE = 22 * ELECTRON_CHARGE  # Coulombs
U235_HEAVY_FRAGMENT_ENERGY = 70e6 * ELECTRON_CHARGE  # Joules (70 MeV)

# Vacuum permeability
MU_0 = 4 * np.pi * 1e-7  # H/m

# Boltzmann constant
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K

# Set up memory-efficient __slots__ for dataclasses
DEFAULT_SLOTS = True