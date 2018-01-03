#!/usr/bin/env py.test
#
# Copyright (C) 2015 Tianyi Li
#
# This file is part of FEniCS Gradient Damage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from gradient_damage import *
from traction_basic import *
import pytest
import shutil

set_log_level(ERROR)

def test_traction():

    # Run a 2-d simulation
    problem = Traction(2)
    problem.solve()

    # Theoretic value (effective Gc)
    Gc_eff = (1+3*0.01/(8*0.1))*(8*0.1/3)
    S_theo = Gc_eff*0.1  # transverse crack near the center with 0.1 the width

    # Find the calculated energies
    S = problem.dissipated_energy_value

    # Cleaning
    shutil.rmtree(problem.save_dir, ignore_errors=True)

    # Pytest
    assert round(S_theo - S, 2) == 0
