# Copyright (C) 2017 Tianyi Li
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
from stability_basic import *

ell_list = pl.linspace(0.1, 1.0, 10)

# AT1 model
for ell in ell_list:
    problem = StabilityBar(ndim=1, ell=ell)
    problem.solve()

# ATk model
# k_list = pl.linspace(2, 12, 6)
# for k in k_list:
#     for ell in ell_list:
#         problem = StabilityBar(ndim=1, ell=ell, law="ATk", k=k)
#         problem.solve()
