#!/usr/bin/env python

from distutils.core import setup

setup(name = "gradient_damage",
      version = "1.0",
      description = "FEniCS Gradient Damage",
      author = "Corrado Maurini and Tianyi Li",
      author_email = "corrado.maurini@upmc.fr",
      url = "https://bitbucket.org/cmaurini/gradient-damage",
      packages = ["gradient_damage"],
      package_dir = {"gradient_damage": "gradient_damage"},
      scripts = [],
      data_files = [],
      license = "GPL version 3 or later",
      platforms = ["Linux", "Mac OS-X", "Unix"],
      long_description = "See https://bitbucket.org/cmaurini/gradient-damage")
