"""
Modified from hydFoilOpt from dtOO by Alex with latent space optimization
"""
import os
import glob
import subprocess
import sys
import numpy as np
from timeit import default_timer as timer

#from scipy.optimize import differential_evolution

import pygmo as pg



# check if foamlib is install else install it
# try:
#    import foamlib
# except ImportError:
#    # Install the foamlib package
#    print("foamlib not found. Installing foamlib...")
#
#    subprocess.check_call([sys.executable, "-m", "pip", "install", "foamlib"])
#    import foamlib


from utils import *
from models.axialTurbine import *


if __name__ == "__main__":

    #----------------------------#
    # Optimization Block
    #----------------------------#

    # set seeds
    set_seeds(seed=42)

    bounds = [(d['min'], d['max']) for d in axialTurbine.DoF()]
    #print(f"bounds: {bounds}")

    # optimization function with x as latent vector
    def optimizeAxialTurbine(x):
        s = dtClusteredSingletonState(
          defObj=x, defFit=axialTurbine.FailedFitness()
        )
        s.update('objective', x)

        try:
            subprocess.run(["python", "./models/axialTurbine.py", str(s.id())], timeout=1500)
            return s.fitness()
        except subprocess.TimeoutExpired:
            logging.info("Timeout occurred")
            return axialTurbine.FailedFitness()
            
    start = timer()

    result = differential_evolution(
        optimizeAxialTurbine,
        bounds=bounds,
        popsize=5,
        maxiter=100,
        polish=False,
        strategy='rand1bin',
        updating='deferred',
        mutation=(0.7, 1),
        recombination=0.9,
    )
    end = timer()

    logging.info(f"Optimization took {end - start} seconds")
    logging.info(f"Result looks something like ...{result}")


    

