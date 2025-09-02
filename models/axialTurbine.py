"""
Create, simulate, and evaluate an axial runner. This machine is the test case
of `[Eyselein_2025] <https://doi.org/10.3390/en18030677>`_, 
`[Rentschler_2024] <https://doi.org/10.1002/pamm.202400126>`_, 
`[Raj_2024] <https://doi.org/10.1002/pamm.202400104>`_, and
`[Ebel_2024] <https://doi.org/10.48550/arXiv.2410.18358>`_. For the 
publication `[Ebel_2024] <https://doi.org/10.48550/arXiv.2410.18358>`_, the
data set is provided in 
`[axial_turbine_database] <https://doi.org/10.5281/zenodo.14014525>`_.
This GitHub repository serves as the database for this demonstration case.

Run this tutorial by executing:

.. code-block:: bash

  export OSLO_LOCK_PATH=/tmp && export FOAM_SIGFPE=0 \\
    && python3.12 -m doctest build.py

Import ``logging`` package and create a configuration:
"""
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s : %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)

# Import necessary classes from ``dtOO``:

from dtOOPythonSWIG import (
  logMe,
  dtXmlParser,
  baseContainer,
  labeledVectorHandlingConstValue,
  labeledVectorHandlingAnalyticFunction,
  labeledVectorHandlingAnalyticGeometry,
  labeledVectorHandlingBoundedVolume,
  labeledVectorHandlingDtCase,
  labeledVectorHandlingDtPlugin,
  lVHOstateHandler,
  jsonPrimitive,
)

# Import packages from ``pyDtOO``:

from pyDtOO import (
  dtScalarDeveloping, 
  dtForceDeveloping, 
  dtDeveloping
)
from pyDtOO import dtClusteredSingletonState
dtClusteredSingletonState.PREFIX = "T5"
dtClusteredSingletonState.ADDDATA = [
  'P',
  'dH',
  'eta',
  'Vcav',
  #'history',
  #'islandID',
]
import sys
dtClusteredSingletonState.ADDDATADEF = [
  [sys.float_info.max,], # P
  [sys.float_info.max,], # dH
  [sys.float_info.max,], # eta
  [sys.float_info.max,], # VCav
]
import pyDtOO as pd
import foamlib
import numpy as np



class axialTurbine:
  """
  This includes all the steps for geometry creation, meshing, simulating and evaluating
  simultaneously for an axial turbine case. 
  """
  import sys

  OMEGA = 7.53982
  DHZUL = -2.4
  ITERMAX = 1000
  def __init__(self, 
               x=None,
               stateNumber=-1,
               ):
    """
    Initializes the axial turbine case.

    Args:
      state_id (str): The ID of the state.
      data_dir (str): The directory where the data is stored.
      prefix (str, optional): The prefix of the state. Defaults to 'T1'.
      case (str, optional): The case of the axial turbine. Defaults to 'tists_ru_of'.
  
    """
    if stateNumber<0:
      self.state_ = pd.dtClusteredSingletonState(
        defObj = [x],
        defFit = [axialTurbine.FailedFitness(),]
      ).state()
      """str: State label."""
      self.x = x
    else:
      s = pd.dtClusteredSingletonState(stateNumber)
      if x==None:
          self.x = s.objective()
      if np.sum(s.objective() - self.x) > 0.001:
        raise ValueError("Objectives does not match.")
      self.state_ = s.state()
     
    self.prefix = "T5"
    self.case = "tistos_ru_of_n"

    self.P_ = 0
    self.dH_ = 0
    self.eta_ = 0
    self.Vcav_ = 0

    self.parser = None # XML parser
    self.bC_ = None# = baseContainer()
    self.cV_ = None# = labeledVectorHandlingConstValue()
    self.aF_ = None# = labeledVectorHandlingAnalyticFunction()
    self.aG_ = None# = labeledVectorHandlingAnalyticGeometry()
    self.bV_ = None# = labeledVectorHandlingBoundedVolume()
    self.dC_ = None# = labeledVectorHandlingDtCase()
    self.dP_ = None# = labeledVectorHandlingDtPlugin()

    #self.weights_ = {"tl": 1/3, "n": 1/3, "vl": 1/3}

    #self.isOk_ = False


    logMe.initLog('build.'+self.state_+'.log')


#  def _initialize_components(self):
  #   """
  #   Initialize geometric components 
  #   """
  #   # Create basic container:




  @staticmethod
  def DoF():
    return [
      {'label': 'cV_ru_alpha_1_ex_0.0', 'min': -0.155, 'max': 0.025}, 
      {'label': 'cV_ru_alpha_1_ex_0.5', 'min': -0.19, 'max': -0.01}, 
      {'label': 'cV_ru_alpha_1_ex_1.0', 'min': -0.19, 'max': -0.01}, 
      {'label': 'cV_ru_alpha_2_ex_0.0', 'min': -0.08, 'max': 0.1}, 
      {'label': 'cV_ru_alpha_2_ex_0.5', 'min': -0.08, 'max': 0.1}, 
      {'label': 'cV_ru_alpha_2_ex_1.0', 'min': -0.08, 'max': 0.07}, 
      {'label': 'cV_ru_offsetM_ex_0.0', 'min': 1.0, 'max': 1.5},
      {'label': 'cV_ru_offsetM_ex_0.5', 'min': 1.0, 'max': 1.5}, 
      {'label': 'cV_ru_offsetM_ex_1.0', 'min': 1.0, 'max': 1.5}, 
      {'label': 'cV_ru_ratio_0.0', 'min': 0.4, 'max': 0.6}, 
      {'label': 'cV_ru_ratio_0.5', 'min': 0.4, 'max': 0.6}, 
      {'label': 'cV_ru_ratio_1.0', 'min': 0.4, 'max': 0.6}, 
      {'label': 'cV_ru_offsetPhiR_ex_0.0', 'min': -0.15, 'max': 0.15}, 
      {'label': 'cV_ru_offsetPhiR_ex_0.5', 'min': -0.15, 'max': 0.15}, 
      {'label': 'cV_ru_offsetPhiR_ex_1.0', 'min': -0.15, 'max': 0.15}, 
      {'label': 'cV_ru_bladeLength_0.0', 'min': 0.4, 'max': 0.8}, 
      {'label': 'cV_ru_bladeLength_0.5', 'min': 0.6, 'max': 1.0}, 
      {'label': 'cV_ru_bladeLength_1.0', 'min': 0.8, 'max': 1.3}, 
      {'label': 'cV_ru_t_le_a_0', 'min': 0.005, 'max': 0.06}, 
      {'label': 'cV_ru_t_le_a_0.5', 'min': 0.005, 'max': 0.06},
      {'label': 'cV_ru_t_le_a_1', 'min': 0.005, 'max': 0.06}, 
      {'label': 'cV_ru_t_mid_a_0', 'min': 0.005, 'max': 0.06}, 
      {'label': 'cV_ru_t_mid_a_0.5', 'min': 0.005, 'max': 0.06},
      {'label': 'cV_ru_t_mid_a_1', 'min': 0.005, 'max': 0.06}, 
      {'label': 'cV_ru_t_te_a_0', 'min': 0.005, 'max': 0.06}, 
      {'label': 'cV_ru_t_te_a_0.5', 'min': 0.005, 'max': 0.06},
      {'label': 'cV_ru_t_te_a_1', 'min': 0.005, 'max': 0.06}, 
      {'label': 'cV_ru_u_mid_a_0', 'min': 0.4, 'max': 0.6}, 
      {'label': 'cV_ru_u_mid_a_0.5', 'min': 0.4, 'max': 0.6},
      {'label': 'cV_ru_u_mid_a_1', 'min': 0.4, 'max': 0.6},
    ]

  def allSteps(self):
    """
    Complete steps includes geometry creation, meshing, simulating and evaluating
    """

    #try:
    self._create_geometry()
    self._simulate()
    
    # toggle isOk_ to True
    #self.isOk_ = True

    return self._evaluate()
    # except:
    #   logging.info('Error in some steps!')
    #   # toggle isOk_ to False
    #   self.isOk_ = False
    #   return self.FailedFitness()

  @staticmethod
  def FailedFitness():
    """Failed fitness.

    Returns the value that represents a failed design.

    Returns
    -------
    float
      Fitness for a failed design.
    """
    return axialTurbine.sys.float_info.max 

  @staticmethod
  def IsFailedFitness(fit):
    """Check if the fitness is a failed fitness.

    Args:
      fit (float): Fitness value.

    Returns
    -------
    bool
      ``True`` if the fitness is a failed fitness.
    """
 
    if fit == axialTurbine.sys.float_info.max:
      return True
    else:
      return False

  def _create_geometry(self):
    logging.info("Create geometry.")

    # Create basic container:
    logging.info("Create container ...")
    lVHOstateHandler.clear()
    self.bC_ = baseContainer()
    self.cV_ = labeledVectorHandlingConstValue()
    self.aF_ = labeledVectorHandlingAnalyticFunction()
    self.aG_ = labeledVectorHandlingAnalyticGeometry()
    self.bV_ = labeledVectorHandlingBoundedVolume()
    self.dC_ = labeledVectorHandlingDtCase()
    self.dP_ = labeledVectorHandlingDtPlugin()

    # Initialize XML parser
    logging.info("Initialize parser ...")
    self.parser = dtXmlParser.init("machine.xml", "templateState.xml").reference()
    self.parser.parse()

    logging.info("Create constValue and loadStateToConst ...")
    self.parser.createConstValue(self.cV_)
    self.parser.loadStateToConst("templateState", self.cV_)

    # Iterate over all parameters from database and set the value in the 
    # correspoding :ref:`constValue` object. The array ``cVArr`` reuturns the
    # correct label:

    logging.info("Assign constValue ...")
    cc = 0
    for anObj in self.x:
      self.cV_[ axialTurbine.DoF()[cc]['label'] ].setValue( anObj )
      cc = cc + 1

    # Create geometry, apply the ``ru_adjustDomain`` plugin, and create geometry
    # again. The plugin adjusts the runner domain by changing the DOFs; therefore,
    # it necessary to create the geometry twice:

    self.parser.destroyAndCreate(self.bC_, self.cV_, self.aF_, self.aG_, self.bV_, self.dC_, self.dP_)
    self.dP_.get('ru_adjustDomain').apply()
    self.parser.destroyAndCreate(self.bC_, self.cV_, self.aF_, self.aG_, self.bV_, self.dC_, self.dP_)
  
    # Make a state in the container of :ref:`constValue`:
    lVHOstateHandler().makeState(self.state_)

  def _simulate(self):
    """
    Simulate the axial Turbine
    """

    # dC = self.dC
    # #for i in ["tl" , "n", "vl"]:
    # try:
    #
    # Create OpenFoam case
    #
    #for i in self.dC_:
    #  print(i.getLabel())
    # print(lVHOstateHandler().commonState())
    self.dC_[self.case].runCurrentState()
    #
    # Initialize ``foamlib`` object; define parallel solution on 4 
    # processors; simulate 100 iterations laminar with a ``faceLimited``
    # scheme
    #
    fc = foamlib.FoamCase( self.dC_[self.case].getDirectory( self.state_ ) )
    # print(f"fc: {fc}")
    fc.decompose_par_dict['method'] = 'metis'
    fc.decompose_par_dict['numberOfSubdomains'] = 32
    fc.control_dict['writeInterval'] = 100
    fc.control_dict['endTime'] = 100
    fc.turbulence_properties["RAS"]["turbulence"] = False
    fc.fv_schemes['gradSchemes']['none'] = 'faceLimited Gauss linear 0.33'
    fc.fv_schemes['gradSchemes']['grad(p)'] = 'faceLimited Gauss linear 0.33'
    fc.fv_schemes['gradSchemes']['grad(U)'] = 'faceLimited Gauss linear 0.33'
    fc.fv_schemes['divSchemes']['div(phi,U)'] = 'Gauss linearUpwindV faceLimited Gauss linear 0.33'
    fc.run()
    #
    # Simulate until 1000 iterations reached as a turbulent simulation with
    # a ``cellLimited`` scheme
    #
    fc.control_dict['writeInterval'] = self.ITERMAX
    fc.control_dict['endTime'] = self.ITERMAX
    fc.turbulence_properties["RAS"]["turbulence"] = True
    fc.fv_schemes['gradSchemes']['none'] = 'cellLimited Gauss linear 0.33'
    fc.fv_schemes['gradSchemes']['grad(p)'] = 'cellLimited Gauss linear 0.33'
    fc.fv_schemes['gradSchemes']['grad(U)'] = 'cellLimited Gauss linear 0.33'
    fc.fv_schemes['divSchemes']['div(phi,U)'] = 'Gauss linearUpwindV cellLimited Gauss linear 0.33'
    fc.run()
    #
    # Reconstruct the case
    #
    fc.reconstruct_par()
    #

    self.directory_ = str(fc.path)


  def _evaluate(self):
    """
    Evaluate the axial turbine case.
    """
    dHZUL = axialTurbine.DHZUL

    # Get case directory
    #fc = foamlib.FoamCase(self.directory_)
    fc = foamlib.FoamCase( self.dC_[self.case].getDirectory( self.state_ ) )

    
    # Extract the rotation speed from the case
    #
    #omega = np.abs(fc.read_dictionary('constant/MRFProperties')['MRF_RU']['omega'])
    omega = axialTurbine.OMEGA
    
    #try:
    #
    # Read data from ``postProcessing`` folder
    #
    Q_ru_dev = dtScalarDeveloping( 
      dtDeveloping(str(fc.path/'postProcessing/Q_ru_in/100')).Read() 
    )
    pIn_ru_dev = dtScalarDeveloping( 
      dtDeveloping(str(fc.path/'postProcessing/ptot_ru_in/100')).Read() 
    )
    pOut_ru_dev = dtScalarDeveloping( 
      dtDeveloping(str(fc.path/'postProcessing/ptot_ru_out/100')).Read() 
    )
    Vcav_dev = dtScalarDeveloping( 
      dtDeveloping(str(fc.path/'postProcessing/V_CAV/100')).Read() 
    )
    F_dev = dtForceDeveloping( 
      dtDeveloping(str(fc.path/'postProcessing/forces')).Read(
        {'force.dat' : ':,4:10', 'moment.dat' : ':,4:10', '*.*' : ''}
      ) 
    )
    #
    # Calculate power, head, efficiency, and cavitation volume; average
    # simulation results over 100 iterations
    #
    self.P_ = F_dev.MomentMeanLast(100)[2] * omega
    self.dH_ = (pOut_ru_dev.MeanLast(100) - pIn_ru_dev.MeanLast(100)) / 9.81
    self.eta_ = self.P_ / (1000. * 9.81 * self.dH_ * Q_ru_dev.MeanLast(100) )
    self.Vcav_ = Vcav_dev.MeanLast(100)

    logging.info(
      "P: %f, dH: %f, eta: %f, Vcav: %f",
      self.P_, self.dH_, self.eta_, self.Vcav_
      )

    # Initialize an object of dtClusteredSingletonState class
    # to have access to the database; adn update the runData files

    sh = pd.dtClusteredSingletonState(self.state_)
    sh.update('dH', self.dH_)
    sh.update('eta', self.eta_)
    sh.update('Vcav', self.Vcav_)
    sh.update('P', self.P_)

    #
    # Check if number of iterations reached and if it is a turbine not a
    # pump
    #
    if fc[-1].name != str(self.ITERMAX):
      raise ValueError('Max number of iterations not reached.')
    if np.abs(self.eta_) > 1:
      raise ValueError('Pump detected.')
    if self.dH_ > 0:
        raise ValueError('For turbine, dH should be less than 0 => inlet > outlet')

    #if self.isOk_:
      #
      # Calculate fitness as
      # fit = abs(1-\eta) + Vcav + abs(dH - dHZUL)
      #
    fit = abs(1.0 - abs(self.eta_)) + self.Vcav_ + abs(self.dH_ - dHZUL)

    # logging.info(
    #   "Fitness: %f, P: %f, dH: %f, eta: %f, Vcav: %f",
    #   fit, self.P_, self.dH_, self.eta_, self.Vcav_
    # )

    sh.update('fitness', fit)

    if self.IsFailedFitness(fit):
      logging.info("Failed fitness!")
      
      return axialTurbine.FailedFitness()

    else:
    # toggle isOk_ to False
    #self.isOk_ = False

      return fit

if __name__ == '__main__':
  import sys
  stateNumber = int(sys.argv[1])
  axialTurbine(stateNumber=stateNumber).allSteps()
