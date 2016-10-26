import numpy as np
import pandas as pd
import datetime as dt


class PyCrop(object):
    """
    Simple Crop model class
    """

    def __init__(self, doyp):
        """
        Initialise the class
                :rtype : object
        """
        self.irrigation = GetData.get_irrigation()
        self.weather = GetData.get_weather()
        self.soil = GetData.get_soil()
        self.sw_state = self.initial_sw_state
        self.plant = GetData.get_plant()
        self.plant['doyp'] = doyp
        self.status = self.initial_status

    @property
    def initial_sw_state(self):
        sw_state = {'WP': self.soil['DP'] * self.soil['WPp'] * 10.0,
                    'FC': self.soil['DP'] * self.soil['FCp'] * 10.0,
                    'ST': self.soil['DP'] * self.soil['STp'] * 10.0,
                    'SWC_INIT': self.soil['SWC'], 'TRAIN': 0.0,
                    'TIRR': 0.0, 'TESA': 0.0, 'TEPA': 0.0, 'TROF': 0.0,
                    'TDRN': 0.0, 'TINF': 0.0, 'SWC_ADJ': 0.0}

        start = self.weather.index[0]  # First date of weather data
        if start in self.irrigation.index:  # If it exists in irrigation data
            sw_state['TIRR'] += self.irrigation.irr[start]
            sw_state['POTINF'] = self.weather.rain[start] + self.irrigation.irr[start]
        else:  # , or if there is only weather data
            sw_state['POTINF'] = self.weather.rain[start]
        sw_state['TRAIN'] += self.weather.rain[start]

        sw_state['ROF'] = Soil_Water.runoff(POTINF=sw_state['POTINF'], CN=self.soil['CN'])

        sw_state['THE'] = sw_state['WP'] + 0.75 * (sw_state['FC'] - sw_state['WP'])
        sw_state['SWFAC1'], sw_state['SWFAC2'] = Soil_Water.stress(
            SWC=self.soil['SWC'], DP=self.soil['DP'],
            FC=sw_state['FC'], ST=sw_state['ST'],
            WP=sw_state['WP'], THE=sw_state['THE'])
        return sw_state

    @property
    def initial_status(self):
        """
        Set a status dictionary used for control flow
        """
        status = {'endsim': False, 'initial': True}
        return status

    def simulate(self):
        """ Run the model using model class"""
        sw_out, p_out = Model.run_model(self)
        return sw_out, p_out


class GetData(object):
    """
    A class to group functions for getting the data
    """

    @staticmethod
    def get_irrigation():
        """
        Get irrigation data from a file in a relative path.
        """
        irrigation_file = ".Data/IRRIG.INP"
        tmp = []
        with open(irrigation_file, 'r') as f:
            for line in f:
                date, irr = line.split()
                tmp.append([int(date), float(irr)])
        dlist = [(CustomFunctions.gen_datetimes(dateint[0])) for dateint in tmp]
        return pd.DataFrame(data=tmp, index=dlist, columns=['date', 'irr'])

    @staticmethod
    def get_plant():
        """
        Get initial data from Plant.inp file.
        """
        plant_file = "Data/Plant.inp"
        plant = {}
        with open(plant_file, 'r') as f:
            firstline = f.readline().split()
            var = ['Lfmax', 'EMP2', 'EMP1', 'PD', 'nb', 'rm', 'fc', 'tb', 'intot',
                   'n', 'lai', 'w', 'wr', 'wc', 'p1', 'sla']
            for n, i in enumerate(firstline):
                plant[var[n]] = float(i)
        plant['int'] = 0.0
        plant['di'] = 0.0
        plant['wf'] = 0.0
        return plant


    @staticmethod
    def get_soil():
        """
        Get soil data from input file in relative path.
        Returns a dictionary obj as a class variable.
        """
        soil_file = "Data/Soil.inp"
        soil = {}
        with open(soil_file, 'r') as f:
            firstline = f.readline().split()
            var = f.readline().split()
            for n, i in enumerate(firstline):
                soil[var[n]] = float(i)
        return soil

    @staticmethod
    def get_weather():
        """
        Get weather data from input file in relative path.
        Returns a pandas Dataframe object as a class variable.
        """
        weather_file = "Data/weather.inp"
        tmp = []
        with open(weather_file, 'r') as f:
            for line in f:
                date, srad, tmax, tmin, rain, par = line.split()
                par = 0.5 * float(srad)  # as in Weather.for
                tmp.append([int(date), float(srad), float(tmax), float(tmin),
                            float(rain), float(par)])
        dlist = [(CustomFunctions.gen_datetimes(dateint[0])) for dateint in tmp]
        return pd.DataFrame(data=tmp, index=dlist, columns=['date', 'srad', 'tmax', 'tmin', 'rain', 'par'])


class Soil_Water(object):
    """
    Subroutines of SW
    These subroutine calculates the soil water availability for the plant,
    considering the rain, runoff, deep percolation (drainage) and water
    use by the plant (evapotranspiration). It is divided in subroutines
    that calculate those parameters separately. Daily data from climate
    comes from WEATHER and daily LAI from PLANT subroutines. SW supplies
    PLANT with daily soil water factor of availability (SWFAC)
    soil water availability for the plant,
    CALL SW(
    DOY, LAI, RAIN, SRAD, TMAX, TMIN,               !Input
    SWFAC1, SWFAC2,                                 !Output
    'INITIAL   ')                                   !Control
    """

    @staticmethod
    def draine(SWC, FC, DRNp):
        """
        DRAINE, calculates vertical drainage.
        Input:  SWC, FC, DRNp
        Output: DRN
        """
        if SWC > FC:
            DRN = (SWC - FC) * DRNp
        else:
            DRN = 0.0
        return DRN

    @staticmethod
    def ESaS(SWC, WP, FC, ESp):
        """
        Calculates the actual daily soil evaporation.
        Input:  SWC, WP, FC, ESp
        Output: ESa
        """
        if SWC < WP:
            a = 0.
        elif SWC < FC:
            a = 1.
        else:
            a = (SWC - WP) / (FC - WP)
        ESa = ESp * a
        return ESa

    @staticmethod
    def ETpS(LAI, TMAX, TMIN, SRAD):
        """
        Calculates the daily potential evapotranspiration.
        Input:  LAI, TMAX, TMIN, SRAD
        Output: ETp
        Local Variables
        ALB  =  ALBEDO OF CROP-SOIL SURFACE
        EEQ  =  EQUILIBRIUM EVAPOTRANSPIRATION (mm)
        Tmed =  ESTIMATED AVERAGE DAILY TEMPERATURE (C)
        f    =
        SUBROUTINE ETpS(SRAD,TMAX,TMIN,LAI,ETp)
        """
        ALB = 0.1 * np.exp(-0.7 * LAI) + 0.2 * (1 - np.exp(-0.7 * LAI))
        Tmed = 0.6 * TMAX + 0.4 * TMIN
        EEQ = SRAD * (4.88E-03 - 4.37E-03 * ALB) * (Tmed + 29)

        if TMAX < 5:
            f = 0.01 * np.exp(0.18 * (TMAX + 20.))
        elif TMAX > 35:
            f = 1.1 + 0.05 * (TMAX - 35.)
        else:
            f = 1.1
        ETp = f * EEQ
        return ETp

    @staticmethod
    def runoff(POTINF, CN):
        """
        SW subroutine RUNOFF calculates the daily runoff
        Input:  POTINF, CN, state(a string indicating flow required)
        Output: ROF
        Local Variables:
        CN = CURVE NUMBER SCS EQUATION
        S  = WATERSHED STORAGE SCS EQUATION (MM)
        """
        S = 254. * (100. / CN - 1.)  # do this. Else do the rest.
        if POTINF > 0.2 * S:
            ROF = ((POTINF - 0.2 * S) ** 2) / (POTINF + 0.8 * S)
        else:
            ROF = 0.0
        return ROF

    @staticmethod
    def stress(SWC, DP, FC, ST, WP, THE):
        """
        Sub-subroutine STRESS calculates soil water stresses.
        Today's stresses will be applied to tomorrow's rate calcs.
        Input:  SWC, DP, FC, ST, WP
        Output: SWFAC1, SWFAC2
        stress_depth is the water table dpth. (mm) below which no stress occurs
        THE is the threshold for drought stress (mm)
        Excess water stress factor - SWFAC2

        FC water is distributed evenly throughout soil profile.  Any
        water in excess of FC creates a free water surface
        WTABLE - thickness of water table (mm)
        DWT - depth to water table from surface (mm)
        The initial version of this program had two states, initial
        and integration. I moved the one line of intiail (which creates
        a value of THE) outside in the initilization of the sw_state dic.
        """
        stress_depth = 250.
        if SWC < WP:
            SWFAC1 = 0.0
        elif SWC > THE:
            SWFAC1 = 1.0
        else:
            SWFAC1 = (SWC - WP) / (THE - WP)
            SWFAC1 = max([min([SWFAC1, 1.0]), 0.0])  # ...this to restrain possible vals.
        if SWC <= FC:
            WTABLE = 0.0                            # Appears to be unused variable
            DWT = DP * 10.  # !DP in cm, DWT in mm  # Appears to be unused variable
            SWFAC2 = 1.0
        else:
            WTABLE = (SWC - FC) / (ST - FC) * DP * 10.
            DWT = DP * 10. - WTABLE
            if DWT > stress_depth:
                SWFAC2 = 1.0
            else:
                SWFAC2 = DWT / stress_depth
            SWFAC2 = max([min([SWFAC2, 1.0]), 0.0])
        return SWFAC1, SWFAC2


class PlantMethods(object):
    """
    PLANT
    These functions subroutine simulates the growth of the plant using pre-determined
    conditions.Hourly values of temperature and photosyntetically active
    radiation come from WEATHER subroutine and daily values of availability
    of water in the soil come from SW subroutine. This subroutine supplies
    the SW subroutine with daily values of leaf area index (LAI).
    SUBROUTINE PLANT(
    DOY, endsim,TMAX,TMIN, PAR, SWFAC1, SWFAC2,     !Input
    LAI,                                            !Output
    DYN)                                            !Control
    """

    @staticmethod
    def lais(FL, di, PD, EMP1, EMP2, N, nb, SWFAC1, SWFAC2, PT, dN,
             sla, p1):
        """
        Calculates the canopy leaf area index (LAI)
        Input:  FL, di, PD, EMP1, EMP2, N, nb, SWFAC1, SWFAC2, PT, dN
        Output: dLAI
        SUBROUTINE LAIS(FL,di,PD,EMP1,EMP2,N,nb,SWFAC1,SWFAC2,PT,
        &         dN,p1, sla, dLAI)
        REAL PD,EMP1,EMP2,N,nb,dLAI, SWFAC,a, dN, p1,sla
        REAL SWFAC1, SWFAC2, PT, di, FL
        """
        SWFAC = np.min([SWFAC1, SWFAC2])
        if FL == 1.0:
            a = np.exp(EMP2 * (N - nb))
            dLAI = SWFAC * PD * EMP1 * PT * (a / (1.0 + a)) * dN
        elif FL == 2.0:
            dLAI = - PD * di * p1 * sla
        return dLAI


    @staticmethod
    def PGS(SWFAC1, SWFAC2, PAR, PD, PT, LAI):
        """
        Calculates the canopy gross photosysntesis rate (PG)
        SUBROUTINE PGS(SWFAC1, SWFAC2,PAR, PD, PT, Lai, Pg)
        REAL PAR, Lai, Pg, PT, Y1
        REAL SWFAC1, SWFAC2, SWFAC,ROWSPC,PD
        ROWSP = row spacing
        Y1 = canopy light extinction coefficient
        """
        SWFAC = np.min([SWFAC1, SWFAC2])
        ROWSPC = 60.0
        Y1 = 1.5 - 0.768 * ((ROWSPC * 0.01) ** 2 * PD) ** 0.1
        Pg = PT * SWFAC * 2.1 * PAR / PD * (1.0 - np.exp(-Y1 * LAI))
        return Pg

    @staticmethod
    def PTS(TMAX, TMIN):
        """
        Calculates the factor that incorporates the effect of temperature
        on photosynthesis
        SUBROUTINE PTS(TMAX,TMIN,PT)
        REAL PT,TMAX,TMIN
        """
        PT = 1.0 - 0.0025 * ((0.25 * TMIN + 0.75 * TMAX) - 26.0) ** 2
        return PT


class CustomFunctions(object):
    """
    Custom functions
    """

    @staticmethod
    def gen_datetimes(tmpdint):
        """
        Given an integer of format YYDDD (decade and day-of-year)
        this will return a datetime object date.
        List comprehension can then be used to create lists of
        dates to be used as index's for time orderd data, e.g.:
        dlist = [(gen_datetimes(dateint)) for dateint in Object.irrigation.date]
        """
        yr = int(str(tmpdint)[0:2])
        doy = int(str(tmpdint)[2:])

        if yr > 50:
            yr += 1900  # Just make a guess to deal with this
        if yr < 50:
            yr += 2000  # irritaingly vage integer date scheme

        return dt.date(yr, 1, 1) + dt.timedelta(doy - 1)


    @staticmethod
    def get_variable_defs():
        """
        Calling this function will create a dictionary object attached to class
        that holds all variable names and definitions from Soil Water.
        (This is currently only the defs from the soil water routine.)
        """
        var_dic = {}
        var_files = ["Data/var_defs.txt"]
        for file in var_files:
            with open(file, 'r') as f:
                for line in f:
                    tmp = (line.split('='))
                    var_dic[str.strip(tmp[0])] = str.strip(tmp[1])
        return var_dic


class Model(object):
    """
    Functions to run the simulation, handels what was called the rate, and integration
    flow.
    """

    def soil_rate(self, n):
        """
        Soil rate section
        """
        if n in self.irrigation.index:  # If there is irrigation and weather data
            self.sw_state['TIRR'] += self.irrigation.irr[n]
            self.sw_state['POTINF'] = self.weather.rain[n] + self.irrigation.irr[n]
        else:  # If there is only weather data
            self.sw_state['POTINF'] = self.weather.rain[n]

        self.sw_state['TRAIN'] += self.weather.rain[n]
        self.sw_state['DRN'] = Soil_Water.draine(SWC=self.soil['SWC'],
                                                 FC=self.sw_state['FC'],
                                                 DRNp=self.soil['DRNp'])

        if self.sw_state['POTINF'] > 0.0:
            self.sw_state['ROF'] = Soil_Water.runoff(POTINF=self.sw_state['POTINF'],
                                                     CN=self.soil['CN'])
            self.sw_state['INF'] = self.sw_state['POTINF'] - self.sw_state['ROF']
        else:
            self.sw_state['ROF'] = 0.0
            self.sw_state['INF'] = 0.0

        # Pot. evapotranspiration (ETp), soil evaporation (ESp) and plant transpiration (EPp)
        self.sw_state['ETp'] = Soil_Water.ETpS(SRAD=self.weather.srad[n],
                                               TMAX=self.weather.tmax[n],
                                               TMIN=self.weather.tmin[n],
                                               LAI=self.plant['lai'])
        self.sw_state['ESp'] = self.sw_state['ETp'] * np.exp(-0.7 * self.plant['lai'])

        self.sw_state['EPp'] = self.sw_state['ETp'] * (1 - np.exp(-0.7 * self.plant['lai']))

        # Actual soil evaporation (ESa), plant transpiration (EPa)
        self.sw_state['ESa'] = Soil_Water.ESaS(ESp=self.sw_state['ESp'],
                                               SWC=self.soil['SWC'],
                                               FC=self.sw_state['FC'],
                                               WP=self.sw_state['WP'])

        self.sw_state['EPa'] = self.sw_state['EPp'] * np.min([self.sw_state['SWFAC1'],
                                                              self.sw_state['SWFAC2']])
        return


    def plant_rate(self, n):
        """
        Info here
        """
        self.plant['TMN'] = np.mean([self.weather['tmax'][n], self.weather['tmin'][n]])
        self.plant['PT'] = PlantMethods.PTS(TMAX=self.weather['tmax'][n],
                                            TMIN=self.weather['tmin'][n])

        self.plant['Pg'] = PlantMethods.PGS(SWFAC1=self.sw_state['SWFAC1'],
                                            SWFAC2=self.sw_state['SWFAC2'],
                                            PAR=self.weather['par'][n],
                                            PD=self.plant['PD'],
                                            PT=self.plant['PT'],
                                            LAI=self.plant['lai'])

        if self.plant['n'] < self.plant['Lfmax']:  # In the vegetative phase
            self.plant['FL'] = 1.0
            self.plant['E'] = 1.0
            self.plant['dN'] = self.plant['rm'] * self.plant['PT']
            self.plant['dLAI'] = PlantMethods.lais(FL=self.plant['FL'], di=self.plant['di'],
                                                   PD=self.plant['PD'], EMP1=self.plant['EMP1'],
                                                   EMP2=self.plant['EMP2'], N=self.plant['n'],
                                                   nb=self.plant['nb'], SWFAC1=self.sw_state['SWFAC1'],
                                                   SWFAC2=self.sw_state['SWFAC2'], PT=self.plant['PT'],
                                                   dN=self.plant['dN'], p1=self.plant['p1'],
                                                   sla=self.plant['sla'])
            self.plant['dw'] = self.plant['E'] * (self.plant['Pg']) * self.plant['PD']
            self.plant['dwc'] = self.plant['fc'] * self.plant['dw']
            self.plant['dwr'] = (1 - self.plant['fc']) * self.plant['dw']
            self.plant['dwf'] = 0.0
        else:  # In the reproductive plant phase...
            self.plant['FL'] = 2.0
            if (self.plant['TMN'] >= self.plant['tb']) and (self.plant['TMN'] <= 25.):
                self.plant['di'] = (self.plant['TMN'] - self.plant['tb'])
            else:
                self.plant['di'] = 0.0
            self.plant['int'] += self.plant['di']
            self.plant['E'] = 1.0
            self.plant['dLAI'] = PlantMethods.lais(FL=self.plant['FL'], di=self.plant['di'],
                                                   PD=self.plant['PD'], EMP1=self.plant['EMP1'],
                                                   EMP2=self.plant['EMP2'], N=self.plant['n'],
                                                   nb=self.plant['nb'], SWFAC1=self.sw_state['SWFAC1'],
                                                   SWFAC2=self.sw_state['SWFAC2'], PT=self.plant['PT'],
                                                   dN=self.plant['dN'], p1=self.plant['p1'],
                                                   sla=self.plant['sla'])
            self.plant['dw'] = self.plant['E'] * (self.plant['Pg']) * self.plant['PD']
            self.plant['dwf'] = self.plant['dw']
            self.plant['dwc'] = 0.0
            self.plant['dwr'] = 0.0
            self.plant['dN'] = 0.0
        return

    def sw_integrate(self):
        """
        Info
        """
        self.soil['SWC'] += (self.sw_state['INF'] - self.sw_state['ESa'] -
                             self.sw_state['EPa'] - self.sw_state['DRN'])
        if self.soil['SWC'] > self.sw_state['ST']:  # If wtr content > storage capacity
            self.sw_state['ROF'] += self.soil['SWC'] + self.sw_state['ST']  # then make runoff
        self.sw_state['TINF'] += self.sw_state['INF']
        self.sw_state['TESA'] += self.sw_state['ESa']
        self.sw_state['TEPA'] += self.sw_state['EPa']
        self.sw_state['TDRN'] += self.sw_state['DRN']
        self.sw_state['TROF'] += self.sw_state['ROF']
        self.sw_state['SWFAC1'], self.sw_state['SWFAC2'] = Soil_Water.stress(
            SWC=self.soil['SWC'], DP=self.soil['DP'],
            FC=self.sw_state['FC'], ST=self.sw_state['ST'],
            WP=self.sw_state['WP'], THE=self.sw_state['THE'])
        return


    def plant_integrate(self, doy):
        self.plant['lai'] += self.plant['dLAI']
        self.plant['w'] += self.plant['dw']
        self.plant['wc'] += self.plant['dwc']
        self.plant['wr'] += self.plant['dwr']
        self.plant['wf'] += self.plant['dwf']

        self.plant['lai'] = np.max(self.plant['lai'], 0.0)
        self.plant['w'] = np.max(self.plant['w'], 0.0)
        self.plant['wc'] = np.max(self.plant['wc'], 0.0)
        self.plant['wr'] = np.max(self.plant['wr'], 0.0)
        self.plant['wf'] = np.max(self.plant['wf'], 0.0)
        self.plant['n'] += self.plant['dN']

        if self.plant['int'] > self.plant['intot']:
            self.status['endsim'] = True
            print('The crop matured on day', doy)
            return


    def sw_output(self, n, doy, last_df):
        """
        In charge of output for SW data, creates or updates as DF object.
        """
        if n in self.irrigation.index:  # If there is irrigation data get it
            irrval = self.irrigation.irr[n]
        else:  # else set to 0
            irrval = 0
        tmpdat = [[doy,
                   self.weather.srad[n],
                   self.weather.tmax[n],
                   self.weather.tmin[n],
                   self.weather.rain[n],
                   irrval,
                   self.sw_state['ROF'],
                   self.sw_state['INF'],
                   self.sw_state['DRN'],
                   self.sw_state['ETp'],
                   self.sw_state['ESa'],
                   self.sw_state['EPa'],
                   self.soil['SWC'],
                   (self.soil['SWC'] / self.soil['DP']),
                   self.sw_state['SWFAC1'],
                   self.sw_state['SWFAC2']
                   ]]
        colnames = ['DOY', 'SRAD', 'TMAX', 'TMIN', 'RAIN', 'IRR', 'ROF',
                    'INF', 'DRN', 'ETP', 'ESa', 'EPa', 'SWC', 'SWC/DP', 'SWFAC1', 'SWFAC2']

        if self.status['initial']:  # If this is the first run then make the dataframe
            return pd.DataFrame(data=tmpdat, index=[n], columns=colnames)
        else:  # if it is not the first run, then update the dataframe
            dfupdate = pd.DataFrame(data=tmpdat, index=[n], columns=colnames)
            return last_df.append(dfupdate)


    def plant_output(self, doy, n, last_df):
        """
        Plant output subroutine
        """
        tmpdat = [[
                      doy,
                      self.plant['n'],
                      self.plant['int'],
                      self.plant['w'],
                      self.plant['wc'],
                      self.plant['wr'],
                      self.plant['wf'],
                      self.plant['lai']
                  ]]

        colnames = ['DOY', 'N', 'INT', 'W', 'Wc', 'Wr', 'Wf', 'LAI']
        if self.status['initial']:  # If this is the first run then make the dataframe
            return pd.DataFrame(data=tmpdat, index=[n], columns=colnames)
        else:  # if it is not the first run, then update the dataframe
            dfupdate = pd.DataFrame(data=tmpdat, index=[n], columns=colnames)
            return last_df.append(dfupdate)


    def run_model(self):
        """
        Running rate and integration of Soil Water and Plant
        DOYP - the date of planting has been added to the self.plant['doyp']
        """
        # note below, diffrent years give diffrent maturation date [subscript to new year]
        # alter the input in some way so I can feed it one year at a time only...
        for n in np.sort(self.weather.index):  # for each time-step (n)
            if not self.status['endsim']:
                # print(n,self.status['endsim'])
                doy = n.timetuple()[7]  # get the DOY
                # ---- Rate calculations ---
                Model.soil_rate(self, n=n)
                # if doy > doyp:         # Calc plant if doy is after a planting doy
                if doy > self.plant['doyp']:
                    Model.plant_rate(self, n=n)
                # --- Integration step --- (i.e. update state dictionaries each time step)
                Model.sw_integrate(self, )
                if doy > self.plant['doyp']:
                    Model.plant_integrate(self, doy=doy)
                # --- Output stage ---
                if self.status['initial']:
                    sw_out = Model.sw_output(self, n=n, doy=doy, last_df=None)
                    p_out = Model.plant_output(self, n=n, doy=doy, last_df=None)
                    self.status['initial'] = False
                else:
                    sw_out = Model.sw_output(self, n=n, doy=doy, last_df=sw_out)
                    p_out = Model.plant_output(self, n=n, doy=doy, last_df=p_out)
        print('Simulation finished')
        return sw_out, p_out
