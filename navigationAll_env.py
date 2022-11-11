#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:07:00 2021

@author: yusheng
"""
# ------------------------------------------------------------------------
# required pacckages: NumPy, SciPy, openAI gym
# written in the framwork of gym
# ------------------------------------------------------------------------
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
# from os import path
from scipy import integrate
import CFDfunctions as cf
# import time as timer
# from colorama import Fore, Back, Style
import importlib
class DipoleSingleEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self,  paramSource = 'envParam_default'):
        param = importlib.import_module('settings.'+paramSource)
        # size and parameters
        self.speed = param.mu
        self.bl = param.bl
        self.bw = param.bw
        self.dt = param.dt
        self.flex = param.flexibility
        self.cfdpath = param.cfdpath

        self.region_offset = param.train_offset
        flow_dict = {
            'CFD': self.__flowVK_CFD,
            'reduced': self.__flowVK_reduced,
            'CFDwRot': self.__flowVK_CFDWR,
            'CFD_flipped': self.__flowVK_CFD_flipped
            }
        obs_dict = {
            'geoOneSensor': (self._get_obs_geo, 5),
            'geoFlowBlind': (self._get_obs_geoflowblind, 3),
            'egoOneSensor': (self._get_obs_ego1, 4),
            'egoOneSensorVor': (self._get_obs_ego1vor, 5),
            'egoLRGradMemory': (self._get_obs_ego2lrgradmemory, 12),
            'egoTwoSensorLR': (self._get_obs_ego2lr, 6),
            'egoTwoSensorLRMagGrad': (self._get_obs_ego2lrmaggrad, 5),
            'egoTwoSensorLRGradOnly': (self._get_obs_ego2lrgradonly, 4),
            'egoTwoSensorLRGradOnlyMemory': (self._get_obs_ego2lrgradonlymemory, 6),
            'egoTwoSensorLRSingleGrad': (self._get_obs_ego2lrsinglegrad, 5),
            'egoTwoSensorLRSingle2Grad': (self._get_obs_ego2lrsingle2grad, 5),
            'egoFourSensorGradOnly': (self._get_obs_ego4gradonly, 6),
            'egoTwoSensorLRGrad': (self._get_obs_ego2lrgrad, 6),
            'egoCustomize': (self._get_obs_egocustomize, 6),
            # 'egoLRGradTrick': (self._get_obs_ego2lrgradtrick, 7),
            # 'geoPeriodic': (self._get_obs_geo_all, 6),
            # 'geoLRGradOnly': (self._get_obs_geolrgradonly, 5),
            # 'egoLRGradVor': (self._get_obs_ego2lrgradvor, 7), #not working
            # 'egoOneSensorPlusOrt': (self._get_obs_ego1ort, 5),
            # 'egoTwoSensorFB': (self._get_obs_ego2fb, 6),
            # 'egoLRGradNoX': (self._get_obs_egolrgradnox, 5), #not working
            # 'egoTwoSensorLRGradDir': (self._get_obs_ego2lrgraddir, 7),
            # 'egoTwoSensorLRMagGrads': (self._get_obs_ego2lrmaggrads, 7),
            # 'egoDirOnlyLRGrad': (self._get_obs_egodirlrgrad, 6),
            # 'egoLRGradFlowDirOnly': (self._get_obs_ego2lrgradflowdironly, 5),
            # 'egoLRGradOnlyFullMemory': (self._get_obs_ego2lrgradonlyfullmemory, 8),
            # 'egoTwoSensorFBGrad': (self._get_obs_ego2fbgrad, 6),
            # 'egoTwoSensorLRDir': (self._get_obs_ego2lrdir, 5),
            # 'egoOneSensorDir': (self._get_obs_ego1dir, 3)
            }
        self.mode = param.flowMode
        self.flow = flow_dict[param.flowMode]
        self.obs, obs_num = obs_dict[param.obsMode]
        if hasattr(param, 'resetMode'):
            self.reset_mode = param.resetMode
        else:
            self.reset_mode = "default"
        if ('CFD' in self.mode):
            self.permittedL = param.cfdDomainL
            self.permittedR = param.cfdDomainR
            self.permittedU = param.cfdDomainU
            self.permittedD = param.cfdDomainD
            time_span = param.time_span
            level_limit = param.level_limit
            source_path = self.cfdpath + "np/"
            print('begin reading CFD data')
            self.cfd_framerate,self.time_span,\
                self.UUU,self.VVV,self.OOO,self.XMIN,self.XMAX,self.YMIN,self.YMAX\
                = cf.adapt_load_data(time_span,source_path,level_limit)
            # print('LOADING')
            # print(time_span, source_path, level_limit)
            print('finished reading CFD data')
        elif (self.mode == 'reduced'):
            self.permittedL = param.reducedDomainL
            self.permittedR = param.reducedDomainR
            self.permittedU = param.reducedDomainU
            self.permittedD = param.reducedDomainD
            self.A = param.A
            self.lam = param.lam
            self.Gamma = param.Gamma
            self.bgflow = param.bgflow
            self.cut = param.cut
            # self.time_span = 
        self.trail = []
        self.target = np.zeros((2,))

        high = np.array([np.finfo(np.float32).max] * obs_num)
        low = np.array([np.finfo(np.float32).min] * obs_num)
        # create the observation space and the action space
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.double)
        self.action_space = spaces.Box(low = -1., high = 1., shape = (1,), dtype = np.double)
        
        self.viewer = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # self.time = 0
        terminal = False
        dt = self.dt
        # compute the fish nose position
        disToTarget_old = np.sqrt((self.pos[0]-self.target[0])**2+(self.pos[1]-self.target[1])**2)

        # integrate the dynamics system
        ang_vel = action[0]*self.flex*2*self.speed/self.bw
        # def reach(t, y):
        #     return np.sqrt((y[0]-self.target[0])**2+(y[1]-self.target[1])**2)-0.2
        # reach.terminal = True
        options = {'rtol':1e-4,'atol':1e-8,'max_step': 1e-2}

        sol = integrate.solve_ivp(lambda t,y: self.__firstorderdt(t,y,ang_vel), [self.time,self.time+dt],
                            self.pos, method = 'RK45',events = None, vectorized=False,
                            dense_output = False, **options)
        self.pos = sol.y[:,-1]
        # update the time
        self.time = self.time + dt
        disToTarget_new = np.sqrt((self.pos[0]-self.target[0])**2+(self.pos[1]-self.target[1])**2)
        
        dDisToTarget = disToTarget_new - disToTarget_old
        
        # terminal = self.__terminal()          # termination condition
        reward = 0
        reward += -dDisToTarget
        if disToTarget_new<0.15:
            reward += 200
            terminal = True
        if self.pos[0]>self.permittedR or self.pos[0]<self.permittedL or self.pos[1]<self.permittedD or self.pos[1]>self.permittedU:
            terminal = True
        return self.obs(), reward, terminal, {}
    def __firstorderdt(self,t,pos,ang_vel):
        u = np.cos(pos[2])*self.speed
        v = np.sin(pos[2])*self.speed
        # angular velocity induced by strength difference
        w = ang_vel
        uVK,vVK,oVK = self.flow((pos[0],pos[1]), t)
        vel = np.array([u+uVK, v+vVK, w+oVK/2])
        # vel = np.array([uVK, vVK, oVK/2])
        self.vel = vel
        return vel
    """CFD wake"""
    def __flowVK_CFD(self, pos, t):
        x = pos[0]
        y = pos[1]
        
        """fixed or adaptive mesh"""
        uVK,vVK,oVK =  cf.adapt_time_interp(self.UUU,self.VVV,self.OOO,self.XMIN,self.XMAX,self.YMIN,self.YMAX,\
                                            self.cfd_framerate,time = t % self.time_span,posX = x,posY = y)

        oVK = 0
        return uVK,vVK,oVK
    def __flowVK_CFD_flipped(self, pos, t):
        x = -pos[0]
        y = -pos[1]
        
        """fixed or adaptive mesh"""
        uVK,vVK,oVK =  cf.adapt_time_interp(self.UUU,self.VVV,self.OOO,self.XMIN,self.XMAX,self.YMIN,self.YMAX,\
                                            self.cfd_framerate,time = t % self.time_span,posX = x,posY = y)

        uVK = -uVK
        vVK = -vVK
        oVK = 0
        return uVK,vVK,oVK
    def __flowVK_CFDWR(self, pos, t):
        x = pos[0]
        y = pos[1]
        """fixed or adaptive mesh"""
        uVK,vVK,oVK =  cf.adapt_time_interp(self.UUU,self.VVV,self.OOO,self.XMIN,self.XMAX,self.YMIN,self.YMAX,\
                                            self.cfd_framerate,time = t % self.time_span,posX = x,posY = y)
        return uVK,vVK,oVK
    """reduced order"""
    def __flowVK_reduced(self, pos, t):
        A = self.A
        Gamma = self.Gamma
        lam = self.lam
        z = pos[0]+1j*pos[1]
        U = Gamma/2/lam*np.tanh(2*np.pi*A/lam)+self.bgflow
        # print("lam",lam,"z",z,"A",A,"t",t,"U",U, "tan", np.pi*(z + 1j*A - t*U)/lam)
        if np.abs(z + 1j*A - t*U) > 0 and np.abs(z-lam/2-1j*A - t*U) > 0:
            wVK = 1j*Gamma/2/lam*(1/np.tan(np.pi*(z + 1j*A - t*U)/lam) - 1/np.tan(np.pi*(z-lam/2-1j*A - t*U)/lam))
        else:
            wVK = 0
        uVK = np.real(wVK)
        vVK = -np.imag(wVK)
        if uVK>self.cut:
            uVK = self.cut
        elif uVK<-self.cut:
            uVK = -self.cut
        if vVK>self.cut:
            vVK = self.cut
        elif vVK<-self.cut:
            vVK = -self.cut
        uVK += self.bgflow
        oVK = 0
        return uVK,vVK,oVK
    def __initialConfig(self,mode, init_num):
        # get the initial configuration of the fish
        ####################circular zone#######################
        r = 2*np.sqrt(np.random.rand())
        the = np.random.rand()*2*np.pi
        center = (self.permittedR + self.permittedL)/2
        X, Y, theta = center+np.cos(the)*r, -(2+self.region_offset)+np.sin(the)*r, np.random.rand()*2*np.pi
        ########################################################
        ############square zone###############
        # X = (self.permittedR + 3*self.permittedL)/4+np.random.rand()*(self.permittedR - self.permittedL)/2
        # Y = -self.region_offset + np.random.rand()*self.permittedD/2
        # theta = np.random.rand()*2*np.pi
        #################################################
        return np.array([X, Y, theta])
    def __prescribedControl(self,time, position = None):
        # Used when manually precribing the control (disregarding the RL policy, mainly for test)
        """swimming"""
        return 0
    def reset(self, position = None, target = None, UpDown = 1, init_time = None):   # reset the environment setting
        # print(Fore.RED + 'RESET ENVIRONMENT')
        # print(Style.RESET_ALL)
        
        # if (self.mode == 'reduced'):
        #     UpDown = np.random.choice([1,-1])
        # UpDown = -1
        self.obsmemory = []
        
        if self.reset_mode == "default":
            swimmer_centerX = target_centerX = (self.permittedR + self.permittedL)/2
            swimmer_centerY = -UpDown*(2+self.region_offset)
            target_centerY = UpDown*(2+self.region_offset)
            # self.pos = self.__initialConfig(1, 1)
            # get the initial configuration of the fish
            ####################circular zone#######################
            r = 2*np.sqrt(np.random.rand())
            the = np.random.rand()*2*np.pi
            self.pos = [swimmer_centerX + np.cos(the)*r, swimmer_centerY + np.sin(the)*r, np.random.rand()*2*np.pi]
            r = 2*np.sqrt(np.random.rand())
            the = np.random.rand()*2*np.pi
            self.set_target(target_centerX + np.cos(the)*r, target_centerY + np.sin(the)*r)
            ########################################################
            ############square zone###############
            # X = (self.permittedR + 3*self.permittedL)/4+np.random.rand()*(self.permittedR - self.permittedL)/2
            # Y = -self.region_offset + np.random.rand()*self.permittedD/2
            # theta = np.random.rand()*2*np.pi
            #################################################
        elif self.reset_mode == "InsideWake":
            swimmer_centerX = (self.permittedR + self.permittedL)/2 - 1.5
            swimmer_centerY = 0
            self.pos = [swimmer_centerX + (np.random.rand()-0.5)*4, swimmer_centerY + (np.random.rand()-0.5)*4, np.random.rand()*2*np.pi]

            target_centerX = (self.permittedR + self.permittedL)/2 + 1.5
            target_centerY = 0
            self.set_target(target_centerX + (np.random.rand()-0.5)*4, target_centerY + (np.random.rand()-0.5)*1)
        elif self.reset_mode == "SameSideUp":
            swimmer_centerX = (self.permittedR + self.permittedL)/2 - 1
            target_centerX = (self.permittedR + self.permittedL)/2 + 1
            swimmer_centerY = UpDown*(2+self.region_offset)
            target_centerY = UpDown*(2+self.region_offset)
            r = 2*np.sqrt(np.random.rand())
            the = np.random.rand()*2*np.pi
            self.pos = [swimmer_centerX + np.cos(the)*r, swimmer_centerY + np.sin(the)*r, np.random.rand()*2*np.pi]
            r = 2*np.sqrt(np.random.rand())
            the = np.random.rand()*2*np.pi
            self.set_target(target_centerX + np.cos(the)*r, target_centerY + np.sin(the)*r)
        elif self.reset_mode == "SquareTwoSide":
            swimmer_centerX = (self.permittedR + self.permittedL)/2 - 1
            swimmer_centerY = 0
            self.pos = [swimmer_centerX + (np.random.rand()-0.5)*4, swimmer_centerY + (np.random.rand()-0.5)*6, np.random.rand()*2*np.pi]
        
            target_centerX = (self.permittedR + self.permittedL)/2 + 1
            target_centerY = 0
            self.set_target(target_centerX + (np.random.rand()-0.5)*4, target_centerY + (np.random.rand()-0.5)*6)
        elif self.reset_mode == "Customize":
            UpDown = np.random.choice([1,-1])
            swimmer_centerX = (self.permittedR + self.permittedL)/2 - 1
            swimmer_centerY = 0
            self.pos = [swimmer_centerX + (np.random.rand()-0.5)*4, swimmer_centerY + (np.random.rand()-0.5)*6, np.random.rand()*2*np.pi]
            # swimmer_centerY = -UpDown*(2+self.region_offset)
            # r = 2*np.sqrt(np.random.rand())
            # the = np.random.rand()*2*np.pi
            # self.pos = [swimmer_centerX + np.cos(the)*r, swimmer_centerY + np.sin(the)*r, np.random.rand()*2*np.pi]
        
            target_centerX = (self.permittedR + self.permittedL)/2 + 1
            target_centerY = UpDown*(2+self.region_offset)
            r = 1.8*np.sqrt(np.random.rand())
            the = np.random.rand()*2*np.pi
            self.set_target(target_centerX + np.cos(the)*r, target_centerY + np.sin(the)*r)
            # target_centerY = 0
            # self.set_target(target_centerX + (np.random.rand()-0.5)*4, target_centerY + (np.random.rand()-0.5)*6)
        elif self.reset_mode == "Everywhere":            
            swimmer_centerX = (self.permittedR + self.permittedL)/2
            swimmer_centerY = 0
            self.pos = [swimmer_centerX + (np.random.rand()-0.5)*4, swimmer_centerY + (np.random.rand()-0.5)*6, np.random.rand()*2*np.pi]

            target_centerX = (self.permittedR + self.permittedL)/2

            target_centerY = 0
            self.set_target(target_centerX + (np.random.rand()-0.5)*4, target_centerY + (np.random.rand()-0.5)*6)
        """When init position/ target position is specifically given"""
        if position is not None:
            self.pos = position
        if target is not None:
            self.set_target(target[0],target[1])
        if init_time is not None:
            self.time = init_time
        else:
            self.time = np.random.rand()*self.time_span
        """"""
        self.vel = np.zeros_like(self.pos)
        self.trail = []
        ort = self.pos[-1]
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        self.oldrelx = dx*np.cos(ort) + dy*np.sin(ort)
        self.oldrely = -dx*np.sin(ort) + dy*np.cos(ort)

        return self.obs()
    """A series of observations to choose from"""
    def _get_obs_geo(self):
        """
        swimmer position, orientation and local flow field in lab frame
        """
        uVK,vVK,oVK = self.flow(self.pos[0:2], self.time)
        ort = angle_normalize(self.pos[-1])
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]
        return np.array([dx,dy,ort,uVK,vVK])
    # def _get_obs_geolrgradonly(self):
    #     """
    #     egocentric swimmer position and local flow field and lateral gradient
    #     """
    #     ort = angle_normalize(self.pos[-1])
    #     posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
    #     posRight = np.array(self.pos[0:2])*2 - posLeft
    #     uVKL,vVKL,_ = self.flow(posLeft,self.time)
    #     uVKR,vVKR,_ = self.flow(posRight,self.time)
        
    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]

    #     relx = dx*np.cos(ort) + dy*np.sin(ort)
    #     rely = -dx*np.sin(ort) + dy*np.cos(ort)
    #     reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
    #     relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
    #     reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
    #     relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
    #     return np.array([relx, rely, ort, (reluL - reluR)/0.1, (relvL - relvR)/0.1])
    # def _get_obs_geo_all(self):
    #     """
    #     swimmer position, orientation and local flow field in lab frame
    #     """
    #     uVK,vVK,oVK = self.flow(self.pos[0:2], self.time)
    #     ort = angle_normalize(self.pos[-1])
    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]
    #     return np.array([dx,dy,np.cos(ort), np.sin(ort),uVK,vVK])

    # def _get_obs_ego1dir(self):
    #     """
    #     egocentric swimmer position and local flow field
    #     """
    #     uVK,vVK,oVK = self.flow(self.pos[0:2], self.time)
    #     ort = angle_normalize(self.pos[-1])
    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]

    #     target_angle = np.arctan2(dy,dx)
    #     relort = angle_normalize(target_angle - ort)

    #     relu = uVK*np.cos(ort) + vVK*np.sin(ort)
    #     relv = -uVK*np.sin(ort) + vVK*np.cos(ort)
    #     return np.array([relort,relu,relv])
    def _get_obs_ego1(self):
        """
        egocentric swimmer angular position and local flow field
        """
        uVK,vVK,oVK = self.flow(self.pos[0:2], self.time)
        ort = angle_normalize(self.pos[-1])
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        relx = dx*np.cos(ort) + dy*np.sin(ort)
        rely = -dx*np.sin(ort) + dy*np.cos(ort)
        relu = uVK*np.cos(ort) + vVK*np.sin(ort)
        relv = -uVK*np.sin(ort) + vVK*np.cos(ort)
        return np.array([relx,rely,relu,relv])
    def _get_obs_geoflowblind(self):
        """
        egocentric swimmer angular position and local flow field
        """
        ort = angle_normalize(self.pos[-1])
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        relx = dx*np.cos(ort) + dy*np.sin(ort)
        rely = -dx*np.sin(ort) + dy*np.cos(ort)
        return np.array([relx,rely,ort])
    def _get_obs_ego1vor(self):
        """
        egocentric swimmer position and local flow field
        """
        uVK,vVK,oVK = self.__flowVK_CFDWR(self.pos[0:2], self.time)
        ort = angle_normalize(self.pos[-1])
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        relx = dx*np.cos(ort) + dy*np.sin(ort)
        rely = -dx*np.sin(ort) + dy*np.cos(ort)
        relu = uVK*np.cos(ort) + vVK*np.sin(ort)
        relv = -uVK*np.sin(ort) + vVK*np.cos(ort)
        return np.array([relx,rely,relu,relv,oVK])
    # def _get_obs_ego1ort(self):
    #     """
    #     egocentric swimmer position, local flow field, plus global orientation
    #     """
    #     uVK,vVK,oVK = self.flow(self.pos[0:2], self.time)
    #     ort = angle_normalize(self.pos[-1])
    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]

    #     relx = dx*np.cos(ort) + dy*np.sin(ort)
    #     rely = -dx*np.sin(ort) + dy*np.cos(ort)
    #     relu = uVK*np.cos(ort) + vVK*np.sin(ort)
    #     relv = -uVK*np.sin(ort) + vVK*np.cos(ort)
    #     return np.array([relx,rely,relu,relv,ort])
    def _get_obs_ego2lr(self):
        """
        egocentric swimmer position and local flow fields from 2 sensors located on the left and right of the swimmer
        """
        ort = angle_normalize(self.pos[-1])
        # posLeft = self.pos[0:2] + np.array([-self.bw/2*np.sin(self.pos[-1]),self.bw/2*np.cos(self.pos[-1])])
        posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
        posRight = np.array(self.pos[0:2])*2 - posLeft
        uVKL,vVKL,_ = self.flow(posLeft,self.time)
        uVKR,vVKR,_ = self.flow(posRight,self.time)

        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        relx = dx*np.cos(ort) + dy*np.sin(ort)
        rely = -dx*np.sin(ort) + dy*np.cos(ort)
        reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
        relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
        reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
        relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
        return np.array([relx,rely,reluL,relvL,reluR,relvR])
    # def _get_obs_ego2lrdir(self):
    #     """
    #     egocentric swimmer angular position and local flow fields from 2 sensors located on the left and right of the swimmer
    #     """
    #     ort = self.pos[-1]
    #     # ort = angle_normalize(self.pos[-1])
    #     # posLeft = self.pos[0:2] + np.array([-self.bw/2*np.sin(self.pos[-1]),self.bw/2*np.cos(self.pos[-1])])
    #     posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
    #     posRight = np.array(self.pos[0:2])*2 - posLeft
    #     uVKL,vVKL,_ = self.flow(posLeft,self.time)
    #     uVKR,vVKR,_ = self.flow(posRight,self.time)

    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]
    #     target_angle = np.arctan2(dy,dx)
    #     relort = angle_normalize(target_angle - ort)

    #     reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
    #     relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
    #     reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
    #     relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
    #     return np.array([relort,reluL,relvL,reluR,relvR])
    def _get_obs_ego2lrgrad(self):
        """
        egocentric swimmer position and local flow field and lateral gradient
        """
        ort = angle_normalize(self.pos[-1])
        posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
        posRight = np.array(self.pos[0:2])*2 - posLeft
        uVKL,vVKL,_ = self.flow(posLeft,self.time)
        uVKR,vVKR,_ = self.flow(posRight,self.time)
        
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        relx = dx*np.cos(ort) + dy*np.sin(ort)
        rely = -dx*np.sin(ort) + dy*np.cos(ort)
        reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
        relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
        reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
        relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
        
        return np.array([relx, rely, (reluL + reluR)/2, (relvL + relvR)/2, (reluL - reluR)/0.1, (relvL - relvR)/0.1])
    # def _get_obs_ego2lrgraddir(self):
    #     """
    #     egocentric swimmer position and local flow field and lateral gradient
    #     """
    #     ort = angle_normalize(self.pos[-1])
    #     posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
    #     posRight = np.array(self.pos[0:2])*2 - posLeft
    #     uVKL,vVKL,_ = self.flow(posLeft,self.time)
    #     uVKR,vVKR,_ = self.flow(posRight,self.time)
        
    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]

    #     relx = dx*np.cos(ort) + dy*np.sin(ort)
    #     rely = -dx*np.sin(ort) + dy*np.cos(ort)
    #     reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
    #     relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
    #     reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
    #     relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
    #     return np.array([relx, rely, ort, (reluL + reluR)/2, (relvL + relvR)/2, (reluL - reluR)/0.1, (relvL - relvR)/0.1])
    # def _get_obs_ego2lrmaggrads(self):
    #     """
    #     egocentric swimmer position and local flow field and lateral gradient
    #     """
    #     ort = angle_normalize(self.pos[-1])
    #     posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
    #     posRight = np.array(self.pos[0:2])*2 - posLeft
    #     uVKL,vVKL,_ = self.flow(posLeft,self.time)
    #     uVKR,vVKR,_ = self.flow(posRight,self.time)
        
    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]

    #     relx = dx*np.cos(ort) + dy*np.sin(ort)
    #     rely = -dx*np.sin(ort) + dy*np.cos(ort)
    #     reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
    #     relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
    #     reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
    #     relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
    #     UL = np.linalg.norm([uVKL, vVKL])
    #     UR = np.linalg.norm([uVKR, vVKR])
    #     return np.array([relx, rely, (reluL + reluR)/2, (relvL + relvR)/2, (reluL - reluR)/0.1, (relvL - relvR)/0.1, (UL - UR)/0.1])
    # def _get_obs_ego2lrgradtrick(self):
    #     """
    #     egocentric swimmer position and local flow field and lateral gradient
    #     """
    #     ort = angle_normalize(self.pos[-1])
    #     posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
    #     posRight = np.array(self.pos[0:2])*2 - posLeft
    #     uVKL,vVKL,_ = self.flow(posLeft,self.time)
    #     uVKR,vVKR,_ = self.flow(posRight,self.time)
        
    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]

    #     relx = dx*np.cos(ort) + dy*np.sin(ort)
    #     rely = -dx*np.sin(ort) + dy*np.cos(ort)
    #     reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
    #     relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
    #     reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
    #     relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
    #     return np.array([relx, rely, (reluL + reluR)/2, (relvL + relvR)/2, (reluL - reluR)/0.1, (relvL - relvR)/0.1, np.sign(self.target[1])])
    def _get_obs_ego2lrgradmemory(self):
        """
        egocentric swimmer position and local flow field and lateral gradient with memory of last observation
        """
        ort = angle_normalize(self.pos[-1])
        posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
        posRight = np.array(self.pos[0:2])*2 - posLeft
        uVKL,vVKL,_ = self.flow(posLeft,self.time)
        uVKR,vVKR,_ = self.flow(posRight,self.time)
        
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        relx = dx*np.cos(ort) + dy*np.sin(ort)
        rely = -dx*np.sin(ort) + dy*np.cos(ort)
        reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
        relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
        reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
        relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)

        obs = np.array([relx, rely, (reluL + reluR)/2, (relvL + relvR)/2, (reluL - reluR)/0.1, (relvL - relvR)/0.1])
        if len(self.obsmemory) == 6:
            obsfull = np.concatenate([self.obsmemory, obs])
        else:
            obsfull = np.concatenate([obs, obs])
        self.obsmemory = obs
        return obsfull
    # def _get_obs_ego2lrgradvor(self):
    #     """
    #     egocentric swimmer position and local flow field and lateral gradient
    #     """
    #     ort = angle_normalize(self.pos[-1])
    #     posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
    #     posRight = np.array(self.pos[0:2])*2 - posLeft
    #     uVKL,vVKL,oL = self.flow(posLeft,self.time)
    #     uVKR,vVKR,oR = self.flow(posRight,self.time)
        
    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]

    #     relx = dx*np.cos(ort) + dy*np.sin(ort)
    #     rely = -dx*np.sin(ort) + dy*np.cos(ort)
    #     reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
    #     relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
    #     reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
    #     relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
    #     return np.array([relx, rely, (reluL + reluR)/2, (relvL + relvR)/2, (reluL - reluR)/0.1, (relvL - relvR)/0.1, (oL + oR)/2])
        
    # def _get_obs_egolrgradnox(self):
    #     """
    #     egocentric swimmer position and local flow field and lateral gradient
    #     """
    #     ort = angle_normalize(self.pos[-1])
    #     posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
    #     posRight = np.array(self.pos[0:2])*2 - posLeft
    #     uVKL,vVKL,_ = self.flow(posLeft,self.time)
    #     uVKR,vVKR,_ = self.flow(posRight,self.time)
        
    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]

    #     # relx = dx*np.cos(ort) + dy*np.sin(ort)
    #     rely = -dx*np.sin(ort) + dy*np.cos(ort)
    #     reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
    #     relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
    #     reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
    #     relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
    #     return np.array([rely, (reluL + reluR)/2, (relvL + relvR)/2, (reluL - reluR)/0.1, (relvL - relvR)/0.1])
    # def _get_obs_egodirlrgrad(self):
    #     """
    #     egocentric swimmer position, local flow direction and lateral gradient
    #     """
    #     ort = angle_normalize(self.pos[-1])
    #     posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
    #     posRight = np.array(self.pos[0:2])*2 - posLeft
    #     uVKL,vVKL,_ = self.flow(posLeft,self.time)
    #     uVKR,vVKR,_ = self.flow(posRight,self.time)
        
    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]

    #     relx = dx*np.cos(ort) + dy*np.sin(ort)
    #     rely = -dx*np.sin(ort) + dy*np.cos(ort)
    #     reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
    #     relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
    #     reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
    #     relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
    #     cosdir = (reluL + reluR)/np.sqrt((reluL + reluR)**2 + (relvL + relvR) **2)
    #     sindir = (relvL + relvR)/np.sqrt((reluL + reluR)**2 + (relvL + relvR) **2)
    #     return np.array([relx, rely, cosdir, sindir, (reluL - reluR)/0.1, (relvL - relvR)/0.1])
    # def _get_obs_ego2lrgradflowdironly(self):
    #     """
    #     egocentric swimmer position and local flow field and lateral gradient
    #     """
    #     ort = angle_normalize(self.pos[-1])
    #     posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
    #     posRight = np.array(self.pos[0:2])*2 - posLeft
    #     uVKL,vVKL,_ = self.flow(posLeft,self.time)
    #     uVKR,vVKR,_ = self.flow(posRight,self.time)
        
    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]

    #     relx = dx*np.cos(ort) + dy*np.sin(ort)
    #     rely = -dx*np.sin(ort) + dy*np.cos(ort)
    #     reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
    #     relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
    #     reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
    #     relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
    #     fdir = np.arctan2(relvL + relvR, reluL + reluR)
    #     return np.array([relx, rely, fdir, (reluL - reluR)/0.1, (relvL - relvR)/0.1])
    def _get_obs_ego2lrmaggrad(self):
        """
        egocentric swimmer position and local flow field and lateral gradient
        """
        ort = angle_normalize(self.pos[-1])
        posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
        posRight = np.array(self.pos[0:2])*2 - posLeft
        uVKL,vVKL,_ = self.flow(posLeft,self.time)
        uVKR,vVKR,_ = self.flow(posRight,self.time)
        
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]
        
        relx = dx*np.cos(ort) + dy*np.sin(ort)
        rely = -dx*np.sin(ort) + dy*np.cos(ort)
        reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
        relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
        reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
        relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
        UL = np.linalg.norm([uVKL, vVKL])
        UR = np.linalg.norm([uVKR, vVKR])
        return np.array([relx, rely, (reluL + reluR)/2, (relvL + relvR)/2, (UL - UR)/0.1])
    def _get_obs_ego2lrgradonly(self):
        """
        egocentric swimmer position and local flow field and lateral gradient
        """
        ort = angle_normalize(self.pos[-1])
        posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
        posRight = np.array(self.pos[0:2])*2 - posLeft
        uVKL,vVKL,_ = self.flow(posLeft,self.time)
        uVKR,vVKR,_ = self.flow(posRight,self.time)
        
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        relx = dx*np.cos(ort) + dy*np.sin(ort)
        rely = -dx*np.sin(ort) + dy*np.cos(ort)
        reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
        relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
        reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
        relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
        return np.array([relx, rely, (reluL - reluR)/0.1, (relvL - relvR)/0.1])
    
    # def _get_obs_ego2lrgradonlyfullmemory(self):
    #     """
    #     egocentric swimmer position and local flow field and lateral gradient
    #     """
    #     ort = angle_normalize(self.pos[-1])
    #     posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
    #     posRight = np.array(self.pos[0:2])*2 - posLeft
    #     uVKL,vVKL,_ = self.flow(posLeft,self.time)
    #     uVKR,vVKR,_ = self.flow(posRight,self.time)
        
    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]

    #     relx = dx*np.cos(ort) + dy*np.sin(ort)
    #     rely = -dx*np.sin(ort) + dy*np.cos(ort)
    #     reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
    #     relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
    #     reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
    #     relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)

    #     obs = np.array([relx, rely, (reluL - reluR)/0.1, (relvL - relvR)/0.1])
    #     if len(self.obsmemory) == 4:
    #         obsfull = np.concatenate([self.obsmemory, obs])
    #     else:
    #         obsfull = np.concatenate([obs, obs])
    #     self.obsmemory = obs
    #     return obsfull
    def _get_obs_ego2lrgradonlymemory(self):
        """
        egocentric swimmer position and local flow field and lateral gradient
        """
        ort = angle_normalize(self.pos[-1])
        posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
        posRight = np.array(self.pos[0:2])*2 - posLeft
        uVKL,vVKL,_ = self.flow(posLeft,self.time)
        uVKR,vVKR,_ = self.flow(posRight,self.time)
        
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        relx = dx*np.cos(ort) + dy*np.sin(ort)
        rely = -dx*np.sin(ort) + dy*np.cos(ort)
        
        reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
        relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
        reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
        relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
        oldrelx = self.oldrelx
        oldrely = self.oldrely
        obs = np.array([relx-oldrelx, rely-oldrely, relx, rely, (reluL - reluR)/0.1, (relvL - relvR)/0.1])
        self.oldrelx = relx
        self.oldrely = rely
        return obs
    def _get_obs_ego2lrsinglegrad(self):
        """
        egocentric swimmer position and local flow field and lateral gradient in lateral direction
        """
        ort = angle_normalize(self.pos[-1])
        posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
        posRight = np.array(self.pos[0:2])*2 - posLeft
        uVKL,vVKL,_ = self.flow(posLeft,self.time)
        uVKR,vVKR,_ = self.flow(posRight,self.time)
        
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        relx = dx*np.cos(ort) + dy*np.sin(ort)
        rely = -dx*np.sin(ort) + dy*np.cos(ort)
        reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
        relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
        reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
        relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
        return np.array([relx, rely, (reluL + reluR)/2, (relvL + relvR)/2, (relvL - relvR)/0.1])
    def _get_obs_ego2lrsingle2grad(self):
        """
        egocentric swimmer position and local flow field and lateral gradient in tangential direction
        """
        ort = angle_normalize(self.pos[-1])
        posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
        posRight = np.array(self.pos[0:2])*2 - posLeft
        uVKL,vVKL,_ = self.flow(posLeft,self.time)
        uVKR,vVKR,_ = self.flow(posRight,self.time)
        
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        relx = dx*np.cos(ort) + dy*np.sin(ort)
        rely = -dx*np.sin(ort) + dy*np.cos(ort)
        reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
        relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
        reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
        relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
        return np.array([relx, rely, (reluL + reluR)/2, (relvL + relvR)/2, (reluL - reluR)/0.1])
    # def _get_obs_ego2fb(self):
    #     """
    #     egocentric swimmer position and local flow fields from 2 sensors located on the front and back of the swimmer
    #     """
    #     ort = angle_normalize(self.pos[-1])
    #     posFront = self.pos[0:2] + np.array([self.bw/2*np.cos(ort), self.bw/2*np.sin(ort)])
    #     posBack = np.array(self.pos[0:2])*2 - posFront
    #     uVKF,vVKF,_ = self.flow(posFront,self.time)
    #     uVKB,vVKB,_ = self.flow(posBack,self.time)

    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]

    #     relx = dx*np.cos(ort) + dy*np.sin(ort)
    #     rely = -dx*np.sin(ort) + dy*np.cos(ort)
    #     reluF = uVKF*np.cos(ort) + vVKF*np.sin(ort)
    #     relvF = -uVKF*np.sin(ort) + vVKF*np.cos(ort)
    #     reluB = uVKB*np.cos(ort) + vVKB*np.sin(ort)
    #     relvB = -uVKB*np.sin(ort) + vVKB*np.cos(ort)
    #     return np.array([relx,rely,reluF,relvF,reluB,relvB])
    # def _get_obs_ego2fbgrad(self):
    #     """
    #     egocentric swimmer position and local flow fields from 2 sensors located on the front and back of the swimmer
    #     """
    #     ort = angle_normalize(self.pos[-1])
    #     posFront = self.pos[0:2] + np.array([0.05*np.cos(ort), 0.05*np.sin(ort)])
    #     posBack = np.array(self.pos[0:2])*2 - posFront
    #     uVKF,vVKF,_ = self.flow(posFront,self.time)
    #     uVKB,vVKB,_ = self.flow(posBack,self.time)
        
    #     dx = self.target[0] - self.pos[0]
    #     dy = self.target[1] - self.pos[1]

    #     relx = dx*np.cos(ort) + dy*np.sin(ort)
    #     rely = -dx*np.sin(ort) + dy*np.cos(ort)
    #     reluF = uVKF*np.cos(ort) + vVKF*np.sin(ort)
    #     relvF = -uVKF*np.sin(ort) + vVKF*np.cos(ort)
    #     reluB = uVKB*np.cos(ort) + vVKB*np.sin(ort)
    #     relvB = -uVKB*np.sin(ort) + vVKB*np.cos(ort)
    #     return np.array([relx, rely, (reluF + reluB)/2, (relvF + relvB)/2, (reluF - reluB)/0.1, (relvF - relvB)/0.1])
    def _get_obs_ego4gradonly(self):
        """
        egocentric swimmer position and local flow field and lateral gradient
        """
        ort = angle_normalize(self.pos[-1])
        posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
        posRight = np.array(self.pos[0:2])*2 - posLeft
        posFront = self.pos[0:2] + np.array([0.05*np.cos(ort), 0.05*np.sin(ort)])
        posBack = np.array(self.pos[0:2])*2 - posFront

        uVKL,vVKL,_ = self.flow(posLeft,self.time)
        uVKR,vVKR,_ = self.flow(posRight,self.time)
        uVKF,vVKF,_ = self.flow(posFront,self.time)
        uVKB,vVKB,_ = self.flow(posBack,self.time)
        
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        relx = dx*np.cos(ort) + dy*np.sin(ort)
        rely = -dx*np.sin(ort) + dy*np.cos(ort)

        reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
        relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
        reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
        relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)

        reluF = uVKF*np.cos(ort) + vVKF*np.sin(ort)
        relvF = -uVKF*np.sin(ort) + vVKF*np.cos(ort)
        reluB = uVKB*np.cos(ort) + vVKB*np.sin(ort)
        relvB = -uVKB*np.sin(ort) + vVKB*np.cos(ort)

        return np.array([relx, rely, (reluL - reluR)/0.1, (relvL - relvR)/0.1, (reluF - reluB)/0.1, (relvF - relvB)/0.1])
    def _get_obs_egocustomize(self):
        """
        egocentric swimmer position and local flow field and lateral gradient
        """
        ort = angle_normalize(self.pos[-1])
        posLeft = self.pos[0:2] + np.array([-0.05*np.sin(ort), 0.05*np.cos(ort)])
        posRight = np.array(self.pos[0:2])*2 - posLeft
        uVKL,vVKL,oL = self.flow(posLeft,self.time)
        uVKR,vVKR,oR = self.flow(posRight,self.time)
        
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        relx = dx*np.cos(ort) + dy*np.sin(ort)
        rely = -dx*np.sin(ort) + dy*np.cos(ort)
        reluL = uVKL*np.cos(ort) + vVKL*np.sin(ort)
        relvL = -uVKL*np.sin(ort) + vVKL*np.cos(ort)
        reluR = uVKR*np.cos(ort) + vVKR*np.sin(ort)
        relvR = -uVKR*np.sin(ort) + vVKR*np.cos(ort)
        if abs(self.pos[1]) > 0.8:
            return np.array([relx, rely, (reluL + reluR)/2, (relvL + relvR)/2, 0, 0])
        elif abs(self.pos[1]) > 0.5:
            return np.array([relx, rely, (reluL + reluR)/2, (relvL + relvR)/2, (reluL - reluR)/0.1, (relvL - relvR)/0.1])
        else:
            return np.array([relx, rely, 0, 0 , (reluL - reluR)/0.1, (relvL - relvR)/0.1])
    def render(self, mode='human'):
#        print(self.pos)
        from gym.envs.classic_control import rendering
        # from pyglet.gl import glRotatef, glPushMatrix

#        class Flip(rendering.Transform):
#            def __init__(self, flipx=False, flipy=False):
#                self.flipx = flipx
#                self.flipy = flipy
#            def enable(self):
#                glPushMatrix()
#                if self.flipx: glRotatef(180, 0, 1., 0)
#                if self.flipy: glRotatef(180, 1., 0, 0)

        def make_ellipse(major=10, minor=5, res=30, filled=True):
            points = []
            for i in range(res):
                ang = 2*np.pi*i / res
                points.append((np.cos(ang)*major, np.sin(ang)*minor))
            if filled:
                return rendering.FilledPolygon(points)
            else:
                return rendering.PolyLine(points, True)
        
        # def draw_lasting_circle(Viewer, radius=10, res=30, filled=True, **attrs):
        #     geom = rendering.make_circle(radius=radius, res=res, filled=filled)
        #     rendering._add_attrs(geom, attrs)
        #     Viewer.add_geom(geom)
        #     return geom
        
        # def draw_lasting_line(Viewer, start, end, **attrs):
        #     geom = rendering.Line(start, end)
        #     rendering._add_attrs(geom, attrs)
        #     Viewer.add_geom(geom)
        #     return geom
        
        def draw_ellipse(Viewer, major=10, minor=5, res=30, filled=True, **attrs):
            geom = make_ellipse(major=major, minor=minor, res=res, filled=filled)
            rendering._add_attrs(geom, attrs)
            Viewer.add_onetime(geom)
            return geom
        class bgimage(rendering.Image):
            def render1(self):
                # l = 102
                # r = 972
                # b = 487
                # t = 53
                # self.img.blit(-self.width/2/(r-l)*(l+r), -self.height/2/(b-t)*(self.img.height*2-b-t), width=self.width/(r-l)*self.img.width, height=self.height/(b-t)*self.img.height)
                self.img.blit(-self.width/4*1, -self.height/2, width=self.width, height=self.height)
        
        x,y,theta = self.pos
        
        if (self.mode == 'reduced'):
            leftbound = -8
            rightbound = 8
            lowerbound = -6
            upperbound = 6
        else:
            leftbound = self.permittedL - 0.5
            rightbound = self.permittedR + 0.5
            lowerbound = self.permittedD - 0.5
            upperbound = self.permittedU + 0.5
        if self.viewer is None:
            scale = 50
            self.viewer = rendering.Viewer((rightbound-leftbound)*scale,(upperbound-lowerbound)*scale)
            # background = draw_lasting_circle(self.viewer,radius=100, res=10)
            # background.set_color(1.0,.8,0)

            # leftbound = -bound+self.target[0]/2
            # rightbound = bound+self.target[0]/2
            # lowerbound = -bound+self.target[1]/2
            # upperbound = bound+self.target[1]/2
            self.viewer.set_bounds(leftbound,rightbound,lowerbound,upperbound)
        if (self.mode == 'reduced'):
            """"vortex street"""
            # vortexN = np.ceil((rightbound - leftbound)/self.lam)
            U = self.Gamma/2/self.lam*np.tanh(2*np.pi*self.A/self.lam)+self.bgflow
            phase = (U*self.time)%self.lam
            vorDownX = np.arange((phase-leftbound)%self.lam+leftbound,rightbound,self.lam)
            vortexN = len(vorDownX)
            for i in range(vortexN):
                vortexUp = self.viewer.draw_circle(radius = 0.1)
                vortexDown = self.viewer.draw_circle(radius = 0.1)
                vorUpTrans = rendering.Transform(translation=(vorDownX[i]+self.lam/2,self.A))
                vortexUp.add_attr(vorUpTrans)
                vortexUp.set_color(1,0,0)
                vorDownTrans = rendering.Transform(translation=(vorDownX[i],-self.A))
                vortexDown.add_attr(vorDownTrans)
                vortexDown.set_color(0,0,1)
        else:
            """Load CFD images"""
            cfdimage = bgimage(cf.read_image(self.time % self.time_span, rootpath = self.cfdpath, dump_interval = 10, frame_rate = self.cfd_framerate),32,16)
            # cfdimage.flip = True     # to flip the image horizontally
            self.viewer.add_onetime(cfdimage)

        """draw the axes"""
#        self.viewer.draw_line((-1000., 0), (1000., 0))
#        self.viewer.draw_line((0,-1000.), (0,1000.))
        
        """target"""
        l = 0.06
        d1 = l*(np.tan(0.3*np.pi)+np.tan(0.4*np.pi))
        d2 = l/np.cos(0.3*np.pi)
        target = self.viewer.draw_polygon(v = [(d2*np.cos(np.pi*0.7),d2*np.sin(np.pi*0.7)),(d1*np.cos(np.pi*0.9),d1*np.sin(np.pi*0.9)),
                                               (d2*np.cos(np.pi*1.1),d2*np.sin(np.pi*1.1)),(d1*np.cos(np.pi*1.3),d1*np.sin(np.pi*1.3)),
                                               (d2*np.cos(np.pi*1.5),d2*np.sin(np.pi*1.5)),(d1*np.cos(np.pi*1.7),d1*np.sin(np.pi*1.7)),
                                               (d2*np.cos(np.pi*1.9),d2*np.sin(np.pi*1.9)),(d1*np.cos(np.pi*0.1),d1*np.sin(np.pi*0.1)),
                                               (d2*np.cos(np.pi*0.3),d2*np.sin(np.pi*0.3)),(d1*np.cos(np.pi*0.5),d1*np.sin(np.pi*0.5))])
        tgTrans = rendering.Transform(translation=(self.target[0], self.target[1]))
        target.add_attr(tgTrans)
        target.set_color(.0,.5,.2)
        """trail"""
        self.trail.append([x,y])
        for i in range(len(self.trail)-1): 
            trail = self.viewer.draw_line(start=(self.trail[i][0],self.trail[i][1]), end=(self.trail[i+1][0],self.trail[i+1][1]))
            trail.linewidth.stroke = 2
            # trail = draw_lasting_circle(self.viewer, radius=0.05, res = 5)
            # trTrans = rendering.Transform(translation=(x,y))
            # trail.add_attr(trTrans)
#            trail.set_color(.6, .106, .118)
#            trail.linewidth.stroke = 10
            trail.set_color(0.2,0.2,0.2)        
        """distance line"""
        # Xnose = x + np.cos(theta)*self.bl/2
        # Ynose = y + np.sin(theta)*self.bl/2
        # self.viewer.draw_line((Xnose, Ynose), (self.target[0], self.target[1]))
        
        """swimmer shape"""
        for i in range(1):
            fish = draw_ellipse(self.viewer,major=self.bl/2, minor=self.bw/2, res=30, filled=False)
            fsTrans = rendering.Transform(rotation=theta,translation=(x,y))
            fish.add_attr(fsTrans)
            fish.set_linewidth(3)
            fish.set_color(.7, .3, .3)
        for i in range(2):
            eye = draw_ellipse(self.viewer,major=self.bl/10, minor=self.bl/10, res=30, filled=True)
            eyngle = theta+np.pi/5.25*(i-.5)*2;
            eyeTrans = rendering.Transform(translation=(x+np.cos(eyngle)*self.bl/4,y+np.sin(eyngle)*self.bl/4))
            eye.add_attr(eyeTrans)
            eye.set_color(.6,.3,.4)
        
        # from pyglet.window import mouse
        # @self.viewer.window.event
#        def on_mouse_press(x, y, buttons, modifiers):
        # def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        #     leftbound = -self.viewer.transform.translation[0]/self.viewer.transform.scale[0]
        #     rightbound = (self.viewer.width-self.viewer.transform.translation[0])/self.viewer.transform.scale[0]
        #     lowerbound = -self.viewer.transform.translation[1]/self.viewer.transform.scale[1]
        #     upperbound = (self.viewer.height-self.viewer.transform.translation[1])/self.viewer.transform.scale[1]
        #     self.set_target((x)/self.viewer.width*(rightbound-leftbound) + leftbound,(y)/self.viewer.height*(upperbound-lowerbound) + lowerbound)
        # return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    def set_target(self,x,y):
        self.target[0] = x
        self.target[1] = y
        # target_angle = np.arctan2(self.target[1] - self.pos[1],self.target[0] - self.pos[0])
        # self.memory_targetAngle = target_angle
        # self.pos[-1] = angle_normalize(self.pos[-1], center = target_angle)
def angle_normalize(x,center = 0,half_period = np.pi):
    return (((x+half_period-center) % (2*half_period)) - half_period+center)
