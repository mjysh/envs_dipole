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
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
# from os import path
from scipy import integrate
# from colorama import Fore, Back, Style
import CFDfunctions as cf
# import time as timer
import importlib
class DipoleSingleEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self,  paramSource = 'envParam_sourceseeking'):
        param = importlib.import_module('settings.'+paramSource)
        # size and parameters
        self.speed = param.mu
        self.bl = param.bl
        self.bw = param.bw
        self.dt = param.dt
        self.angSpeed = param.angularSpeed
        self.cfdpath = param.cfdpath
        self.sensorL = param.sensorLocation
        flow_dict = {
            'CFD': self.__flowVK_CFD,
            'reduced': self.__flowVK_reduced,
            }
        obs_dict = {
            'EgoVelMagGrad': (self._get_obs_egovelmaggrad, 1)
            }
        self.mode = param.flowMode
        self.flow = flow_dict[param.flowMode]
        self.obs, obs_num = obs_dict[param.obsMode]
        if (self.mode == 'CFD'):
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
        self.oldpos = None
        self.trail = []
        self.target = np.zeros((2,))

        high = np.array([np.finfo(np.float32).max] * obs_num)
        low = np.array([np.finfo(np.float32).min] * obs_num)
        # create the observation space and the action space
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low = -1., high = 1., shape = (1,), dtype = np.float32)

        self.viewer = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # self.time = 0
        terminal = False
        dt = self.dt
        self.oldpos = list(self.pos)
        # compute the fish nose position
        disToTarget_old = np.sqrt((self.pos[0]-self.target[0])**2+(self.pos[1]-self.target[1])**2)

        # integrate the dynamics system
        angspeed_adjusted = action[0]*self.angSpeed
        
        def reach(t, y):
            return np.sqrt((y[0]-self.target[0])**2+(y[1]-self.target[1])**2)-0.2
        reach.terminal = True
        options = {'rtol':1e-4,'atol':1e-8,'max_step': 1e-2}

        sol = integrate.solve_ivp(lambda t,y: self.__firstorderdt(t,y,angspeed_adjusted), [self.time,self.time+dt],
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
        # if self.pos[0] > -2 and abs(self.pos[1]<1):
        if self.pos[0]**2 + self.pos[1]**2 < 4:
            reward += 200
            terminal = True
        if self.pos[0]>self.permittedR or self.pos[0]<self.permittedL or self.pos[1]<self.permittedD or self.pos[1]>self.permittedU:
            terminal = True
            
        return self.obs(), reward, terminal, {}
    def __firstorderdt(self,t,pos,angular_speed):
        u = np.cos(pos[2])*self.speed
        v = np.sin(pos[2])*self.speed
        # angular velocity induced by strength difference
        w = angular_speed
        vel = np.array([u, v, w])
        self.vel = vel
        return vel
    """CFD wake"""
    def __flowVK_CFD(self, pos, t):
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
    def __prescribedControl(self,time, position = None):
        # Used when manually precribing the control (disregarding the RL policy, mainly for test)
        """swimming"""
        return 0
    def reset(self, position = None):   # reset the environment setting
        # print(Fore.RED + 'RESET ENVIRONMENT')
        # print(Style.RESET_ALL)
        center = (self.permittedR + self.permittedL)/2
        width = 5
        height = 3
        if position is not None:
            self.pos = position
        else:
            ####################circular zone#######################
            # self.pos = [center + width*np.random.rand(), height*np.random.rand(), np.random.rand()*2*np.pi]
            self.pos = [4*np.random.rand()-9, 6*np.random.rand()-3, np.random.rand()*2*np.pi - np.pi]
            #################################################
        self.set_target(0, 0)
        """"""
        self.vel = np.zeros_like(self.pos)
        self.time = 0
        self.trail = []
        return self.obs()
    """A series of observations to choose from"""
    def _get_obs_egovelmaggrad(self):
        """
        egocentric lateral gradient
        """
        x,y,ort = self.pos

        sensorPos = np.array([x + self.sensorL*np.cos(ort), y + self.sensorL*np.sin(ort)]) 
        posLeft = sensorPos + np.array([- 0.05*np.sin(ort), 0.05*np.cos(ort)])
        posRight = sensorPos*2 - posLeft

        # print(self.time)
        uVKL,vVKL,_ = self.flow(posLeft,self.time)
        uVKR,vVKR,_ = self.flow(posRight,self.time)
        UL = np.sqrt(uVKL**2 + vVKL**2)
        UR = np.sqrt(uVKR**2 + vVKR**2)
        # print('flow speed',UL,UR)
        return np.array([(UL - UR)/0.1])
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
                self.img.blit(-self.width/4*3, -self.height/2, width=self.width, height=self.height)
        
        
        x,y,theta = self.pos
        if (self.mode == 'CFD'):
            leftbound = -24
            rightbound = 1
            lowerbound = -8
            upperbound = 8
        elif (self.mode == 'reduced'):
            leftbound = -8
            rightbound = 8
            lowerbound = -6
            upperbound = 6
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
        
        
        """fish shape"""
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
        @self.viewer.window.event
#        def on_mouse_press(x, y, buttons, modifiers):
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            leftbound = -self.viewer.transform.translation[0]/self.viewer.transform.scale[0]
            rightbound = (self.viewer.width-self.viewer.transform.translation[0])/self.viewer.transform.scale[0]
            lowerbound = -self.viewer.transform.translation[1]/self.viewer.transform.scale[1]
            upperbound = (self.viewer.height-self.viewer.transform.translation[1])/self.viewer.transform.scale[1]
            self.set_target((x)/self.viewer.width*(rightbound-leftbound) + leftbound,(y)/self.viewer.height*(upperbound-lowerbound) + lowerbound)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    def set_target(self,x,y):
        self.target[0] = x
        self.target[1] = y
        target_angle = np.arctan2(self.target[1] - self.pos[1],self.target[0] - self.pos[0])
        # self.memory_targetAngle = target_angle
        self.pos[-1] = angle_normalize(self.pos[-1], center = target_angle)
def angle_normalize(x,center = 0,half_period = np.pi):
    return (((x+half_period-center) % (2*half_period)) - half_period+center)
