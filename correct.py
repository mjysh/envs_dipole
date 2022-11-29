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
        if (self.mode == 'CFD' or self.mode == 'CFDwRot'):
            leftbound = -24
            rightbound = 0
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