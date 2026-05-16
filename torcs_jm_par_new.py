import socket
import sys
import getopt
import os
import time
import numpy as np
PI= 3.14159265359

data_size = 2**17

ophelp=  'Options:\n'
ophelp+= ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp+= ' --port, -p <port>    TORCS port. [3001]\n'
ophelp+= ' --id, -i <id>        ID for server. [SCR]\n'
ophelp+= ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp+= ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp+= ' --track, -t <track>  Your name for this track. Used for learning. [unknown]\n'
ophelp+= ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp+= ' --mode <mode>        Driving mode: modular, qlearn, custom. [modular]\n'
ophelp+= ' --model-file <path>  File to load/save NN weights.\n'
ophelp+= ' --train              Train while running.\n'
ophelp+= ' --train-steer        Train only steering network.\n'
ophelp+= ' --train-throttle     Train only throttle/brake network.\n'
ophelp+= ' --train-gear         Train only gear network.\n'
ophelp+= ' --fixed-steer        Use fixed steering controller instead of learned steering.\n'
ophelp+= ' --fixed-throttle     Use fixed throttle/brake controller instead of learned throttle.\n'
ophelp+= ' --fixed-gear         Use fixed gear shifting instead of learned gear.\n'
ophelp+= ' --load-model         Load saved model weights at start.\n'
ophelp+= ' --save-model         Save weights at exit.\n'
ophelp+= ' --steer-model-file <path>    File for steering network weights.\n'
ophelp+= ' --throttle-model-file <path> File for throttle/brake network weights.\n'
ophelp+= ' --gear-model-file <path>     File for gear network weights.\n'
ophelp+= ' --epsilon <float>    Maximum epsilon for exploration in Q-learning. Actual per-episode epsilon is sampled between 0.0 and this value. [0.1]\n'
ophelp+= ' --gamma <float>      Discount factor for Q-learning. [0.99]\n'
ophelp+= ' --train-fast         Enable fast training mode (reduced output, shorter episodes, smaller networks).\n'
ophelp+= ' --novision           Disable vision from TORCS.\n'
ophelp+= ' --debug, -d          Output full telemetry.\n'
ophelp+= ' --help, -h           Show this help.\n'
ophelp+= ' --version, -v        Show current version.'
usage= 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage= usage + ophelp
version= "20130505-2"

def clip(v,lo,hi):
    if v<lo: return lo
    elif v>hi: return hi
    else: return v

def bargraph(x,mn,mx,w,c='X'):
    '''Draws a simple asciiart bar graph. Very handy for
    visualizing what's going on with the data.
    x= Value from sensor, mn= minimum plottable value,
    mx= maximum plottable value, w= width of plot in chars,
    c= the character to plot with.'''
    if not w: return '' # No width!
    if x<mn: x= mn      # Clip to bounds.
    if x>mx: x= mx      # Clip to bounds.
    tx= mx-mn # Total real units possible to show on graph.
    if tx<=0: return 'backwards' # Stupid bounds.
    upw= tx/float(w) # X Units per output char width.
    if upw<=0: return 'what?' # Don't let this happen.
    negpu, pospu, negnonpu, posnonpu= 0,0,0,0
    if mn < 0: # Then there is a negative part to graph.
        if x < 0: # And the plot is on the negative side.
            negpu= -x + min(0,mx)
            negnonpu= -mn + x
        else: # Plot is on pos. Neg side is empty.
            negnonpu= -mn + min(0,mx) # But still show some empty neg.
    if mx > 0: # There is a positive part to the graph
        if x > 0: # And the plot is on the positive side.
            pospu= x - max(0,mn)
            posnonpu= mx - x
        else: # Plot is on neg. Pos side is empty.
            posnonpu= mx - max(0,mn) # But still show some empty pos.
    nnc= int(negnonpu/upw)*'-'
    npc= int(negpu/upw)*c
    ppc= int(pospu/upw)*c
    pnc= int(posnonpu/upw)*'_'
    return '[%s]' % (nnc+npc+ppc+pnc)

class Client():
    def __init__(self,H=None,p=None,i=None,e=None,t=None,s=None,d=None,vision=False):
        self.vision = vision

        self.host= 'localhost'
        self.port= 3001
        self.sid= 'SCR'
        self.maxEpisodes=1 # "Maximum number of learning episodes to perform"
        self.trackname= 'unknown'
        self.stage= 3 # 0=Warm-up, 1=Qualifying 2=Race, 3=unknown <Default=3>
        self.debug= False
        self.maxSteps= 100000  # 50steps/second
        self.mode = 'modular'
        self.model_file = None
        self.train = False
        self.train_steer = False
        self.train_throttle = False
        self.train_gear = False
        self.fixed_steer = False
        self.fixed_throttle = False
        self.fixed_gear = False
        self.load_model = False
        self.save_model = False
        self.steer_model_file = None
        self.throttle_model_file = None
        self.gear_model_file = None
        self.epsilon = 0.1
        self.gamma = 0.99
        self.train_fast = False
        self.parse_the_command_line()
        if self.train and not any([self.train_steer, self.train_throttle, self.train_gear]):
            self.train_steer = True
            self.train_throttle = True
            self.train_gear = True
        if H: self.host= H
        if p: self.port= p
        if i: self.sid= i
        if e: self.maxEpisodes= e
        if t: self.trackname= t
        if s: self.stage= s
        if d: self.debug= d
        self.S= ServerState()
        self.R= DriverAction()
        self.setup_connection()

    def setup_connection(self):
        try:
            self.so= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as emsg:
            print('Error: Could not create socket...')
            sys.exit(-1)
        self.so.settimeout(1)

        n_fail = 1
        while True:
            a= "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"

            initmsg='%s(init %s)' % (self.sid,a)

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as emsg:
                sys.exit(-1)
            sockdata= str()
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print("Waiting for server on %d............" % self.port)
                print("Count Down : " + str(n_fail))
                if n_fail < 0:
                    print("relaunch torcs")
                    os.system('pkill torcs')
                    time.sleep(1.0)
                    if self.vision is False:
                        os.system('torcs -nofuel -nodamage -nolaptime &')
                        time.sleep(1.0)
                        os.system('sh autostart.sh')
                    else:
                        os.system('torcs -nofuel -nodamage -nolaptime -r practice.xml &')
                    time.sleep(0.1)

                    n_fail = 5
                n_fail -= 1

            identify = '***identified***'
            if identify in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                       ['host=','port=','id=','steps=',
                        'episodes=','track=','stage=',
                        'mode=','model-file=','train',
                        'train-steer','train-throttle','train-gear',
                        'fixed-steer','fixed-throttle','fixed-gear',
                        'load-model','save-model',
                        'steer-model-file=','throttle-model-file=','gear-model-file=',
                        'epsilon=','gamma=',
                        'train-fast',
                        'debug','help','version', 'novision'])
        except getopt.error as why:
            print('getopt error: %s\n%s' % (why, usage))
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] == '-h' or opt[0] == '--help':
                    print(usage)
                    sys.exit(0)
                if opt[0] == '-d' or opt[0] == '--debug':
                    self.debug= True
                if opt[0] == '-H' or opt[0] == '--host':
                    self.host= opt[1]
                if opt[0] == '-i' or opt[0] == '--id':
                    self.sid= opt[1]
                if opt[0] == '-t' or opt[0] == '--track':
                    self.trackname= opt[1]
                if opt[0] == '-s' or opt[0] == '--stage':
                    self.stage= int(opt[1])
                if opt[0] == '-p' or opt[0] == '--port':
                    self.port= int(opt[1])
                if opt[0] == '-e' or opt[0] == '--episodes':
                    self.maxEpisodes= int(opt[1])
                if opt[0] == '-m' or opt[0] == '--steps':
                    self.maxSteps= int(opt[1])
                if opt[0] == '--mode':
                    self.mode = opt[1]
                if opt[0] == '--model-file':
                    self.model_file = opt[1]
                if opt[0] == '--train':
                    self.train = True
                if opt[0] == '--train-steer':
                    self.train_steer = True
                if opt[0] == '--train-throttle':
                    self.train_throttle = True
                if opt[0] == '--train-gear':
                    self.train_gear = True
                if opt[0] == '--fixed-steer':
                    self.fixed_steer = True
                if opt[0] == '--fixed-throttle':
                    self.fixed_throttle = True
                if opt[0] == '--fixed-gear':
                    self.fixed_gear = True
                if opt[0] == '--load-model':
                    self.load_model = True
                if opt[0] == '--save-model':
                    self.save_model = True
                if opt[0] == '--steer-model-file':
                    self.steer_model_file = opt[1]
                if opt[0] == '--throttle-model-file':
                    self.throttle_model_file = opt[1]
                if opt[0] == '--gear-model-file':
                    self.gear_model_file = opt[1]
                if opt[0] == '--epsilon':
                    self.epsilon = float(opt[1])
                if opt[0] == '--gamma':
                    self.gamma = float(opt[1])
                if opt[0] == '--train-fast':
                    self.train_fast = True
                if opt[0] == '--novision':
                    self.vision = True
                if opt[0] == '-v' or opt[0] == '--version':
                    print('%s %s' % (sys.argv[0], version))
                    sys.exit(0)
        except ValueError as why:
            print('Bad parameter \'%s\' for option %s: %s\n%s' % (
                                       opt[1], opt[0], why, usage))
            sys.exit(-1)
        if len(args) > 0:
            print('Superflous input? %s\n%s' % (', '.join(args), usage))
            sys.exit(-1)
        self.validate_options()

    def validate_options(self):
        if self.train_fast and not self.train:
            print('Error: --train-fast requires --train.\n%s' % usage)
            sys.exit(-1)
        if any([self.train_steer, self.train_throttle, self.train_gear]) and not self.train:
            print('Error: --train-steer/--train-throttle/--train-gear require --train.\n%s' % usage)
            sys.exit(-1)

    def get_servers_input(self):
        '''Server's input is stored in a ServerState object'''
        if not self.so: return
        sockdata= str()

        while True:
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print('.', end=' ')
            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                continue
            elif '***shutdown***' in sockdata:
                print((("Server has stopped the race on %d. "+
                        "You were in %d place.") %
                        (self.port,self.S.d['racePos'])))
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                print("Server has restarted the race on %d." % self.port)
                self.shutdown()
                return
            elif not sockdata: # Empty?
                continue       # Try again.
            else:
                self.S.parse_server_str(sockdata)
                if self.debug:
                    sys.stderr.write("\x1b[2J\x1b[H") # Clear for steady output.
                    print(self.S)
                break # Can now return from this function.

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1],str(emsg[0])))
            sys.exit(-1)
        if self.debug: print(self.R.fancyout())

    def shutdown(self):
        if not self.so: return
        print(("Race terminated or %d steps elapsed. Shutting down %d."
               % (self.maxSteps,self.port)))
        self.so.close()
        self.so = None

class ServerState():
    '''What the server is reporting right now.'''
    def __init__(self):
        self.servstr= str()
        self.d= dict()

    def parse_server_str(self, server_string):
        '''Parse the server string.'''
        self.servstr= server_string.strip()[:-1]
        sslisted= self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w= i.split(' ')
            self.d[w[0]]= destringify(w[1:])

    def __repr__(self):
        return self.fancyout()
        out= str()
        for k in sorted(self.d):
            strout= str(self.d[k])
            if type(self.d[k]) is list:
                strlist= [str(i) for i in self.d[k]]
                strout= ', '.join(strlist)
            out+= "%s: %s\n" % (k,strout)
        return out

    def fancyout(self):
        '''Specialty output for useful ServerState monitoring.'''
        out= str()
        sensors= [ # Select the ones you want in the order you want them.
        # 'stuckTimer', 
        'fuel',
        'distRaced',
        'distFromStart',
        'opponents',
        'wheelSpinVel',
        'z',
        'speedZ',
        'speedY',
        'speedX',
        # 'targetSpeed',
        'rpm',
        'skid',
        'slip',
        'track',
        'trackPos',
        'angle',
        ]

        for k in sensors:
            if type(self.d.get(k)) is list: # Handle list type data.
                if k == 'track': # Nice display for track sensors.
                    strout= str()
                    raw_tsens= ['%.1f'%x for x in self.d['track']]
                    strout+= ' '.join(raw_tsens[:9])+'_'+raw_tsens[9]+'_'+' '.join(raw_tsens[10:])
                elif k == 'opponents': # Nice display for opponent sensors.
                    strout= str()
                    for osensor in self.d['opponents']:
                        if   osensor >190: oc= '_'
                        elif osensor > 90: oc= '.'
                        elif osensor > 39: oc= chr(int(osensor/2)+97-19)
                        elif osensor > 13: oc= chr(int(osensor)+65-13)
                        elif osensor >  3: oc= chr(int(osensor)+48-3)
                        else: oc= '?'
                        strout+= oc
                    strout= ' -> '+strout[:18] + ' ' + strout[18:]+' <-'
                else:
                    strlist= [str(i) for i in self.d[k]]
                    strout= ', '.join(strlist)
            else: # Not a list type of value.
                if k == 'gear': # This is redundant now since it's part of RPM.
                    gs= '_._._._._._._._._'
                    p= int(self.d['gear']) * 2 + 2  # Position
                    l= '%d'%self.d['gear'] # Label
                    if l=='-1': l= 'R'
                    if l=='0':  l= 'N'
                    strout= gs[:p]+ '(%s)'%l + gs[p+3:]
                elif k == 'damage':
                    strout= '%6.0f %s' % (self.d[k], bargraph(self.d[k],0,10000,50,'~'))
                elif k == 'fuel':
                    strout= '%6.0f %s' % (self.d[k], bargraph(self.d[k],0,100,50,'f'))
                elif k == 'speedX':
                    cx= 'X'
                    if self.d[k]<0: cx= 'R'
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k],-30,300,50,cx))
                elif k == 'speedY': # This gets reversed for display to make sense.
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k]*-1,-25,25,50,'Y'))
                elif k == 'speedZ':
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k],-13,13,50,'Z'))
                elif k == 'z':
                    strout= '%6.3f %s' % (self.d[k], bargraph(self.d[k],.3,.5,50,'z'))
                elif k == 'trackPos': # This gets reversed for display to make sense.
                    cx='<'
                    if self.d[k]<0: cx= '>'
                    strout= '%6.3f %s' % (self.d[k], bargraph(self.d[k]*-1,-1,1,50,cx))
                elif k == 'stucktimer':
                    if self.d[k]:
                        strout= '%3d %s' % (self.d[k], bargraph(self.d[k],0,300,50,"'"))
                    else: strout= 'Not stuck!'
                elif k == 'rpm':
                    g= self.d['gear']
                    if g < 0:
                        g= 'R'
                    else:
                        g= '%1d'% g
                    strout= bargraph(self.d[k],0,10000,50,g)
                elif k == 'angle':
                    asyms= [
                          "  !  ", ".|'  ", "./'  ", "_.-  ", ".--  ", "..-  ",
                          "---  ", ".__  ", "-._  ", "'-.  ", "'\\.  ", "'|.  ",
                          "  |  ", "  .|'", "  ./'", "  .-'", "  _.-", "  __.",
                          "  ---", "  --.", "  -._", "  -..", "  '\\.", "  '|."  ]
                    rad= self.d[k]
                    deg= int(rad*180/PI)
                    symno= int(.5+ (rad+PI) / (PI/12) )
                    symno= symno % (len(asyms)-1)
                    strout= '%5.2f %3d (%s)' % (rad,deg,asyms[symno])
                elif k == 'skid': # A sensible interpretation of wheel spin.
                    frontwheelradpersec= self.d['wheelSpinVel'][0]
                    skid= 0
                    if frontwheelradpersec:
                        skid= .5555555555*self.d['speedX']/frontwheelradpersec - .66124
                    strout= bargraph(skid,-.05,.4,50,'*')
                elif k == 'slip': # A sensible interpretation of wheel spin.
                    frontwheelradpersec= self.d['wheelSpinVel'][0]
                    slip= 0
                    if frontwheelradpersec:
                        slip= ((self.d['wheelSpinVel'][2]+self.d['wheelSpinVel'][3]) -
                              (self.d['wheelSpinVel'][0]+self.d['wheelSpinVel'][1]))
                    strout= bargraph(slip,-5,150,50,'@')
                else:
                    strout= str(self.d[k])
            out+= "%s: %s\n" % (k,strout)

        if STATUS_CURRENT_EPISODE is not None and STATUS_TOTAL_EPISODES is not None:
            if STATUS_STEP is not None and STATUS_TOTAL_STEPS is not None and STATUS_TOTAL_STEPS > 0:
                pct = float(STATUS_STEP) / STATUS_TOTAL_STEPS
                progress_pct = int(100.0 * pct)
                progress_pct = max(0, min(100, progress_pct))
                progress_bar = bargraph(progress_pct, 0, 100, 30, '#')
                out += "Episode %d/%d: %s %3d%%\n" % (
                    STATUS_CURRENT_EPISODE, STATUS_TOTAL_EPISODES,
                    progress_bar, progress_pct)
            else:
                out += "Episode %d/%d\n" % (STATUS_CURRENT_EPISODE, STATUS_TOTAL_EPISODES)

        if STATUS_REWARD is not None:
            out += "Reward: %7.2f\n" % STATUS_REWARD
        return out

class DriverAction():
    '''What the driver is intending to do (i.e. send to the server).
    Composes something like this for the server:
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus 0)(meta 0) or
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus -90 -45 0 45 90)(meta 0)'''
    def __init__(self):
       self.actionstr= str()
       self.d= { 'accel':0.2,
                   'brake':0,
                  'clutch':0,
                    'gear':1,
                   'steer':0,
                   'focus':[-90,-45,0,45,90],
                    'meta':0
                    }

    def clip_to_limits(self):
        """There pretty much is never a reason to send the server
        something like (steer 9483.323). This comes up all the time
        and it's probably just more sensible to always clip it than to
        worry about when to. The "clip" command is still a snakeoil
        utility function, but it should be used only for non standard
        things or non obvious limits (limit the steering to the left,
        for example). For normal limits, simply don't worry about it."""
        self.d['steer']= clip(self.d['steer'], -1, 1)
        self.d['brake']= clip(self.d['brake'], 0, 1)
        self.d['accel']= clip(self.d['accel'], 0, 1)
        self.d['clutch']= clip(self.d['clutch'], 0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear']= 0
        if self.d['meta'] not in [0,1]:
            self.d['meta']= 0
        if type(self.d['focus']) is not list or min(self.d['focus'])<-180 or max(self.d['focus'])>180:
            self.d['focus']= 0

    def __repr__(self):
        self.clip_to_limits()
        out= str()
        for k in self.d:
            out+= '('+k+' '
            v= self.d[k]
            if not type(v) is list:
                out+= '%.3f' % v
            else:
                out+= ' '.join([str(x) for x in v])
            out+= ')'
        return out
        return out+'\n'

    def fancyout(self):
        '''Specialty output for useful monitoring of bot's effectors.'''
        out= str()
        od= self.d.copy()
        od.pop('gear','') # Not interesting.
        od.pop('meta','') # Not interesting.
        od.pop('focus','') # Not interesting. Yet.
        for k in sorted(od):
            if k == 'clutch' or k == 'brake' or k == 'accel':
                strout=''
                strout= '%6.3f %s' % (od[k], bargraph(od[k],0,1,50,k[0].upper()))
            elif k == 'steer': # Reverse the graph to make sense.
                strout= '%6.3f %s' % (od[k], bargraph(od[k]*-1,-1,1,50,'S'))
            else:
                strout= str(od[k])
            out+= "%s: %s\n" % (k,strout)
        return out

def destringify(s):
    '''makes a string into a value or a list of strings into a list of
    values (if possible)'''
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]

def drive_example(c):
    '''This is only an example. It will get around the track but the
    correct thing to do is write your own `drive()` function.'''
    S,R= c.S.d,c.R.d
    target_speed=80
    R['steer']= S['angle']*25 / PI
    R['steer']-= S['trackPos']*.25

    R['accel'] = max(0.0, min(1.0, R['accel']))
    

    if S['speedX'] < target_speed - (R['steer']*2.5):
        R['accel']+= .4
    else:
        R['accel']-= .2
    if S['speedX']<10:
       R['accel']+= 1/(S['speedX']+.1)

    if ((S['wheelSpinVel'][2]+S['wheelSpinVel'][3]) -
       (S['wheelSpinVel'][0]+S['wheelSpinVel'][1]) > 2):
       R['accel']-= 0.1



    R['gear']=1
    if S['speedX']>60:
        R['gear']=2
    if S['speedX']>100:
        R['gear']=3
    if S['speedX']>140:
        R['gear']=4
    if S['speedX']>190:
        R['gear']=5
    if S['speedX']>220:
        R['gear']=6
    return

# if __name__ == "__main__":
#     C= Client(p=3001)
#     for step in range(C.maxSteps,0,-1):
#         C.get_servers_input()
#         drive_example(C)
#         C.respond_to_server()
#     C.shutdown()



#############################################
# MODULAR DRIVE LOGIC WITH USER PARAMETERS  #
#############################################

import math

# ================= USER CONFIGURABLE PARAMETERS =================
TARGET_SPEED = 30  # Target speed in km/h. Increasing this makes the car go faster but may reduce stability.
STEER_GAIN = 30     # Steering sensitivity. Higher values make the car turn more aggressively.
CENTERING_GAIN = 0.6  # How strongly the car corrects its position toward the center of the track.
BRAKE_THRESHOLD = 0.7  # Angle threshold for braking. Lower values brake earlier.
GEAR_SPEEDS = [0, 40, 50, 100, 120, 180]  # Speed thresholds for gear shifting.
ENABLE_TRACTION_CONTROL = True  # Toggle traction control system.

# ================= HELPER FUNCTIONS =================
def calculate_steering(S):
    steer = (S['angle'] * STEER_GAIN / math.pi) - (S['trackPos'] * CENTERING_GAIN)
    return max(-1, min(1, steer))

def calculate_throttle(S, R):
    if S['speedX'] < TARGET_SPEED - (R['steer'] * 2.5):
        accel = min(1.0, R['accel'] + 0.4)
    else:
        accel = max(0.0, R['accel'] - 0.2)
    if S['speedX'] < 10:
        accel += 1 / (S['speedX'] + 0.1)
    return max(0.0, min(1.0, accel))

def apply_brakes(S):
    return 0.8 if abs(S['angle']) > BRAKE_THRESHOLD else 0.0

def shift_gears(S):
    gear = 1
    for i, speed in enumerate(GEAR_SPEEDS):
        if S['speedX'] > speed:
            gear = i + 1
    return min(gear, 6)

def traction_control(S, accel):
    if ENABLE_TRACTION_CONTROL:
        if ((S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) - (S['wheelSpinVel'][0] + S['wheelSpinVel'][1])) > 2:
            accel -= 0.1
    return max(0.0, accel)

# ================= MAIN DRIVE FUNCTION =================
def drive_modular(c):
    S, R = c.S.d, c.R.d
    R['steer'] = calculate_steering(S)
    R['accel'] = calculate_throttle(S, R)
    R['brake'] = apply_brakes(S)
    R['accel'] = traction_control(S, R['accel'])
    R['gear'] = shift_gears(S)
    return

# ================= Q-LEARNING / CUSTOM ML SUPPORT =================
class TaskNetwork:
    def __init__(self, input_dim, output_dim, hidden_dim=64, lr=1e-3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        # Xavier-like initialization for better training stability
        w1_init_scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        w2_init_scale = np.sqrt(2.0 / (hidden_dim + output_dim))
        self.w1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * w1_init_scale
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * w2_init_scale
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, state):
        x = np.array(state, dtype=np.float32)
        h = np.dot(x, self.w1) + self.b1
        h_relu = np.maximum(h, 0)
        q_values = np.dot(h_relu, self.w2) + self.b2
        return q_values, h_relu, x

    def predict(self, state):
        q_values, _, _ = self.forward(state)
        return q_values

    def update(self, state, action_idx, target_q, invalid=False):
        """Update only the taken action's Q-value (proper Q-learning).
        
        Args:
            state: Current state vector
            action_idx: Index of action taken
            target_q: Target Q-value for this action
            invalid: If True, this is a terminal/invalid state
        """
        q_values, h_relu, x = self.forward(state)
        
        # Create error only for the taken action
        error = np.zeros(self.output_dim, dtype=np.float32)
        error[action_idx] = q_values[action_idx] - target_q
        
        # Backprop through output layer
        d_out = 2.0 * error
        grad_w2 = np.outer(h_relu, d_out)
        grad_b2 = d_out
        
        # Backprop through hidden layer (only non-zero gradients for taken action)
        dh = np.dot(self.w2, d_out)
        dh[h_relu <= 0] = 0  # ReLU derivative
        grad_w1 = np.outer(x, dh)
        grad_b1 = dh
        
        # Adaptive learning rate scaling for invalid states (slower learning on crashes)
        lr_scale = 0.5 if invalid else 1.0
        
        self.w2 -= self.lr * lr_scale * grad_w2
        self.b2 -= self.lr * lr_scale * grad_b2
        self.w1 -= self.lr * lr_scale * grad_w1
        self.b1 -= self.lr * lr_scale * grad_b1

    def save(self, filepath):
        if filepath is None:
            return
        np.savez_compressed(filepath, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, filepath):
        if filepath is None:
            return None
        data = np.load(filepath)
        net = cls(data['w1'].shape[0], data['w2'].shape[1], data['w1'].shape[1])
        net.w1 = data['w1']
        net.b1 = data['b1']
        net.w2 = data['w2']
        net.b2 = data['b2']
        return net


def model_filename(base, suffix):
    if base is None:
        return None
    return '%s_%s.npz' % (base, suffix)


def load_network(path):
    if path is None:
        return None
    try:
        return TaskNetwork.load(path)
    except Exception as ex:
        print('Failed to load model %s: %s' % (path, ex))
        return None


def build_steer_actions():
    return [0.0, -1.0, -0.5, 0.5, 1.0]


def build_throttle_actions():
    return [
        {'accel': 0.0, 'brake': 0.0},
        {'accel': 0.3, 'brake': 0.0},
        {'accel': 0.6, 'brake': 0.0},
        {'accel': 1.0, 'brake': 0.0},
        {'accel': 0.0, 'brake': 0.6},
        {'accel': 0.0, 'brake': 1.0},
    ]


def build_gear_actions():
    return [1, 2, 3, 4, 5, 6]


def smooth_value(prev, target, alpha=0.25):
    return prev + alpha * (target - prev)


def normalize_state(state, state_type='general'):
    """Normalize state features to [-1, 1] range for better learning.
    
    Args:
        state: numpy array of state features
        state_type: 'general', 'steer', 'throttle', or 'gear'
    """
    normalized = state.copy().astype(np.float32)
    
    if state_type == 'steer' and len(normalized) == 4:
        # [angle, trackPos, speedX, track[9]]
        normalized[0] = np.clip(normalized[0] / np.pi, -1, 1)      # angle [-π, π] -> [-1, 1]
        normalized[1] = np.clip(normalized[1], -1, 1)              # trackPos [-1, 1]
        normalized[2] = np.clip(normalized[2] / 300.0, -1, 1)      # speedX [0, 300] -> [0, 1]
        normalized[3] = np.clip(normalized[3] / 200.0 - 0.5, -1, 1) # track distance [-1, 1]
    elif state_type == 'throttle' and len(normalized) == 5:
        # [speedX, angle, trackPos, rpm, wheelSpinVel]
        normalized[0] = np.clip(normalized[0] / 300.0, 0, 1)       # speedX [0, 300] -> [0, 1]
        normalized[1] = np.clip(normalized[1] / np.pi, -1, 1)      # angle
        normalized[2] = np.clip(normalized[2], -1, 1)              # trackPos
        normalized[3] = np.clip(normalized[3] / 10000.0 - 0.4, -1, 1) # rpm
        normalized[4] = np.clip(normalized[4] / 150.0 - 0.5, -1, 1) # wheelSpinVel mean
    elif state_type == 'gear' and len(normalized) == 4:
        # [speedX, rpm, gear, trackPos]
        normalized[0] = np.clip(normalized[0] / 300.0, 0, 1)       # speedX
        normalized[1] = np.clip(normalized[1] / 10000.0 - 0.4, -1, 1) # rpm
        normalized[2] = np.clip((normalized[2] - 3.5) / 3.5, -1, 1) # gear [1-6] centered at 3.5
        normalized[3] = np.clip(normalized[3], -1, 1)              # trackPos
    
    return normalized

def extract_state(S):
    return np.array([
        S['speedX'],
        S['angle'],
        S['trackPos'],
        np.mean(S['track']),
        S['track'][9],
        S['track'][-1],
        np.mean(S['wheelSpinVel']),
    ], dtype=np.float32)


def extract_state_steer(S):
    state = np.array([
        S['angle'],
        S['trackPos'],
        S['speedX'],
        S['track'][9],
    ], dtype=np.float32)
    return normalize_state(state, 'steer')


def extract_state_throttle(S):
    state = np.array([
        S['speedX'],
        S['angle'],
        S['trackPos'],
        S['rpm'],
        np.mean(S['wheelSpinVel']),
    ], dtype=np.float32)
    return normalize_state(state, 'throttle')


def extract_state_gear(S):
    state = np.array([
        S['speedX'],
        S['rpm'],
        S['gear'],
        S['trackPos'],
    ], dtype=np.float32)
    return normalize_state(state, 'gear')


def extract_state_steer_fast(S):
    state = np.array([
        S['angle'],
        S['trackPos'],
        S['speedX'],
    ], dtype=np.float32)
    # For fast training, only normalize first 3 features
    state[0] = np.clip(state[0] / np.pi, -1, 1)
    state[1] = np.clip(state[1], -1, 1)
    state[2] = np.clip(state[2] / 300.0, 0, 1)
    return state


def extract_state_throttle_fast(S):
    state = np.array([
        S['speedX'],
        S['angle'],
        S['rpm'],
    ], dtype=np.float32)
    # For fast training, only normalize 3 features
    state[0] = np.clip(state[0] / 300.0, 0, 1)
    state[1] = np.clip(state[1] / np.pi, -1, 1)
    state[2] = np.clip(state[2] / 10000.0 - 0.4, -1, 1)
    return state


def extract_state_gear_fast(S):
    state = np.array([
        S['speedX'],
        S['rpm'],
        S['gear'],
    ], dtype=np.float32)
    # For fast training, only normalize 3 features
    state[0] = np.clip(state[0] / 300.0, 0, 1)
    state[1] = np.clip(state[1] / 10000.0 - 0.4, -1, 1)
    state[2] = np.clip((state[2] - 3.5) / 3.5, -1, 1)
    return state


def extract_state_fast(S):
    state = np.array([
        S['speedX'],
        S['angle'],
        S['trackPos'],
        S['track'][9],
    ], dtype=np.float32)
    # For fast training, normalize 4 features
    state[0] = np.clip(state[0] / 300.0, 0, 1)
    state[1] = np.clip(state[1] / np.pi, -1, 1)
    state[2] = np.clip(state[2], -1, 1)
    state[3] = np.clip(state[3] / 200.0 - 0.5, -1, 1)
    return state


def fixed_steer_to_center(S):
    steer = S['angle'] * 2.0 #- S['trackPos'] * 1.5
    return clip(steer, -1.0, 1.0)


def fixed_throttle_control(S, target_speed=TARGET_SPEED):
    error = target_speed - S['speedX']
    if error > 10:
        accel = 1.0
        brake = 0.0
    elif error > 1:
        accel = 0.3 + 0.05 * error
        brake = 0.0
    elif error >= -2:
        accel = 0.0
        brake = 0.0
    else:
        accel = 0.0
        brake = 0.6
    return {'accel': clip(accel, 0.0, 1.0), 'brake': clip(brake, 0.0, 1.0)}


def apply_steer(R, S, steer_value):
    R['steer'] = clip(smooth_value(R.get('steer', 0.0), steer_value, ACTION_SMOOTHING), -1, 1)


def apply_throttle(R, S, action):
    R['accel'] = clip(smooth_value(R.get('accel', 0.0), action['accel'], ACTION_SMOOTHING), 0, 1)
    R['brake'] = clip(action.get('brake', 0.0), 0, 1)


def apply_gear(R, action):
    R['gear'] = action

# Reward functions for separate networks
# --------------------------------------
def compute_reward_steer(S, prev_S=None):
    reward = -abs(S['angle']) * 2.0 - abs(S['trackPos']) * 1.0
    invalid = False
    if prev_S is not None:
        reward += (S['distFromStart'] - prev_S['distFromStart']) * 0.1
    if is_invalid_state(S):
        reward += INVALID_TIME_PENALTY
        invalid = True
    return reward, invalid


def compute_reward_throttle(S, prev_S=None):
    reward = S['speedX'] * np.cos(S['angle'])
    reward -= S['speedX'] * abs(S['trackPos'])
    invalid = False
    if prev_S is not None:
        delta_dist = S['distFromStart'] - prev_S['distFromStart']
        reward += delta_dist * DISTANCE_REWARD_MULTIPLIER
        if delta_dist < 0.01 and S['speedX'] < 1.0:
            reward += STANDSTILL_PENALTY
        elif delta_dist < SLOW_MOVEMENT_DISTANCE_THRESHOLD or S['speedX'] < SLOW_MOVEMENT_SPEED_THRESHOLD:
            reward += SLOW_MOVEMENT_PENALTY
    if is_invalid_state(S):
        reward += INVALID_TIME_PENALTY
        invalid = True
    return reward, invalid


def compute_reward_gear(S, prev_S=None):
    reward = 0.0
    invalid = False
    target_rpm = 8000.0
    rpm_error = abs(S['rpm'] - target_rpm)
    reward -= rpm_error * 0.002
    if prev_S is not None:
        delta_dist = S['distFromStart'] - prev_S['distFromStart']
        reward += delta_dist * 0.5
    if S['gear'] < 1 or S['gear'] > 6:
        reward -= 5.0
    if is_invalid_state(S):
        reward += INVALID_TIME_PENALTY
        invalid = True
    return reward, invalid

# Stałe nagradzania i sterowania
# -----------------------------
# Kara za stan invalidny, gdy agent wjeżdża poza trasę lub kończy się epizod błędem.
INVALID_TIME_PENALTY = -10000000.0
# Kara za jazdę w złym kierunku (kąt większy niż 90 stopni względem toru).
INVALID_ANGLE_PENALTY = -50000.0
# Kara za zbyt dużą prędkość przy niebezpiecznym kącie lub innych krytycznych sytuacjach.
INVALID_SPEED_PENALTY = -30.0
# Kara za zbyt duże odchylenie od środka toru.
INVALID_TRACK_POS_PENALTY = -20.0
# Kara za bycie zbyt blisko ściany.
WALL_PROXIMITY_PENALTY = -400.0
# Próg odległości od ściany poniżej którego zaczyna się kara za bliskie podejście.
WALL_PROXIMITY_THRESHOLD = 6.0
# Współczynnik płynnego przejścia dla sterowania (niskoprzepustowy filtr akcji).
ACTION_SMOOTHING = 0.25
# Minimalny dystans, po którym uznajemy, że nastąpił postęp i nie jest to stojąca w miejscu sytuacja.
STALL_DISTANCE_MIN = 1.0
# Limit kroków bez ruchu przed uznaniem epizodu za zablokowany (stall).
STALL_TIME_LIMIT_STEPS = 500
# Kara za zatrzymanie lub brak postępu przy bardzo niskiej prędkości.
STANDSTILL_PENALTY = -2000.0
# Kara za wolną jazdę lub niewystarczający postęp w czasie kroku.
SLOW_MOVEMENT_PENALTY = -10.0
# Próg prędkości uznawany za zbyt wolny ruch.
SLOW_MOVEMENT_SPEED_THRESHOLD = 4.0
# Próg zmiany dystansu uznawany za wolny postęp.
SLOW_MOVEMENT_DISTANCE_THRESHOLD = 0.3
# Mnożnik nagrody za dystans pokonany od poprzedniego kroku.
DISTANCE_REWARD_MULTIPLIER = 600.0
# Mnożnik nagrody za dystans pokonany od początku wyścigu.
ABSOLUTE_DISTANCE_REWARD_MULTIPLIER = 10.0
# Domyślny współczynnik eksploracji epsilon przy wyborze losowego działania.
DEFAULT_EXPLORATION = 0.2
# Zmienna statusowa do wyświetlania numeru bieżącego epizodu.
STATUS_CURRENT_EPISODE = None
# Zmienna statusowa do wyświetlania całkowitej liczby epizodów.
STATUS_TOTAL_EPISODES = None
# Zmienna statusowa do wyświetlania numeru bieżącego kroku w epizodzie.
STATUS_STEP = None
# Zmienna statusowa do wyświetlania maksymalnej liczby kroków w epizodzie.
STATUS_TOTAL_STEPS = None
# Zmienna statusowa do wyświetlania aktualnej wartości nagrody.
STATUS_REWARD = None

MINIMUM_EPSILON = 0.01
# Epsilon decay rate: multiply by this each episode for exponential decay
EPSILON_DECAY_RATE = 0.98


def compute_reward(S, prev_S=None):
    reward = S['speedX'] * np.cos(S['angle'])
    invalid = False
    delta_dist = 0.0
    reward += S['distRaced'] * ABSOLUTE_DISTANCE_REWARD_MULTIPLIER
    if prev_S is not None:
        delta_dist = S['distFromStart'] - prev_S['distFromStart']
        reward += delta_dist * DISTANCE_REWARD_MULTIPLIER
        if delta_dist < 0.01 and S['speedX'] < 1.0:
            reward += STANDSTILL_PENALTY
        elif delta_dist < SLOW_MOVEMENT_DISTANCE_THRESHOLD or S['speedX'] < SLOW_MOVEMENT_SPEED_THRESHOLD:
            reward += SLOW_MOVEMENT_PENALTY

    track_pos_penalty = INVALID_TRACK_POS_PENALTY * abs(S['trackPos'])
    reward += track_pos_penalty

    closest_wall = min(S['track'])
    if closest_wall < WALL_PROXIMITY_THRESHOLD:
        proximity_factor = max(0.0, (WALL_PROXIMITY_THRESHOLD - closest_wall) / WALL_PROXIMITY_THRESHOLD)
        reward += WALL_PROXIMITY_PENALTY * proximity_factor

    if prev_S is not None and 'damage' in S and 'damage' in prev_S:
        if S['damage'] > prev_S['damage']:
            reward -= 10000.0
            invalid = True

    if min(S['track']) < 0 or abs(S['trackPos']) > 1.0:
        reward += INVALID_TIME_PENALTY
        invalid = True
    if np.cos(S['angle']) < 0:
        reward += INVALID_ANGLE_PENALTY
        invalid = True

    if abs(S['angle']) > 1.0 and S['speedX'] > 20:
        reward += INVALID_SPEED_PENALTY

    return reward, invalid


def is_invalid_state(S):
    return min(S['track']) < 0 or abs(S['trackPos']) > 1.0 or np.cos(S['angle']) < 0


def check_lap_drop(S, prev_S):
    if prev_S is None:
        return False
    required = ['distFromStart', 'speedX']
    for key in required:
        if key not in S or key not in prev_S:
            return False
    if is_invalid_state(prev_S) or is_invalid_state(S):
        return False
    if prev_S['distFromStart'] <= S['distFromStart']:
        return False
    if prev_S['distFromStart'] < 2500.0:
        return False
    if S['distFromStart'] > 100.0:
        return False
    if S['speedX'] <= 5.0:
        return False
    return True


def restart_race(c):
    print('Invalid event detected, restarting race automatically.')
    c.R.d['meta'] = 1
    try:
        c.respond_to_server()
    except Exception:
        pass
    time.sleep(0.2)
    c.shutdown()
    time.sleep(0.5)
    os.system('pkill torcs')
    time.sleep(0.5)
    if c.vision is False:
        os.system('torcs -nofuel -nodamage -nolaptime &')
        time.sleep(1.0)
        os.system('sh autostart.sh')
    else:
        os.system('torcs -nofuel -nodamage -nolaptime -r practice.xml &')
    time.sleep(1.0)
    c.S = ServerState()
    c.R = DriverAction()
    c.setup_connection()
    c.get_servers_input()


def select_action(qnet, state, actions, epsilon):
    if qnet is None or actions is None:
        return None
    if np.random.rand() < epsilon:
        return np.random.randint(len(actions))
    return int(np.argmax(qnet.predict(state)))


def drive_q_learning(c, steer_net, throttle_net, gear_net,
                     steer_actions, throttle_actions, gear_actions,
                     epsilon, steer_state, throttle_state, gear_state):
    S, R = c.S.d, c.R.d
    steer_idx = None
    throttle_idx = None
    gear_idx = None

    if steer_net is not None and not c.fixed_steer:
        steer_idx = select_action(steer_net, steer_state, steer_actions, epsilon)
        apply_steer(R, S, steer_actions[steer_idx])
    else:
        apply_steer(R, S, fixed_steer_to_center(S))

    if throttle_net is not None and not c.fixed_throttle:
        throttle_idx = select_action(throttle_net, throttle_state, throttle_actions, epsilon)
        apply_throttle(R, S, throttle_actions[throttle_idx])
    else:
        apply_throttle(R, S, fixed_throttle_control(S))

    if gear_net is not None and not c.fixed_gear:
        gear_idx = select_action(gear_net, gear_state, gear_actions, epsilon)
        apply_gear(R, gear_actions[gear_idx])
    else:
        apply_gear(R, shift_gears(S))

    return steer_state, throttle_state, gear_state, steer_idx, throttle_idx, gear_idx


def drive_custom(c, steer_net=None, throttle_net=None, gear_net=None,
                 steer_actions=None, throttle_actions=None, gear_actions=None,
                 steer_state=None, throttle_state=None, gear_state=None):
    S, R = c.S.d, c.R.d

    if steer_state is None:
        steer_state = extract_state_steer(S)
    if throttle_state is None:
        throttle_state = extract_state_throttle(S)
    if gear_state is None:
        gear_state = extract_state_gear(S)

    if steer_net is not None and not c.fixed_steer:
        steer_idx = int(np.argmax(steer_net.predict(steer_state)))
        apply_steer(R, S, steer_actions[steer_idx])
    else:
        apply_steer(R, S, fixed_steer_to_center(S))

    if throttle_net is not None and not c.fixed_throttle:
        throttle_idx = int(np.argmax(throttle_net.predict(throttle_state)))
        apply_throttle(R, S, throttle_actions[throttle_idx])
    else:
        apply_throttle(R, S, fixed_throttle_control(S))

    if gear_net is not None and not c.fixed_gear:
        gear_idx = int(np.argmax(gear_net.predict(gear_state)))
        apply_gear(R, gear_actions[gear_idx])
    else:
        apply_gear(R, shift_gears(S))

    return steer_state, throttle_state, gear_state

# ================= MAIN LOOP =================
if __name__ == "__main__":
    C = Client(p=3001)
    if C.train_fast:
        if C.epsilon == 0.1:
            C.epsilon = 0.05   # Lower max epsilon for faster convergence when not overridden
        C.debug = False    # No debug output
        if not any([C.train_steer, C.train_throttle, C.train_gear]):
            C.train_steer = True  # Train only steer by default
        C.fixed_throttle = True  # Use fixed throttle
        C.fixed_gear = True      # Use fixed gear
    steer_actions = build_steer_actions()
    throttle_actions = build_throttle_actions()
    gear_actions = build_gear_actions()
    steer_net = None
    throttle_net = None
    gear_net = None

    C.get_servers_input()

    if not C.train_fast:
        print(C.S.fancyout())

    steer_extract_func = extract_state_steer_fast if C.train_fast else extract_state_steer
    throttle_extract_func = extract_state_throttle_fast if C.train_fast else extract_state_throttle
    gear_extract_func = extract_state_gear_fast if C.train_fast else extract_state_gear
    steer_state_length = len(steer_extract_func(C.S.d))
    throttle_state_length = len(throttle_extract_func(C.S.d))
    gear_state_length = len(gear_extract_func(C.S.d))
    steer_file = C.steer_model_file or model_filename(C.model_file, 'steer')
    throttle_file = C.throttle_model_file or model_filename(C.model_file, 'throttle')
    gear_file = C.gear_model_file or model_filename(C.model_file, 'gear')

    if C.load_model:
        steer_net = load_network(steer_file) if not C.fixed_steer else None
        throttle_net = load_network(throttle_file) if not C.fixed_throttle else None
        gear_net = load_network(gear_file) if not C.fixed_gear else None

    hidden_dim = 64  # Always use full network size for final models
    # Optimize learning rates per task: steering is most critical (low lr), throttle moderate, gear flexible
    steer_lr = 5e-4      # Steering: conservative learning for stability
    throttle_lr = 1e-3   # Throttle: moderate learning rate
    gear_lr = 1.5e-3     # Gear: can learn faster (discrete actions)
    
    if C.train_fast:
        # For fast training, use smaller networks and higher learning rates
        hidden_dim = 32
        steer_lr = 8e-4
        throttle_lr = 1.5e-3
        gear_lr = 2e-3
    
    if steer_net is None and not C.fixed_steer and C.mode == 'qlearn' and C.train_steer:
        steer_net = TaskNetwork(input_dim=steer_state_length, output_dim=len(steer_actions), hidden_dim=hidden_dim, lr=steer_lr)
    if throttle_net is None and not C.fixed_throttle and C.mode == 'qlearn' and C.train_throttle:
        throttle_net = TaskNetwork(input_dim=throttle_state_length, output_dim=len(throttle_actions), hidden_dim=hidden_dim, lr=throttle_lr)
    if gear_net is None and not C.fixed_gear and C.mode == 'qlearn' and C.train_gear:
        gear_net = TaskNetwork(input_dim=gear_state_length, output_dim=len(gear_actions), hidden_dim=hidden_dim, lr=gear_lr)

    if C.mode == 'qlearn' and not C.train and steer_net is None and throttle_net is None and gear_net is None:
        print('No Q-networks available for qlearn mode; falling back to modular control.')
        C.mode = 'modular'


    for episode in range(C.maxEpisodes):
        # Implement exponential epsilon decay for better exploration-exploitation tradeoff
        max_epsilon = max(MINIMUM_EPSILON, C.epsilon * (EPSILON_DECAY_RATE ** episode))
        # Each episode uses the current max_epsilon (no randomization per episode)
        episode_epsilon = max_epsilon
        stall_start_dist = C.S.d.get('distRaced', C.S.d['distFromStart'])
        stall_steps = 0
        STATUS_CURRENT_EPISODE = episode + 1
        STATUS_TOTAL_EPISODES = C.maxEpisodes
        STATUS_REWARD = 0.0
        print('Starting episode %d / %d (epsilon=%.4f)' % (episode + 1, C.maxEpisodes, episode_epsilon))
        prev_S = None
        current_steer_state = steer_extract_func(C.S.d)
        current_throttle_state = throttle_extract_func(C.S.d)
        current_gear_state = gear_extract_func(C.S.d)

        lap_drop_count = 0
        start_dist = C.S.d.get('distFromStart', 0.0)
        require_two_drops = start_dist > 2500.0
        lap_completed = False
        for step in range(C.maxSteps):
            STATUS_STEP = step + 1
            STATUS_TOTAL_STEPS = C.maxSteps
            # print(bargraph(C.S.d['speedX'], 0.0, 1.0, 100))
            if not C.train_fast:
                print(C.S.fancyout())
            

            if C.mode == 'qlearn':
                current_steer_state, current_throttle_state, current_gear_state, steer_idx, throttle_idx, gear_idx = drive_q_learning(
                    C, steer_net, throttle_net, gear_net,
                    steer_actions, throttle_actions, gear_actions,
                    episode_epsilon,
                    current_steer_state, current_throttle_state, current_gear_state)
            elif C.mode == 'custom':
                current_steer_state, current_throttle_state, current_gear_state = drive_custom(
                    C, steer_net=steer_net, throttle_net=throttle_net, gear_net=gear_net,
                    steer_actions=steer_actions, throttle_actions=throttle_actions, gear_actions=gear_actions,
                    steer_state=current_steer_state, throttle_state=current_throttle_state, gear_state=current_gear_state)
                steer_idx = throttle_idx = gear_idx = None
            elif C.mode == 'modular':
                drive_modular(C)
                current_steer_state = steer_extract_func(C.S.d)
                current_throttle_state = throttle_extract_func(C.S.d)
                current_gear_state = gear_extract_func(C.S.d)
                steer_idx = throttle_idx = gear_idx = None
            else:
                raise ValueError('Unknown mode: %s' % C.mode)

            C.respond_to_server()
            C.get_servers_input()

            next_steer_state = steer_extract_func(C.S.d)
            next_throttle_state = throttle_extract_func(C.S.d)
            next_gear_state = gear_extract_func(C.S.d)
            if check_lap_drop(C.S.d, prev_S):
                lap_drop_count += 1
                print('LAP_DROP %d detected at episode %d step %d' % (lap_drop_count, episode + 1, step + 1))
                if (require_two_drops and lap_drop_count >= 2) or (not require_two_drops and lap_drop_count >= 1):
                    lap_completed = True
                    print('LAP_COMPLETED at episode %d step %d' % (episode + 1, step + 1))
            reward, invalid = compute_reward(C.S.d, prev_S)
            steer_reward, steer_invalid = compute_reward_steer(C.S.d, prev_S)
            throttle_reward, throttle_invalid = compute_reward_throttle(C.S.d, prev_S)
            gear_reward, gear_invalid = compute_reward_gear(C.S.d, prev_S)
            STATUS_REWARD = reward
            prev_S = C.S.d.copy()

            if C.mode == 'qlearn' and C.train:
                # Compute target Q-values using Bellman equation: Q(s,a) = r + γ*max(Q(s',a'))
                if steer_net is not None and not C.fixed_steer and steer_idx is not None and C.train_steer:
                    print('REWARD STEER: %.2f' % steer_reward)
                    if steer_invalid:
                        # For terminal states, Q-value is just the reward
                        target_q = steer_reward
                    else:
                        # For non-terminal states, use Bellman equation
                        target_q = steer_reward + C.gamma * np.max(steer_net.predict(next_steer_state))
                    steer_net.update(current_steer_state, steer_idx, target_q, invalid=steer_invalid)
                    
                if throttle_net is not None and not C.fixed_throttle and throttle_idx is not None and C.train_throttle:
                    print('REWARD THROTTLE: %.2f' % throttle_reward)
                    if throttle_invalid:
                        target_q = throttle_reward
                    else:
                        target_q = throttle_reward + C.gamma * np.max(throttle_net.predict(next_throttle_state))
                    throttle_net.update(current_throttle_state, throttle_idx, target_q, invalid=throttle_invalid)
                    
                if gear_net is not None and not C.fixed_gear and gear_idx is not None and C.train_gear:
                    print('REWARD GEAR: %.2f' % gear_reward)
                    if gear_invalid:
                        target_q = gear_reward
                    else:
                        target_q = gear_reward + C.gamma * np.max(gear_net.predict(next_gear_state))
                    gear_net.update(current_gear_state, gear_idx, target_q, invalid=gear_invalid)
                

            current_steer_state = next_steer_state
            current_throttle_state = next_throttle_state
            current_gear_state = next_gear_state

            current_dist = C.S.d.get('distRaced', C.S.d['distFromStart'])
            stall_delta = current_dist - stall_start_dist
            if stall_delta >= STALL_DISTANCE_MIN:
                stall_start_dist = current_dist
                stall_steps = 0
            else:
                stall_steps += 1

            if stall_steps >= STALL_TIME_LIMIT_STEPS:
                invalid = True
                reward += STANDSTILL_PENALTY * 2
                print('Stall detected: no forward movement for 10s, restarting.')

            if invalid or is_invalid_state(C.S.d):
                print('Invalid event at step %d, reward %.2f, restarting.' % (step, reward))
                if C.mode == 'qlearn' and C.train and C.save_model:
                    if C.steer_model_file or C.model_file:
                        steer_file = C.steer_model_file or model_filename(C.model_file, 'steer')
                        if steer_net is not None:
                            steer_net.save(steer_file)
                    if C.throttle_model_file or C.model_file:
                        throttle_file = C.throttle_model_file or model_filename(C.model_file, 'throttle')
                        if throttle_net is not None:
                            throttle_net.save(throttle_file)
                    if C.gear_model_file or C.model_file:
                        gear_file = C.gear_model_file or model_filename(C.model_file, 'gear')
                        if gear_net is not None:
                            gear_net.save(gear_file)
                restart_race(C)
                break

            if min(C.S.d['track']) < 0 or np.cos(C.S.d['angle']) < 0:
                print('Episode finished on step %d' % step)
                restart_race(C)
                break

        if C.mode != 'qlearn' or not C.train:
            # if we are not training, we only run one episode
            break
        if lap_completed and episode + 1 >= C.maxEpisodes:
            print('Episode %d completed a lap after minimum %d episodes.' % (episode + 1, C.maxEpisodes))
            break

    if C.save_model:
        if C.steer_model_file or C.model_file:
            steer_file = C.steer_model_file or model_filename(C.model_file, 'steer')
            if steer_net is not None:
                steer_net.save(steer_file)
                print('Saved steer model to %s' % steer_file)
        if C.throttle_model_file or C.model_file:
            throttle_file = C.throttle_model_file or model_filename(C.model_file, 'throttle')
            if throttle_net is not None:
                throttle_net.save(throttle_file)
                print('Saved throttle model to %s' % throttle_file)
        if C.gear_model_file or C.model_file:
            gear_file = C.gear_model_file or model_filename(C.model_file, 'gear')
            if gear_net is not None:
                gear_net.save(gear_file)
                print('Saved gear model to %s' % gear_file)

    C.shutdown()
    if C.train:
        os.system('pkill torcs')
