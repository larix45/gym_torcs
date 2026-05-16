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
ophelp+= ' --load-model         Load saved model weights at start.\n'
ophelp+= ' --save-model         Save weights at exit.\n'
ophelp+= ' --epsilon <float>    Epsilon for exploration in Q-learning. [0.1]\n'
ophelp+= ' --gamma <float>      Discount factor for Q-learning. [0.99]\n'
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
        self.load_model = False
        self.save_model = False
        self.epsilon = 0.1
        self.gamma = 0.99
        self.parse_the_command_line()
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
                    else:
                        os.system('torcs -nofuel -nodamage -nolaptime -vision &')

                    time.sleep(1.0)
                    os.system('sh autostart.sh')
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
                        'load-model','save-model',
                        'epsilon=','gamma=',
                        'debug','help','version'])
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
                if opt[0] == '--load-model':
                    self.load_model = True
                if opt[0] == '--save-model':
                    self.save_model = True
                if opt[0] == '--epsilon':
                    self.epsilon = float(opt[1])
                if opt[0] == '--gamma':
                    self.gamma = float(opt[1])
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
                          "---  ", ".__  ", "-._  ", "'-.  ", "'\.  ", "'|.  ",
                          "  |  ", "  .|'", "  ./'", "  .-'", "  _.-", "  __.",
                          "  ---", "  --.", "  -._", "  -..", "  '\.", "  '|."  ]
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
TARGET_SPEED = 99  # Target speed in km/h. Increasing this makes the car go faster but may reduce stability.
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
class QNetwork:
    def __init__(self, input_dim, output_dim, hidden_dim=64, lr=1e-3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.w1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.1
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

    def update(self, state, target):
        q_values, h_relu, x = self.forward(state)
        error = q_values - target
        d_out = 2.0 * error
        grad_w2 = np.outer(h_relu, d_out)
        grad_b2 = d_out
        dh = np.dot(self.w2, d_out)
        dh[h_relu <= 0] = 0
        grad_w1 = np.outer(x, dh)
        grad_b1 = dh
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

    def save(self, filepath):
        np.savez_compressed(filepath, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, filepath):
        data = np.load(filepath)
        net = cls(data['w1'].shape[0], data['w2'].shape[1], data['w1'].shape[1])
        net.w1 = data['w1']
        net.b1 = data['b1']
        net.w2 = data['w2']
        net.b2 = data['b2']
        return net


def build_action_set():
    return [
        {'steer': 0.0, 'accel': 0.8},
        {'steer': 0.0, 'accel': 1.0},
        {'steer': -0.5, 'accel': 0.6},
        {'steer': 0.5, 'accel': 0.6},
        {'steer': -1.0, 'accel': 0.2},
        {'steer': 1.0, 'accel': 0.2},
        {'steer': 0.0, 'accel': 0.0, 'brake': 1.0},
    ]


def smooth_value(prev, target, alpha=0.25):
    return prev + alpha * (target - prev)


def extract_state(S):
    return np.array([
        S['speedX'],
        S['angle'],
        S['trackPos'],
        S['track'][0],
        S['track'][9],
        S['track'][-1],
        np.mean(S['wheelSpinVel']),
    ], dtype=np.float32)

INVALID_TIME_PENALTY = -1000.0
INVALID_ANGLE_PENALTY = -5000.0
INVALID_SPEED_PENALTY = -30.0
INVALID_TRACK_POS_PENALTY = -20.0
WALL_PROXIMITY_PENALTY = -200.0
WALL_PROXIMITY_THRESHOLD = 60.0
ACTION_SMOOTHING = 0.25
STALL_DISTANCE_MIN = 1.0
STALL_TIME_LIMIT_STEPS = 500
STANDSTILL_PENALTY = -20.0
SLOW_MOVEMENT_PENALTY = -10.0
SLOW_MOVEMENT_SPEED_THRESHOLD = 4.0
SLOW_MOVEMENT_DISTANCE_THRESHOLD = 0.3
DISTANCE_REWARD_MULTIPLIER = 600.0
DEFAULT_EXPLORATION = 0.2
STATUS_CURRENT_EPISODE = None
STATUS_TOTAL_EPISODES = None
STATUS_STEP = None
STATUS_TOTAL_STEPS = None
STATUS_REWARD = None


def compute_reward(S, prev_S=None):
    reward = S['speedX'] * np.cos(S['angle'])
    invalid = False
    delta_dist = 0.0

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
    else:
        os.system('torcs -nofuel -nodamage -nolaptime -vision &')
    time.sleep(1.0)
    os.system('sh autostart.sh')
    time.sleep(1.0)
    c.S = ServerState()
    c.R = DriverAction()
    c.setup_connection()
    c.get_servers_input()


def select_action(qnet, state, actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(actions))
    return int(np.argmax(qnet.predict(state)))


def apply_action_to_driver(R, S, action):
    R['steer'] = clip(smooth_value(R.get('steer', 0.0), action['steer'], ACTION_SMOOTHING), -1, 1)
    R['accel'] = clip(smooth_value(R.get('accel', 0.0), action['accel'], ACTION_SMOOTHING), 0, 1)
    R['brake'] = clip(smooth_value(R.get('brake', 0.0), action.get('brake', 0.0), ACTION_SMOOTHING), 0, 1)
    R['gear'] = shift_gears(S)


def drive_q_learning(c, qnet, actions, epsilon):
    S, R = c.S.d, c.R.d
    state = extract_state(S)
    action_idx = select_action(qnet, state, actions, epsilon)
    apply_action_to_driver(R, S, actions[action_idx])
    return action_idx


def drive_custom(c, model=None):
    S, R = c.S.d, c.R.d
    if model is not None:
        action_set = build_action_set()
        state = extract_state(S)
        action_idx = int(np.argmax(model.predict(state)))
        apply_action_to_driver(R, S, action_set[action_idx])
    else:
        R['steer'] = S['angle'] * 25 / PI - S['trackPos'] * 0.3
        R['accel'] = 0.6 if S['speedX'] < TARGET_SPEED else 0.3
        R['brake'] = apply_brakes(S)
        R['gear'] = shift_gears(S)

# ================= MAIN LOOP =================
if __name__ == "__main__":
    C = Client(p=3001)
    actions = build_action_set()
    qnet = None

    C.get_servers_input()

    print(C.S.fancyout())

    state_length = len(extract_state(C.S.d))

    if C.load_model and C.model_file:
        try:
            qnet = QNetwork.load(C.model_file)
            print('Loaded model from %s' % C.model_file)
        except Exception as ex:
            print('Failed to load model: %s' % ex)
            qnet = None
    elif C.mode == 'qlearn':
        qnet = QNetwork(input_dim=state_length, output_dim=len(actions), hidden_dim=64, lr=1e-3)


    if qnet is None:
        qnet = QNetwork(input_dim=state_length, output_dim=len(actions), hidden_dim=64, lr=1e-3)


    for episode in range(C.maxEpisodes):
        episode_epsilon = max(0.05, C.epsilon if C.epsilon != 0.1 else DEFAULT_EXPLORATION)
        episode_epsilon = max(0.05, episode_epsilon * (0.95 ** episode))
        stall_start_dist = C.S.d.get('distRaced', C.S.d['distFromStart'])
        stall_steps = 0
        STATUS_CURRENT_EPISODE = episode + 1
        STATUS_TOTAL_EPISODES = C.maxEpisodes
        STATUS_REWARD = 0.0
        print('Starting episode %d / %d (epsilon=%.3f)' % (episode + 1, C.maxEpisodes, episode_epsilon))
        prev_S = None
        current_state = extract_state(C.S.d)

        for step in range(C.maxSteps):
            STATUS_STEP = step + 1
            STATUS_TOTAL_STEPS = C.maxSteps
            # print(bargraph(C.S.d['speedX'], 0.0, 1.0, 100))
            print(C.S.fancyout())
            

            if C.mode == 'qlearn':
                action_idx = drive_q_learning(C, qnet, actions, episode_epsilon)
            elif C.mode == 'custom':
                drive_custom(C, qnet)
            elif C.mode == 'modular':
                drive_modular(C)
            else:
                raise ValueError('Unknown mode: %s' % C.mode)

            C.respond_to_server()
            C.get_servers_input()

            next_state = extract_state(C.S.d)
            reward, invalid = compute_reward(C.S.d, prev_S)
            STATUS_REWARD = reward
            prev_S = C.S.d.copy()

            if C.mode == 'qlearn' and C.train and qnet is not None:
                target = qnet.predict(current_state)
                if invalid:
                    target[action_idx] = reward  # Terminal state: no future value
                    # Additionally penalize all actions in invalid state to learn what not to do
                    target = np.full(len(target), reward)
                else:
                    target[action_idx] = reward + C.gamma * np.max(qnet.predict(next_state))
                qnet.update(current_state, target)

            current_state = next_state

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
                if C.mode == 'qlearn' and C.train and qnet is not None:
                    if C.save_model and C.model_file:
                        qnet.save(C.model_file)
                restart_race(C)
                break

            if min(C.S.d['track']) < 0 or np.cos(C.S.d['angle']) < 0:
                print('Episode finished on step %d' % step)
                restart_race(C)
                break

        if C.mode != 'qlearn' or not C.train:
            # if we are not training, we only run one episode
            break

    if C.save_model and C.model_file and qnet is not None:
        qnet.save(C.model_file)
        print('Saved model to %s' % C.model_file)

    C.shutdown()
