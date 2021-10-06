import numpy as np
import pandas as pd
from scipy.integrate import odeint
import time
from tqdm import tqdm
from bokeh.io import curdoc, show, output_file
from bokeh.plotting import figure
from bokeh.themes import built_in_themes
from bokeh.layouts import gridplot
output_file(filename="results.html", title="RESULTS DYNAMICS MICROGRID")
#curdoc().theme = 'dark_minimal'

print('Reading data from excel file')
lines = pd.read_excel('cigre.xlsx','lines')
loads = pd.read_excel('cigre.xlsx','loads')
convs = pd.read_excel('cigre.xlsx','converters')
num_nodes = np.max([np.max(lines['From']),np.max(lines['To'])])+1
num_lines = len(lines)
num_loads = len(loads)
num_convs = len(convs)
nodes = list(range(num_nodes))

print('Building the Ybus')
YB = np.zeros((num_nodes,num_nodes))*(0j)
for k in nodes:
  n1 = lines['From'][k]
  n2 = lines['To'][k]
  ykm = 1/(lines['Rpu'][k] + 1j*lines['Xpu'][k])
  YB[n1,n1] = YB[n1,n1] + ykm
  YB[n1,n2] = YB[n1,n2] - ykm
  YB[n2,n1] = YB[n2,n1] - ykm
  YB[n2,n2] = YB[n2,n2] + ykm
# converters
S = np.zeros(num_nodes)*(0j)
for k in range(num_convs):
  n1 = convs['Node'][k]
  S[n1] = S[n1] + convs['Pgenmax'][k]  
# loads
for k in range(num_loads):
  n1 = loads['Node'][k]
  S[n1] = S[n1] - loads['Pload'][k]
  S[n1] = S[n1] - 1j*loads['Qload'][k] 

print('Load flow grid connected')
v0 = 1+0j
sN = S[1:num_nodes]
YN0 = YB[1:num_nodes,0]
YNN = YB[1:num_nodes,1:num_nodes]
vN = np.ones(num_nodes-1)*v0
for t in range(10):
  vN = np.linalg.solve(YNN,np.conj(sN/vN)-v0*YN0)
  vT = np.hstack([v0,vN])
  sT = vT*np.conj(YB@vT)
  err = np.linalg.norm(sT[1:num_nodes]-sN)
print('After 10 iterations the error is: ',err)

  
# Split submatrices
nodes_conv = [convs['Node'][k] for k in range(num_convs)]  
nodes_step = list(set(nodes)-set(nodes_conv))
YSS = YB[np.ix_(nodes_step,nodes_step)]
YSC = YB[np.ix_(nodes_step,nodes_conv)]
YCS = YB[np.ix_(nodes_conv,nodes_step)]
YCC = YB[np.ix_(nodes_conv,nodes_conv)]

# notar que algunas cargas pueden estar en el mismo nodo que un convertidor
SL = np.zeros(num_nodes)*1j
for k in range(num_loads):
  n1 = loads['Node'][k]
  SL[n1] = SL[n1] - loads['Pload'][k]
  SL[n1] = SL[n1] - 1j*loads['Qload'][k]

w_base = 2*np.pi*60
p_ref = {}
q_ref = {}
v_ref = {}
tauPV = {}
tauQF = {}
droopPV = {}
droopQF = {}
theta_conv = {}
for k in range(num_convs):
  n1 = convs['Node'][k]  
  p_ref[n1] = convs['Pgenmax'][k]
  q_ref[n1] = 0
  v_ref[n1] = np.abs(vT[n1])
  droopPV[n1] = convs['droopPV'][k]
  droopQF[n1] = convs['droopQF'][k]
  tauPV[n1] = convs['tauPV'][k]
  tauQF[n1] = convs['tauQF'][k]
  theta_conv[n1] = np.angle(vT[n1])  
p_cal = {}
q_cal = {}
p_conv = p_ref.copy()
q_conv = q_ref.copy()
w_conv = {}
v_conv = {}

V=vT
IL = np.conj(SL/V)
dt = 3E-4
num_points = 7000
output_v = np.zeros((num_points,num_nodes))*1j
output_s = np.zeros((num_points,num_nodes))*1j
output_w = np.ones((num_points,num_nodes))
for iteracion in tqdm(range(num_points)):
    theta_coi = np.mean(np.angle(V))
    v_mag = np.abs(V)
    v_ang = np.angle(V)-theta_coi
    V = v_mag*np.exp(1j*v_ang) # cambiar referencia al centro de inercia
    VC = V[np.ix_(nodes_conv)]
    IL = np.conj(SL/V)
    ILS = IL[np.ix_(nodes_step)]
    ILC = IL[np.ix_(nodes_conv)]
    VS = np.linalg.solve(YSS,ILS-YSC@VC)    
    IC = YCS@VS + YCC@VC - ILC
    SC = VC*np.conj(IC)    
    for k in range(len(nodes_step)):
        n1 = nodes_step[k]
        V[n1] = VS[k]
        output_v[iteracion,n1] = V[n1]
        output_s[iteracion,n1] = SL[n1]
    for k in range(num_convs):
        n1 = nodes_conv[k]
        p_cal[n1] = np.real(SC[k])
        q_cal[n1] = np.imag(SC[k])
        p_conv[n1] = p_conv[n1] + (p_cal[n1]-p_conv[n1])/tauPV[n1]*dt
        q_conv[n1] = q_conv[n1] + (q_cal[n1]-q_conv[n1])/tauQF[n1]*dt
        w_conv[n1] = 1 + droopQF[n1]*(q_conv[n1]-q_ref[n1])        
        v_conv[n1] = v_ref[n1] - droopPV[n1]*(p_conv[n1]-p_ref[n1])        
        theta_conv[n1] = theta_conv[n1] + (w_conv[n1]-1)*w_base*dt
        V[n1] = v_conv[n1]*np.exp(1j*theta_conv[n1])
        output_v[iteracion,n1] = V[n1]
        output_w[iteracion,n1] = w_conv[n1]
        output_s[iteracion,n1] = p_conv[n1] + 1j*q_conv[n1]



output_vc = np.array(output_v)
t = np.linspace(0,dt*num_points,num_points)

# voltages
fig_v = figure(plot_width=500,plot_height=250,title = "Voltages",title_location="left")
plt_v = num_nodes*[0]
for k in range(num_nodes):
    plt_v[k] = fig_v.line(t,np.abs(output_v[:,k]))

# frecuencias solo en los convertidores
fig_w = figure(plot_width=500,plot_height=250,title = "Frequencies",title_location="left")
plt_w = []
for k in nodes_conv:  
  plt_w += [fig_w.line(t,output_w[:,k])]

# Active power solo en los convertidores
fig_p = figure(plot_width=500,plot_height=250,title = "Active power",title_location="left")
plt_p = []
for k in nodes_conv:
    plt_p += [fig_p.line(t,np.real(output_s[:,k]))]

# Reactive power solo en los convertidores
fig_q = figure(plot_width=500,plot_height=250,title = "Reactive power",title_location="left")
plt_q = []
for k in nodes_conv:
    plt_q += [fig_q.line(t,np.imag(output_s[:,k]))] 




show(gridplot([[fig_v, fig_w], [fig_p, fig_q]]))
