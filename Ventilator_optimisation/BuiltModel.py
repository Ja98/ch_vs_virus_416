from collections import namedtuple
import pandas as pd
from docplex.mp.model import Model
from docplex.util.environment import get_environment

distance_matrix=np.load("/Users/portia_murray/Dropbox/PhD/Python_scripts/VersusVirus/ch_vs_virus_416/data/hospital_resources/dist_mat.npy")
Hospital_data=pd.read_csv(r'/Users/portia_murray/Dropbox/PhD/Python_scripts/VersusVirus/ch_vs_virus_416/data/hospital_resources/Hospital_resources_portia.csv',encoding="ISO-8859-1")
hospitals=range(0,len(Hospital_data['Inst']))
#Please note that AI, AR, and OW have been removed because they have no acute cases and have no intensive care facilities. If acute cases arise, they will have to be allocated other Kantons
kantons=['AG','BE','BL','BS','FR','GE','GL','GR','JU','LU','NE','NW','SG','SH','SO','SZ','TG','TI','UR','VD','VS','ZG','ZH']
VentDemand=pd.read_csv(r'/Users/portia_murray/Dropbox/PhD/Python_scripts/VersusVirus/ch_vs_virus_416/data/Patient_data/covid19_vent_switzerland_openzh.csv')
Hospitalized=pd.read_csv(r'/Users/portia_murray/Dropbox/PhD/Python_scripts/VersusVirus/ch_vs_virus_416/data/Patient_data/covid19_hospitalized_switzerland_openzh.csv')
horizon=range(0,39)
Hospitalized=Hospitalized.fillna(0)[0:39]
VentDemand=VentDemand.fillna(0)[0:39]
ventdemand=VentDemand.loc[38,kantons].values

hospitals_per_kanton={}
for kanton in kantons:
    tempHosp=Hospital_data[Hospital_data['KT']==kanton]
    hospitals_per_kanton[kanton]=tempHosp['ICU_beds'].sum()
for i in range(0,len(Hospital_data)):
    Hospital_data.loc[i,'pICUinKanton']=Hospital_data.loc[i,'ICU_beds']/hospitals_per_kanton[Hospital_data.loc[i,'KT']]

HVentDemand=pd.DataFrame(index=range(0,39),columns=hospitals)
for h in hospitals:
    for t in range(0,39):
        HVentDemand.loc[t,h]=round(Hospital_data.loc[h,'pICUinKanton']*VentDemand.loc[t,Hospital_data.loc[h,'KT']])

from docplex.mp.model import Model

# Define model
mdl = Model(name="Ventilators")
# --- decision variables ---
mdl.yv_transport = mdl.binary_var_cube(horizon,hospitals,hospitals, name='yv_transport')
mdl.num_transport = mdl.integer_var_cube(horizon,hospitals,hospitals, name='num_transport')
mdl.dummy_transport=mdl.integer_var_cube(horizon,hospitals,hospitals,name='dummy_transport')
mdl.slack = mdl.integer_var_matrix(horizon,hospitals, name='slack')

# --- constraints ---

# dummy variable
mdl.add_constraints(mdl.dummy_transport[i,j]<=mdl.yv_transport[i,j]*Hospital_data.loc[i,'Ventilators'] for i in hospitals for j in hospitals)
mdl.add_constraints(mdl.dummy_transport[i,j]<=mdl.num_transport[i,j] for i in hospitals for j in hospitals)
mdl.add_constraints(mdl.dummy_transport[i,j]>=mdl.num_transport[i,j]-Hospital_data.loc[i,'Ventilators']*(1-mdl.yv_transport[i,j]) for i in hospitals for j in hospitals)
# cannot transport to/from same hospital
mdl.add_constraints(mdl.yv_transport[i,j]==0 for i in hospitals for j in hospitals if i==j)
mdl.add_constraints(mdl.num_transport[i,j]<=mdl.yv_transport[i,j]*Hospital_data.loc[i,'Ventilators'] for i in hospitals for j in hospitals)
# Number of patients on ventilators + slack (only to be used in case there isn't enough ventilators in the country and is to be minimized in objective)
# must be less than or equal to the ventilations currently in hospital + ventilators arriving hospital - ventilators leaving the hospital
mdl.add_constraints(HVentDemand.loc[38,i]*2+mdl.slack[i]<=Hospital_data.loc[i,'Ventilators']-mdl.sum(mdl.dummy_transport[i,j] for j in hospitals)+mdl.sum(mdl.dummy_transport[j,i] for j in hospitals) for i in hospitals)

# total number of transported ventilators leaving hospital must be less than or equal to the total number of ventilators ---
mdl.add_constraints(mdl.sum(mdl.num_transport[i,j]for j in hospitals)<=Hospital_data.loc[i,'Ventilators'] for i in hospitals)


# --- objective function --- minimise total distance travelled to transport + slack variable in case there are not enough variables
mdl.minimize(mdl.sum(mdl.yv_transport[i,j]*distance_matrix[i,j]/10000 for i in hospitals for j in hospitals)+mdl.sum(mdl.slack[i]**2 for i in hospitals))

#Solve optimisation
assert mdl.solve(), "!!! Solve of the model fails"
mdl.report()