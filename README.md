# Epidemia
Python based toolbox to agent-based or metapopulation epidemic modelling

## Usage
Class `AgentBasedModel` simulates the interactions and mobility patterns of agents individually
```python
import numpy as np
from epidemia import AgentBasedModel

# set T for the growth of normal alpha-syn
T = 20000
ref_model = AgentBasedModel(v=1 , N_regions=42, dt=0.01, dist=dist, weight=weight, growth=growth, clearance=clearance, region_size=region_size, fconn=np.zeros((42, 42)), fcscale=0)

# start the process of normal alpha-syn growth
for t in range(T):
    ref_model.record_to_history()
    ref_model.normal_growth_edge()
    ref_model.normal_growth_region()

print(ref_model.nor)

# initiate the spreading process
ref_model.inject_mis()
T_spread = 20000
for t in range(T_spread):
    ref_model.record_to_history()
    ref_model.misfolded_alpha_syn_spread_edge()
    ref_model.misfolded_alpha_syn_spread_region()

print(ref_model.mis)   
```

Class `PopulationModel` is a much faster implementation, modelling each subgroup as a population.
```python
import numpy as np
from epidemia import PopulationModel

# set T for the growth of normal alpha-syn
T = 10000
ref_model = PopulationModel(v=1 , N_regions=42, dt=0.01, dist=dist, weight=weight, growth_rate=growth_rate, clearance_rate=clearance_rate, region_size=region_size, fconn=np.zeros((42, 42)), fcscale=0)

# start the process of normal alpha-syn growth
for t in range(T):
    ref_model.record_to_history()
    ref_model.normal_growth_edge()
    ref_model.normal_growth_region()

print(ref_model.nor)

# initiate the spreading process 
ref_model.inject_mis()
T_spread = 20000

for t in range(T_spread):
    ref_model.record_to_history()
    ref_model.misfolded_spread_edge()
    ref_model.misfolded_spread_region()

print(ref_model.mis)
```
