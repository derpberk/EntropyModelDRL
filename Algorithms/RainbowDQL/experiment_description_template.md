# Deep Q Learning Experiment Descriptor

Experiment Name: {self.name}

## Parameters

|        Parameter            |         Value                          
|:---------------------------:|:----------------------:
|
| **Dueling Architecture**    | Yes
| $\gamma$                    | 0.99              
| **Target update**           | Hard
| **Target update int.**      | 100
| $\tau$                      | 0.0001               
| **Prioritized Exp. Replay** | Yes                   
| $\alpha$                    | 0.5                   
| $\beta$                     | 0.4 $\rightarrow$ 1.0 
| $\beta_{interval}$          | 0 $\rightarrow$ 0.9   
| **Noisy Layers**            | No					  
| $\epsilon_{interval}$       | 0 $\rightarrow$ 0.5   
| **N-step**                  | 1
| **Distributional**          | No
| **Num. of supports**        | 51
| $v$ range                   | $[-5,60]$               

