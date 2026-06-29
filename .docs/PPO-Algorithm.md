# Proximal Policy Optimization (PPO-Algorithm) 

* Defines the Policy and Value networks
* Makes training batches
* trains the networks with PPO

## Method train_net()

This function implements the clipped surrogate objective from the PPO paper: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017). 

It combines:
            
* A policy loss (using a clipped and a unclipped surrogate objective)
* A value loss (mean squared error)
* An entropy bonus (to encourage exploration)

Objective
---------
        
    PPO optimizes the following clipped objective:

        L_CLIP(θ) = E_t [ min( r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]

    where:

        r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
        A_t    = advantage estimate for sample t

The clipping prevents excessively large policy updates, improving training stability.

Ratio
-----
Computes the ratio of probabilities of new policy (policy after network update) and old policy (stored in trajectory buffer)

    Ratio = π_new / π_old = exp(log new_prob - log old_prob)

    where:
        new_log_probs = new policy probability of the choosen action from the updated policy network
        old_log_probs = old policy probability of the same action stored when collecting the trajectory

Unclipped obejctive
-------------------
Is the plan vanila Actor Crtic Algorithm

    where:

        unclipped = ratio(t) * advantage(t)

Clipped objective
-----------------
The clipped version uses the ratio to prevent large policy updates that can destabilize learning. The clipping limits how far the new policy can deviate from old policy with a clipping factor ε = 0,1 .. 0,2. If the policy changes too much (ratio > 1 + ε or < 1 − ε) → gradient is cut off

    where:

        clipped = torch.clamp(ratio(t), 1 - eps_clip, 1 + eps_clip) * advantage(t) 
        
            → ensures that the ratio stays between [1 - ε, 1 + ε] 

    
Advantage is used for gradient update. This scales gradient update by how good the action was.

    where:
    
        advantage_tensor = the advantage estimate for each sample. 

Policy loss 
-----------
Takes the smaller (more conservative) objective (clipped or unclipped) for the network update. 
If clipping does apply we take the smaller one, so the update is more conservative. 

The negative sign in policy loss is because we want to maximize the objective, but optimizers minimize loss. So we use a negativ sign to maximize...  
        
    where:

        loss = - min(clipped objective, unclipped objective)

Value loss
--------------------

This term trains the value network with MSE to predict better returns.
    
    where:

        returns_tensor = advantages + values 

        The tensor is detached and prevents gradients from flowing back into the target computation

Combined loss                                                                      
-------------

Taske all losses together with the COEFFICIENTS and backprop through network for an update

    where 
    
        Policy loss → teaches the agent how to act

        Value loss → teaches the agent how to estimate state values

        Entropy loss → supports more exploration of the agent