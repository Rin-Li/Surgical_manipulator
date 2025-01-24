import numpy as np
from DLS_inverse import DLS 


def posture_without_restrict(p1, p2, Jw_psudo, W_inverse, q):
    delta_p = p2 - p1
    
    q_new = q + W_inverse @ Jw_psudo @ delta_p
    print("Normal", q_new)
    return q_new

def posture_with_restrict(p1, p2, Jw_psudo, J_psudo, J, Jo, W_inverse, q, Jwo, av, ah, xo):
 
    delta_p = p2 - p1
    null_space = np.eye(4) - J_psudo @ J


    inver = Jo @ null_space
    
    q_new = q + W_inverse @ Jw_psudo @ delta_p + ah *  null_space  @ np.linalg.pinv(inver) @ (av * xo - Jwo @ Jw_psudo @ delta_p)
    print("Restrict", q_new)
    return q_new