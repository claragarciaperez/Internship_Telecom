#%%
import picos as pc
import numpy as np
from itertools import product, combinations
import scipy.sparse as sp
import gurobipy
from tqdm import tqdm
import time



#%%
class TW_AB_E():
    """
    Class to solve inequalities given the TW. The TW is imposed in A,B but not
    in E.
    
    Parameters
    ----------
    ma : int, optional
        Number of outcomes for Alice. Default is 2.
    mb : int, optional
        Number of outcomes for Bob. Default is 2.
    me : int, optional
        Number of outcomes for Eve. Default is 2.
    kx : int, optional
        Number of measurements/settings for Alice. Default is 3.
    ky : int, optional
        Number of measurements/settings for Bob. Default is 3.
    kz : int, optional
        Number of measurements/settings for Eve. Default is 3.
    solver : str, optional
        Solver used for linear programming. Default is 'mosek'.
    BR: bool, optional
        If True, applies BR constrain. Default is 'False'.

    
    """
    def __init__(self, ma=2, mb=2, me=2, kx=3, ky=3, kz=3, solver = 'mosek', BR = 'False'):
        self.ma = ma
        self.mb = mb
        if BR=='True':
            self.me = ma*mb
        else: 
            self.me = me
        self.kx = kx
        self.ky = ky
        if BR=='True':
            self.kz = kx*ky
        else: 
            self.kz = kz
        self.BR = BR
        self.outputs = list(product(range(self.ma), range(self.mb), range(self.me)))
        self.inputs = list(product(range(self.kx), range(self.ky), range(self.kz)))
        self.outputs_twin = list(product(range(self.ma),range(self.ma), range(self.mb), range(self.mb), range(self.me)))
        self.inputs_twin = list(product(range(self.kx), range(self.kx),range(self.ky), range(self.ky), range(self.kz)))
        self.solver = solver
    
    def pos(self, a, b, e, x, y, z):
        """
        Index of the probability matrix P for P(abe|xyz).
        
        Parameters
        ----------
        a : int
            Outcome of Alice
        b : int
            Outcome of Bob
        e : int
            Outcome of Eve
        x : int
            Setting of Alice
        y : int
            Setting of Bob
        y : int
            Setting of Eve
        Returns
        -------
        (int,int)
            Position of the probability on the P matrix
        """
        return (a*(self.mb*self.me) + b*self.me + e, x*(self.ky*self.kz) + y*self.kz + z)

    def pos_twin(self, a, at,b,bt,e, x,xt,y,yt, z):
        """
        Index of the probability matrix P for P(abe|xyz).
        
        Parameters
        ----------
        a : int
            Outcome of Alice
        b : int
            Outcome of Bob
        e : int
            Outcome of Eve
        x : int
            Setting of Alice
        y : int
            Setting of Bob
        y : int
            Setting of Eve
        Returns
        -------
        (int,int)
            Position of the probability on the P matrix
        """
        return (a*self.ma*(self.mb**2*self.me) + at*(self.mb**2*self.me) + b*self.mb*self.me + bt*self.me + e, x*self.kx*(self.ky**2*self.kz)+ xt*(self.ky**2*self.kz) + y*self.ky*self.kz + yt*self.kz + z)

    def get_P_twin(self, P_twin, problem):
        """
        Retrieves the probability from P_twin corresponding to a specific
        combination of outcomes and settings using the index map.
    
        Parameters
        ----------
        P_twin : array-like
            1D array or list containing probabilities for all canonical tuples.
        index_map : dict
            Dictionary mapping tuples `(a, at, b, bt, e, x, xt, y, yt, z)`
            to indices in `P_twin`.
        a, at : int
            Outcome and alternate labeling for party A
        b, bt : int
            Outcome and alternate labeling for party B
        e : int
            Outcome labeling for party E
        x, xt : int
            Setting and alternate labeling for party A
        y, yt : int
            Setting and alternate labeling for party B
        z : int
            Setting labeling for party E
    
        Returns
        -------
        float
            Probability corresponding to the given outcomes and settings in `P_twin`.
        """
        for x,xt,y,yt,z in self.inputs_twin:
            for a,at,b,bt in list(product(range(self.ma),range(self.ma), range(self.mb), range(self.mb))):
                P2 = 0
                P1 = 0
                P3 = 0
                P4 = 0
                for e in range(self.me):  
                    P1+= P_twin[self.pos_twin(a, at, b, bt, e, x, xt, y, yt, z)]
                    P2+= P_twin[self.pos_twin(at, a, bt, b, e, xt, x, yt, y, z)]
                    P3+= P_twin[self.pos_twin(at, a, b, bt, e, xt, x, y, yt, z)]
                    P4+= P_twin[self.pos_twin(a, at, bt, b, e, x, xt, yt, y, z)]
                problem.add_constraint(P1 == P2)
                problem.add_constraint(P1 == P3)
                problem.add_constraint(P1 == P4)
    
    

    def add_ns_constraints(self, P_twin, problem):
        """
        Adds No-Signaling constrains to P_twin
    
        Parameters
        ----------
        P_twin : pc.RealVariable
            Vector of reduced probability variables representing canonical twin tuples.
            Each element corresponds to a unique tuple (a, at, b, bt,e, x, xt, y, yt,z).
        problem : pc.Problem
            Linear program object where constraints will be added. Must support 
            `add_constraint` method.
        index_map : dict
            Dictionary mapping each tuple (a, at, b, bt,e x, xt, y, yt,z) to its index in P_twin.
    
        Returns
        -------
        none
        """
    # -------- No-signalling A --------
        for x in range(self.kx):
            for a in range(self.ma):
                ref_terms = [
                    P_twin[self.pos_twin(a, a_twin, b, b_twin, e, x, 0, 0, 0, 0)]
                    for a_twin, b, b_twin, e in product(range(self.ma), range(self.mb), range(self.mb), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for xt, y, yt, z in product(range(self.kx), range(self.ky), range(self.ky), range(self.kz)):
                    terms = []
                    for a_twin, b, b_twin, e in product(range(self.ma), range(self.mb), range(self.mb), range(self.me)):
                        val = P_twin[self.pos_twin(a, a_twin, b, b_twin, e, x, xt, y, yt, z)]
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling A' --------
        for a_twin in range(self.ma):
            for xt in range(self.kx):
                ref_terms = [
                    P_twin[self.pos_twin(a, a_twin, b, b_twin, e, 0, xt, 0, 0, 0)]
                    for a, b, b_twin, e in product(range(self.ma), range(self.mb), range(self.mb), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for x, y, yt, z in product(range(self.kx), range(self.ky), range(self.ky), range(self.kz)):
                    terms = []
                    for a, b, b_twin, e in product(range(self.ma), range(self.mb), range(self.mb), range(self.me)):
                        val = P_twin[self.pos_twin(a, a_twin, b, b_twin, e, x, xt, y, yt, z)]
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling B --------
        for y in range(self.ky):
            for b in range(self.mb):
                ref_terms = [
                    P_twin[self.pos_twin(a, a_twin, b, b_twin, e, 0, 0, y, 0, 0)]
                    for a, a_twin, b_twin, e in product(range(self.ma), range(self.ma), range(self.mb), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for x, xt, yt, z in product(range(self.kx), range(self.kx), range(self.ky), range(self.kz)):
                    terms = []
                    for a, a_twin, b_twin, e in product(range(self.ma), range(self.ma), range(self.mb), range(self.me)):
                        val = P_twin[self.pos_twin(a, a_twin, b, b_twin, e, x, xt, y, yt, z)]
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)
        
        # -------- No-signalling B' --------
        for yt in range(self.ky):
            for b_twin in range(self.mb):
                ref_terms = [
                    P_twin[self.pos_twin(a, a_twin, b, b_twin, e, 0, 0, 0, yt, 0)]
                    for a, a_twin, b, e in product(range(self.ma), range(self.ma), range(self.mb), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for x, xt, y, z in product(range(self.kx), range(self.kx), range(self.ky), range(self.kz)):
                    terms = []
                    for a, a_twin, b, e in product(range(self.ma), range(self.ma), range(self.mb), range(self.me)):
                        val = P_twin[self.pos_twin(a, a_twin, b, b_twin, e, x, xt, y, yt, z)]
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling C (Eve) --------
        for e in range(self.me):
            for z in range(self.kz):
                ref_terms = [
                    P_twin[self.pos_twin(a, a_twin, b, b_twin, e, 0, 0, 0, 0, z)]
                    for a, a_twin, b, b_twin in product(range(self.ma), range(self.ma), range(self.mb), range(self.mb))
                ]
                ref = pc.sum(ref_terms)
                for x, xt, y, yt in product(range(self.kx), range(self.kx), range(self.ky), range(self.ky)):
                    terms = []
                    for a, a_twin, b, b_twin in product(range(self.ma), range(self.ma), range(self.mb), range(self.mb)):
                        val = P_twin[self.pos_twin(a, a_twin, b, b_twin, e, x, xt, y, yt, z)]
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

    def add_ns_constraints_P(self,P, problem):
        # -------- No-signalling A --------
        for x in range(self.kx):
            for a in range(self.ma):
                ref_terms = [
                    P[self.pos(a, b, e, x, 0, 0)]
                    for  b, e in product( range(self.mb), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for y,  z in product(range(self.ky),range(self.kz)):
                    terms = []
                    for b, e in product(range(self.mb), range(self.me) ):
                        val = P[self.pos(a, b, e, x, y, z)]
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)


        # -------- No-signalling B --------
        for y in range(self.ky):
            for b in range(self.mb):
                ref_terms = [
                    P[self.pos(a, b, e, 0, y, 0)]
                    for  a, e in product( range(self.ma), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for x,  z in product(range(self.kx),range(self.kz)):
                    terms = []
                    for a, e in product(range(self.ma), range(self.me) ):
                        val = P[self.pos(a, b, e, x, y, z)]
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)
        

        # -------- No-signalling C --------
        for z in range(self.kz):
            for e in range(self.me):
                ref_terms = [
                    P[self.pos(a, b, e, 0, 0, z)]
                    for  a, b in product( range(self.ma), range(self.mb))
                ]
                ref = pc.sum(ref_terms)
                for x,  y in product(range(self.kx),range(self.ky)):
                    terms = []
                    for a, b in product(range(self.ma), range(self.mb) ):
                        val = P[self.pos(a, b, e, x, y, z)]
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)
    

   
    def normalization_twin(self, P_twin, problem):
        """
        Adds Normalization constrains to P_twin. sum_{a,a',b,b',e,e'}P_twin(aa'bb'e|xx'yy'z) = 1 forall x,x',y,y',z
    
        Parameters
        ----------
        P_twin : pc.RealVariable
            Vector of reduced probability variables representing canonical twin tuples.
            Each element corresponds to a unique tuple (a, at, b, bt,e, x, xt, y, yt,z).
        problem : pc.Problem
            Linear program object where constraints will be added. Must support 
            `add_constraint` method.
        index_map : dict
            Dictionary mapping each tuple (a, at, b, bt,e, x, xt, y, yt,z) to its index in P_twin.
    
        Returns
        -------
        none
        """
        for x, xt, y, yt, z in self.inputs_twin:

            terms = []

            for a, at, b, bt, e in self.outputs_twin:
                val = P_twin[self.pos_twin(a, at, b, bt, e, x, xt, y, yt, z)]
                terms.append(val)

            problem.add_constraint(pc.sum(terms) == 1)
    
    def normalization_P(self, P, problem):
        for x, y, z in self.inputs:

            terms = []

            for a, b, e in self.outputs:
                val = P[self.pos(a, b, e, x, y, z)]
                terms.append(val)

            problem.add_constraint(pc.sum(terms) == 1)


    def relate_P_twin_P(self, P_twin, P, problem):
        """
        Adds constrains relating P with Ptwin such that sum_{a'b'e'}P_twin(aa'bb'ee'|xx'yy'zz') = P(abe|xyz) forall xx'yy'zz'
    
        Parameters
        ----------
        P_twin : pc.RealVariable
            Vector of reduced probability variables representing canonical twin tuples.
            Each element corresponds to a unique tuple (a, at, b, bt,e, x, xt, y, yt,z).
        P : pc.RealVariable
            Matrix of size (ma*mb*me)x(kx*ky*kz) with the probabilities P(abc|xy)
        problem : pc.Problem
            Linear program object where constraints will be added. Must support 
            `add_constraint` method.
        index_map : dict
            Dictionary mapping each tuple (a, at, b, bt, e, x, xt, y, yt, z) to its index in P_twin.
    
        Returns
        -------
        none
        """
        for x, xt, y, yt, z in self.inputs_twin:
            for a, b, e in self.outputs:

                terms = []

                for at, bt in product(range(self.ma), range(self.mb)):
                    val = P_twin[self.pos_twin(a, at, b, bt, e, x, xt, y, yt, z)]
                    terms.append(val)

                problem.add_constraint(
                    P[self.pos(a, b, e, x, y, z)] == pc.sum(terms)
                )
    
    def pos_ab(self, a, b,  x, y):
        """
        Returns the position of the probability P(ab|xy) in the P matrix.    
        Parameters
        ----------
        a : int
            Outcome of Alice
        b : int
            Outcome of Bob
        x : int
            Setting of Alice
        y : int
            Setting of Bob
        Returns
        -------
        (int,int)
            Position of the probability on the P matrix
        """      
        return (a*(self.mb) + b, x*(self.ky) + y)
    
    def constrain_BR1(self,P, P_ab, problem):
        """
        Adds constrains relating P with P_ab such that sum_{e}P(abe|xyz) = P(ab|xy) forall xyz
        Parameters
        ----------
        P : pc.RealVariable
            Matrix of size (ma*mb*me)x(kx*ky*kz) with the probabilities P(abe|xy)
        P_ab : pc.RealVariable
            Matrix of size (ma*mb)x(kx*ky) with the probabilities P(ab|xy)
        problem : pc.Problem
            Linear program object where constraints will be added. Must support 
            `add_constraint` method.
        Returns
        -------
        none
        """
        for x,y,z in self.inputs:
            for a in range(self.ma):
                for b in range (self.mb):
                    terms = []
                    for e in range(self.me):
                        terms.append(P[self.pos(a,b,e,x,y,z)])
                    problem.add_constraint(pc.sum(terms) == P_ab[self.pos_ab(a,b,x,y)])
    def pos_z(self,x,y):
        """
        Returns the position of the setting z in the P matrix.
    
        Parameters
        ----------
        x : int
            Setting of Alice
        y : int
            Setting of Bob
        Returns
        -------
        int
             Position of the setting z in the P matrix.
        """
        return (x*self.ky + y)
    
    def pos_e(self,a,b):
        """
        Returns the position of the outcome e in the P matrix.
    
        Parameters
        ----------
        a : int
            Outcome of Alice
        b : int
            Outcome of Bob
        Returns
        -------
        int
             Position of the outcome e in the P matrix.
        """
        return (a*self.mb + b)

    def constrain_BR2(self,P, problem):
        """
        Adds constrains to P such that P(abe|xyz = (x,y)) = 0 if e != (a,b) forall abe xy
        Parameters
        ----------
        P : pc.RealVariable
            Matrix of size (ma*mb*me)x(kx*ky*kz) with the probabilities P(abe|xy)
        problem : pc.Problem
        Returns
        -------
        none
        """
        for x in range(self.kx):
            for y in range(self.ky):
                z = self.pos_z(x,y)
                for a in range(self.ma):
                    for b in range (self.mb):
                        for e in range(self.me):
                            if(self.pos_e(a,b) != e ):
                                problem.add_constraint(P[self.pos(a,b,e,x,y,z)] == 0)

    def constrain_BR2_vers2(self,P_twin, problem):
        """
        Adds constrains to P such that P(abe|xyz = (x,y)) = 0 if e != (a,b) forall abe xy
        Parameters
        ----------
        P : pc.RealVariable
            Matrix of size (ma*mb*me)x(kx*ky*kz) with the probabilities P(abe|xy)
        problem : pc.Problem
        Returns
        -------
        none
        """
        for x, xt, y, yt in product(range(self.kx), range(self.kx), range(self.ky), range(self.ky)):
            
            # Calculamos directamente los 4 escenarios de inputs que puede recibir Eve
            z1 = self.pos_z(x, y)   # Real / Real
            z2 = self.pos_z(xt, yt) # Gemelo / Gemelo
            z3 = self.pos_z(xt, y)  # Gemelo / Real
            z4 = self.pos_z(x, yt)  # Real / Gemelo
            
            # Iteramos sobre las combinaciones de outputs de los gemelos
            for a, at, b, bt, e in self.outputs_twin:
                
                # Caso 1: Eve recibe z1 -> Debe registrar los outputs (a, b)
                if self.pos_e(a, b) != e:
                    problem.add_constraint(P_twin[self.pos_twin(a, at, b, bt, e, x, xt, y, yt, z1)] == 0)
                    
                # Caso 2: Eve recibe z2 -> Debe registrar los outputs (at, bt)
                if self.pos_e(at, bt) != e:
                    problem.add_constraint(P_twin[self.pos_twin(a, at, b, bt, e, x, xt, y, yt, z2)] == 0)
                    
                # Caso 3: Eve recibe z3 -> Debe registrar los outputs (at, b)
                if self.pos_e(at, b) != e:
                    problem.add_constraint(P_twin[self.pos_twin(a, at, b, bt, e, x, xt, y, yt, z3)] == 0)
                    
                # Caso 4: Eve recibe z4 -> Debe registrar los outputs (a, bt)
                if self.pos_e(a, bt) != e:
                    problem.add_constraint(P_twin[self.pos_twin(a, at, b, bt, e, x, xt, y, yt, z4)] == 0)

    def E(self,x, y, z, P):
        """
        Calculates the expectation value E(x,y,z) = sum_{a,b,e} (-1)^(a+b+e) P(abe|xyz)

        Parameters
        ----------
        x : int
            Setting of Alice
        y : int
            Setting of Bob
        z : int
            Setting of Eve
        P : pc.RealVariable
            Matrix of size (ma*mb*me)x(kx*ky*kz) with the probabilities P(abe|xyz)
        Returns
        -------
        double
            Value of the expectation E(x,y,z)
        """
        return sum(((-1)**(a+b+e)) * P[self.pos(a, b, e, x, y, z)] for a in range(self.ma) for b in range(self.mb) for e in range(self.me))        

    def solve_Mermin(self):
        """
        Calculates the maximum value of the Bell-Mermin inequality. Only valid for k = 2, m=2.
        In this version a in (-1,1), so the local bound is 2.
    
        Parameters
        ----------
        none
    
        Returns
        -------
        prints the maximum value of the Bell Mermin inequality.
        """
        if (self.kx ==2 and self.ky == 2 and self.kz ==2 and self.ma ==2 and self.mb ==2 and self.me ==2):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb*self.me, self.kx*self.ky*self.kz), lower=0, upper=1)
            P_twin = pc.RealVariable("P_TWIN", (self.ma**2*self.mb**2*self.me, self.kx**2*self.ky**2*self.kz), lower=0, upper=1)
            #self.add_ns_constraints( P_twin, problem)ç
            self.get_P_twin(P_twin, problem)
            self.add_ns_constraints_P(P,problem)
            self.normalization_twin(P_twin, problem)
            self.relate_P_twin_P(P_twin, P, problem)
        
            M = (self.E(0,1,1, P) + self.E(1,0,1, P) + self.E(1,1,0, P) - self.E(0,0,0, P))
            problem.set_objective('max', M)
            problem.solve(solver=self.solver) 
            print("Max value of Bell-Mermin:", M.value)
        else: 
            print("Bell-Mermin only defined for ma=mb=me=2 and kx=ky=kz=2")
    
    def solve_Svetlichny(self):
        """
        Calculates the maximum value of the Svetlichny inequality. Only valid for k = 2, m=2.
        In this version a in (-1,1), so the local bound is 4.
    
        Parameters
        ----------
        none
    
        Returns
        -------
        prints the maximum value of the Svetlichny inequality.
        """
        if (self.kx ==2 and self.ky == 2 and self.kz ==2 and self.ma ==2 and self.mb ==2 and self.me ==2):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb*self.me, self.kx*self.ky*self.kz), lower=0, upper=1)
            P_twin = pc.RealVariable("P_TWIN", (self.ma**2*self.mb**2*self.me, self.kx**2*self.ky**2*self.kz), lower=0, upper=1)
            #self.add_ns_constraints( P_twin, problem)ç
            self.get_P_twin(P_twin, problem)
            self.add_ns_constraints_P(P,problem)
            self.normalization_twin(P_twin, problem)
            self.relate_P_twin_P(P_twin, P, problem)
        
    
            S = (self.E(0,0,1,P)
                 + self.E(0,1,0,P)
                 + self.E(1,0,0,P)
                 -self.E(1,1,1,P)
                 +self.E(1,1,0,P)
                 +self.E(1,0,1,P)
                 +self.E(0,1,1,P)
                 -self.E(0,0,0,P)
            )
            problem.set_objective( 'max', S )
            problem.solve(solver=self.solver) 
            print("Max value of Svetlichny:", S.value)
        else: 
            print("Svetlichny only defined for ma=mb=me=2 and kx=ky=kz=2")

    
    def inequality_k3(self, P):
        """
        Calculates the I_3,3 inequlity given the prob distribution
    
        Parameters
        ----------
        P : pc.RealVariable
            Matrix of size (ma*mb*me)x(kx*ky*kz) with the probabilities P(abe|xyz)
    
        Returns
        -------
        I: double
            Value of the inequality
        """
        return (
            self.E(1,1,2,P) + self.E(1,2,1,P) + self.E(2,1,1,P)
            -2*(self.E(1,0,0,P) + self.E(0,1,0,P) + self.E(0,0,1,P))
            +self.E(1,1,0,P) + self.E(0,1,1,P) + self.E(1,0,1,P)
            -(self.E(2,2,0,P) + self.E(2,0,2,P) +self.E(0,2,2,P) )
            + self.E(2,1,0,P) + self.E(2,0,1,P) + self.E(1,0,2,P)
            +self.E(1,2,0,P) + self.E(0,1,2,P) + self.E(0,2,1,P)
            +2*self.E(0,0,0,P) + 4*self.E(1,1,1,P) - self.E(2,2,2,P)
        )/8
    
    def solve_I3_3(self):
        """
        Solves the tripartite I3_3 modular inequality optimization problem. Local bound is 1.
    
        Parameters
        ----------
        None
    
        Returns
        -------
        None
            Prints the minimal value of the inequality if the system dimensions
            match the required settings. Otherwise, prints a message indicating
            the function is undefined for other parameters.
        """
        if (self.kx ==3 and self.ky == 3 and self.kz ==3 and self.ma ==2 and self.mb ==2 and self.me ==2):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb*self.me, self.kx*self.ky*self.kz), lower=0, upper=1)
            P_twin = pc.RealVariable("P_TWIN", (self.ma**2*self.mb**2*self.me, self.kx**2*self.ky**2*self.kz), lower=0, upper=1)
            #self.add_ns_constraints( P_twin, problem)ç
            self.get_P_twin(P_twin, problem)
            self.add_ns_constraints_P(P,problem)
            self.normalization_twin(P_twin, problem)
            self.relate_P_twin_P(P_twin, P, problem)
      
      
            I = self.inequality_k3(P)
    
            problem.set_objective('max', I) 
            problem.solve(solver=self.solver)  
            print("Max value of the inequality:", I.value)
        else:
            print("I^3_3 only defined for ma=mb=me=2 and kx=ky=kz=3")


    def PA0(self, x, P):
        """
        Computes the marginal probability P(a=0 | x).
    
        Parameters
        ----------
        x : int
            Setting of Alice.
        P : array-like
            Array containing joint probabilities P(a, b | x, y).
    
        Returns
        -------
        float
            Marginal probability of Alice obtaining outcome 0 given setting x.
        """
        return sum(
            pc.sum(P[self.pos_ab(0, b, x, y)] for b in range(self.mb))
            for y in range(self.ky)
        ) / self.ky
    
    
    def PA1(self, x, P):
        """
        Computes the marginal probability P(a=1 | x).
    
        Parameters
        ----------
        x : int
            Setting of Alice.
        P : array-like
            Array containing joint probabilities P(a, b | x, y).
    
        Returns
        -------
        float
            Marginal probability of Alice obtaining outcome 1 given setting x.
        """
        return sum(
            sum(P[self.pos_ab(1, b, x, y)] for b in range(self.mb))
            for y in range(self.ky)
        ) / self.ky
    
    
    def PB0(self, y, P):
        """
        Computes the marginal probability P(b=0 | y).
    
        Parameters
        ----------
        y : int
            Setting of Bob.
        P : array-like
            Array containing joint probabilities P(a, b | x, y).
    
        Returns
        -------
        float
            Marginal probability of Bob obtaining outcome 0 given setting y.
        """
        return sum(
            sum(P[self.pos_ab(a, 0, x, y)] for a in range(self.ma))
            for x in range(self.kx)
        ) / self.kx
    
    
    def PB1(self, y, P):
        """
        Computes the marginal probability P(b=1 | y).
    
        Parameters
        ----------
        y : int
            Setting of Bob.
        P : array-like
            Array containing joint probabilities P(a, b | x, y).
    
        Returns
        -------
        float
            Marginal probability of Bob obtaining outcome 1 given setting y.
        """
        return sum(
            sum(P[self.pos_ab(a, 1, x, y)] for a in range(self.ma))
            for x in range(self.kx)
        ) / self.kx
    
    
    def P00(self, x, y, P):
        """
        Returns the joint probability P(a=0, b=0 | x, y).
    
        Parameters
        ----------
        x : int
            Setting of Alice.
        y : int
            Setting of Bob.
        P : array-like
            Array containing joint probabilities P(a, b | x, y).
    
        Returns
        -------
        float
            Probability of outcome (a=0, b=0) for settings x, y.
        """
        return P[self.pos_ab(0, 0, x, y)]

    
    def solve_CHSH(self):
        """
        Calculates the maximum value of the CHSH inequality. Only valid for k = 2, m=2.
        In this version a in (0,1) so the local bound is 0

        Parameters
        ----------
        none

        Returns
        -------
        prints the maximum value of the CHSH inequality.
        """

        if(self.kx ==2 and self.ky == 2 and self.ma ==2 and self.mb ==2):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb*self.me, self.kx*self.ky*self.kz), lower=0, upper=1)
            P_ab = pc.RealVariable("P_ab", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            P_twin = pc.RealVariable("P_TWIN", (self.ma**2*self.mb**2*self.me, self.kx**2*self.ky**2*self.kz), lower=0, upper=1)
            #self.add_ns_constraints( P_twin, problem)ç
            self.get_P_twin(P_twin, problem)
            self.add_ns_constraints_P(P,problem)
            self.normalization_twin(P_twin, problem)
            self.relate_P_twin_P(P_twin, P, problem)
            self.constrain_BR1(P, P_ab, problem)
            if self.BR == 'True':
                self.constrain_BR2(P, problem)
            CHSH = self.P00(0,0,P_ab) + self.P00(0,1,P_ab) + self.P00(1,0,P_ab) - self.P00(1,1,P_ab) - self.PA0(0,P_ab) - self.PB0(0,P_ab)

    
            problem.set_objective('max', CHSH)
            problem.solve(solver=self.solver)  # SCS/ECOS también funcionan
            print("Max value of CHSH:", CHSH.value)
            """  
            with open("resultS_chsh.txt", "w", encoding="utf-8") as f:
                for x in range(self.kx):
                    for y in range(self.ky):
                        for z in range(self.kz):
                            for a in range(self.ma):
                                for b in range(self.mb):
                                    for e in range(self.me):
                                        line = f"P(a={a}, b={b}, e={e} | x={x}, y={y}, z={z}) = {P[self.pos(a,b,e,x,y,z)].value}\n"
                                        f.write(line)
            """
            return P_twin, P_ab, P
            
        else:
            print("CHSH for TW with A,B,E only defined for ma=mb=2 and kx=ky=2 and Bound Randomness constrain")

    
    
    def solve_I3322(self):
        """
        Calculates the maximum value of the I3322 inequality. Only valid for k = 3, m=2.
        In this version a in (0,1) so the local bound is 0
    
        Parameters
        ----------
        none
    
        Returns
        -------
        prints the maximum value of the I3322 inequality.
        """
        if (self.kx ==3 and self.ky == 3 and self.ma ==2 and self.mb ==2):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb*self.me, self.kx*self.ky*self.kz), lower=0, upper=1)
            P_ab = pc.RealVariable("P_ab", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            P_twin = pc.RealVariable("P_TWIN", (self.ma**2*self.mb**2*self.me, self.kx**2*self.ky**2*self.kz), lower=0, upper=1)
            #self.add_ns_constraints( P_twin, problem)
            print('ns constrain done')
            self.get_P_twin(P_twin, problem)
            self.add_ns_constraints_P(P,problem)
            self.normalization_twin(P_twin, problem)
            self.relate_P_twin_P(P_twin, P, problem)
            self.constrain_BR1(P, P_ab, problem)
            print('Bound randomness ')
            if self.BR == 'True':
                self.constrain_BR2_vers2(P_twin, problem)
                print('Bound randomness done')
    
    
            I3322= (
                    + self.P00(0,0,P_ab) + self.P00(0,1,P_ab) + self.P00(0,2,P_ab)
                    + self.P00(1,0,P_ab) + self.P00(1,1,P_ab) - self.P00(1,2,P_ab)
                    + self.P00(2,0,P_ab) - self.P00(2,1,P_ab)
                    - self.PA0(0,P_ab)
                    - 2*self.PB0(0,P_ab) - self.PB0(1,P_ab)
                )
    
            problem.set_objective('max', I3322)
            problem.solve(solver=self.solver) 
            print("Max value of I3322:", I3322.value)

            return P_twin, P_ab, P

        else:
            print("I3322 for TW with A,B,E only defined for ma=mb=2 and kx=ky=3")
        
    def solve_I4422(self):
        """
        Calculates the maximum value of the I4422 inequality. Only valid for k = 4, m=2.
        https://arxiv.org/pdf/quant-ph/0306129
        
        Parameters
        ----------
        none
    
        Returns
        -------
        prints the maximum value of the I4422 inequality.
        """
        if (self.kx ==4 and self.ky == 4 and self.ma ==2 and self.mb ==2):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb*self.me, self.kx*self.ky*self.kz), lower=0, upper=1)
            P_ab = pc.RealVariable("P_ab", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            P_twin = pc.RealVariable("P_TWIN", (self.ma**2*self.mb**2*self.me, self.kx**2*self.ky**2*self.kz), lower=0, upper=1)
            #self.add_ns_constraints( P_twin, problem)
            print('ns constrain done')
            self.get_P_twin(P_twin, problem)
            self.add_ns_constraints_P(P,problem)
            self.normalization_twin(P_twin, problem)
            self.relate_P_twin_P(P_twin, P, problem)
            self.constrain_BR1(P, P_ab, problem)
            print('Bound randomness ')
            if self.BR == 'True':
                #self.constrain_BR2_vers2(P_twin, problem)
                self.constrain_BR2(P_twin, problem)
                print('Bound randomness done')
            

            I4422= (
                + self.P00(0,0,P_ab) + self.P00(0,1,P_ab) + self.P00(0,2,P_ab) + self.P00(0,3,P_ab)
                + self.P00(1,0,P_ab) + self.P00(1,1,P_ab) + self.P00(1,2,P_ab)- self.P00(1,3,P_ab)
                + self.P00(2,0,P_ab) + self.P00(2,1,P_ab) - self.P00(2,2,P_ab)
                + self.P00(3,0,P_ab) - self.P00(3,1,P_ab)

                - self.PA0(0,P_ab) 
                - 3*self.PB0(0,P_ab) - 2*self.PB0(1,P_ab) - self.PB0(2,P_ab)
            )

            problem.set_objective('max', I4422)
            problem.solve(solver=self.solver) 
            print("Maximum value of I4422:", I4422.value)
            return P_twin, P_ab, P
        else:
            print("I4422 only defined for ma=mb=2 and kx=ky=4")
    
    def organize_index(self, line):
        d = np.zeros((4,4))
        c = np.zeros(4)
        e = np.zeros(4)
        for i in range(4):
            for j in range(4):
                d[i,j] = line[i*4+j]
            c[i] = line[16+i]
            e[i] = line[20+i]
        return d, c, e
    
    def solve_I5522(self):
        """
        Calculates the maximum value of the I5522 inequality. Only valid for k = 4, m=2.
        https://arxiv.org/pdf/quant-ph/0603094
        
        Parameters
        ----------
        none
    
        Returns
        -------
        prints the maximum value of the I4422 inequality.
        """
        if (self.kx ==5 and self.ky == 5 and self.ma ==2 and self.mb ==2):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb*self.me, self.kx*self.ky*self.kz), lower=0, upper=1)
            P_ab = pc.RealVariable("P_ab", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            #self.add_ns_constraints( P_twin, problem, index_map)
            self.add_ns_constraints_P( P, problem)
            print('ns constrains created')
            #self.normalization_P(P, problem)
            self.normalization_twin(P_twin, problem, index_map)
            print('normalization constrains created')
            self.relate_P_twin_P(P_twin, P, problem, index_map)
            print('P = sum Pt constrains created')
            self.constrain_BR1(P, P_ab, problem)
            print('BR1 constrains created')
            if self.BR == 'True':
                self.constrain_BR2(P, problem)
                print('BR2 constrains created')
            

            I5522= (
                + self.P00(0,0,P_ab) + self.P00(0,1,P_ab) + self.P00(0,2,P_ab) + self.P00(0,3,P_ab) +  self.P00(0,4,P_ab)
                + self.P00(1,0,P_ab) + self.P00(1,1,P_ab) + self.P00(1,2,P_ab)+ self.P00(1,3,P_ab) -  self.P00(1,4,P_ab)
                + self.P00(2,0,P_ab) + self.P00(2,1,P_ab) + self.P00(2,2,P_ab) - self.P00(2,3,P_ab)
                + self.P00(3,0,P_ab) + self.P00(3,1,P_ab) - self.P00(3,2,P_ab)
                + self.P00(4,0,P_ab) - self.P00(4,1,P_ab)

                - self.PA0(0,P_ab) 
                -4*self.PB0(0,P_ab) - 3*self.PB0(1,P_ab) - 2*self.PB0(2,P_ab) - self.PB0(3,P_ab)
            )

            problem.set_objective('max', I5522)
            problem.solve(solver=self.solver) 
            print("Maximum value of I5522:", I5522.value)
            return P_twin, P_ab, P, index_map
        else:
            print("I5522 only defined for ma=mb=2 and kx=ky=4")
    
    def organize_index(self, line):
        d = np.zeros((4,4))
        c = np.zeros(4)
        e = np.zeros(4)
        for i in range(4):
            for j in range(4):
                d[i,j] = line[i*4+j]
            c[i] = line[16+i]
            e[i] = line[20+i]
        return d, c, e
        
    def solve_choose(self, index):
        """
        Calculates the maximum value of an inequality, given by the index of paper
        https://doi.org/10.1103/PhysRevA.99.022104
        
        Parameters
        ----------
        index : int
            Index of the inequality to be solved. 0 for CHSH, 1 for I3322,... (See paper, where 1-->0, 2-->1...)
    
        Returns
        -------
        prints the maximum value of the inequality.
        """
        line = []

        with open('DATA_APS.txt', "r") as f:
            for i, linea in enumerate(f):
                if i == index:
                    line = [int(x) for x in linea.split()]
                    break
        d,c,e = self.organize_index(line)
        
        if (self.kx ==2 and self.ky == 2 and self.ma ==2 and self.mb ==2 and index == 0):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb*self.me, self.kx*self.ky*self.kz), lower=0, upper=1)
            P_ab = pc.RealVariable("P_ab", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.add_ns_constraints( P_twin, problem, index_map)
            self.normalization_twin(P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
            self.constrain_BR1(P, P_ab, problem)
            if self.BR == 'True':
                self.constrain_BR2(P, problem)

            
            CHSH = d[0,0]*self.P00(0,0,P_ab) + d[0,1]*self.P00(0,1,P_ab) + d[1,0]*self.P00(1,0,P_ab) + d[1,1]*self.P00(1,1,P_ab) + c[0]*self.PA0(0,P_ab) + c[1]*self.PA0(1,P_ab) + e[0]*self.PB0(0,P_ab) + e[1]*self.PB0(1,P_ab)

    
            problem.set_objective('max', CHSH)
            problem.solve(solver=self.solver)  # SCS/ECOS también funcionan
            print("Max value of CHSH:", CHSH.value)


        if (self.kx ==3 and self.ky == 3 and self.ma ==2 and self.mb ==2 and index ==1):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb*self.me, self.kx*self.ky*self.kz), lower=0, upper=1)
            P_ab = pc.RealVariable("P_ab", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.normalization_twin(P_twin, problem, index_map)
            #self.normalization_P(P, problem)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
            self.constrain_BR1(P, P_ab, problem)
            print('Bound randomness')
            if self.BR == 'True':
                self.constrain_BR2_vers2(P_twin, problem)
                print('Bound randomness done')
            self.add_ns_constraints( P_twin, problem, index_map)

            print(n_vars)
    
    
            I3322= (
                    + d[0,0]*self.P00(0,0,P_ab) + d[0,1]*self.P00(0,1,P_ab) + d[0,2]*self.P00(0,2,P_ab)
                    + d[1,0]*self.P00(1,0,P_ab) + d[1,1]*self.P00(1,1,P_ab) + d[1,2]*self.P00(1,2,P_ab)
                    + d[2,0]*self.P00(2,0,P_ab) + d[2,1]*self.P00(2,1,P_ab) + d[2,2]*self.P00(2,2,P_ab)
                    + c[0]*self.PA0(0,P_ab) + c[1]*self.PA0(1,P_ab) + c[2]*self.PA0(2,P_ab)
                    + e[0]*self.PB0(0,P_ab) + e[1]*self.PB0(1,P_ab) + e[2]*self.PB0(2,P_ab)
                )
    
            problem.set_objective('max', I3322)
            problem.solve(solver=self.solver) 
            print("Max value of I3322:", I3322.value)

        if (self.kx ==4 and self.ky == 3 and self.ma ==2 and self.mb ==2 and index >=2 and index <5):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb*self.me, self.kx*self.ky*self.kz), lower=0, upper=1)
            P_ab = pc.RealVariable("P_ab", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.normalization_twin(P_twin, problem, index_map)
            #self.normalization_P(P, problem)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
            self.constrain_BR1(P, P_ab, problem)
            #self.add_ns_constraints_P( P, problem)
            if self.BR == 'True':
                self.constrain_BR2(P, problem)
            self.add_ns_constraints( P_twin, problem, index_map)

            print(n_vars)


    
    
            I4322= (
                    + d[0,0]*self.P00(0,0,P_ab) + d[0,1]*self.P00(0,1,P_ab) + d[0,2]*self.P00(0,2,P_ab) 
                    + d[1,0]*self.P00(1,0,P_ab) + d[1,1]*self.P00(1,1,P_ab) + d[1,2]*self.P00(1,2,P_ab)
                    + d[2,0]*self.P00(2,0,P_ab) + d[2,1]*self.P00(2,1,P_ab) + d[2,2]*self.P00(2,2,P_ab) 
                    + d[3,0]*self.P00(3,0,P_ab) + d[3,1]*self.P00(3,1,P_ab) + d[3,2]*self.P00(3,2,P_ab) 
                    + c[0]*self.PA0(0,P_ab) + c[1]*self.PA0(1,P_ab) + c[2]*self.PA0(2,P_ab) + c[3]*self.PA0(3,P_ab)
                    + e[0]*self.PB0(0,P_ab) + e[1]*self.PB0(1,P_ab) + e[2]*self.PB0(2,P_ab) 
                )
    
            problem.set_objective('max', I4322)
            problem.solve(solver=self.solver) 
            print("Max value of I4322:", I4322.value)

        if (self.kx ==4 and self.ky == 4 and self.ma ==2 and self.mb ==2 and index >=5):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb*self.me, self.kx*self.ky*self.kz), lower=0, upper=1)
            P_ab = pc.RealVariable("P_ab", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            #self.add_ns_constraints( P_twin, problem, index_map)
            self.add_ns_constraints_P( P, problem)
            #self.normalization_P(P, problem)
            self.normalization_twin(P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
            self.constrain_BR1(P, P_ab, problem)
            if self.BR == 'True':
                self.constrain_BR2(P, problem)

        
            I4422= (
                    + d[0,0]*self.P00(0,0,P_ab) + d[0,1]*self.P00(0,1,P_ab) + d[0,2]*self.P00(0,2,P_ab) + d[0,3]*self.P00(0,3,P_ab)
                    + d[1,0]*self.P00(1,0,P_ab) + d[1,1]*self.P00(1,1,P_ab) + d[1,2]*self.P00(1,2,P_ab) + d[1,3]*self.P00(1,3,P_ab)
                    + d[2,0]*self.P00(2,0,P_ab) + d[2,1]*self.P00(2,1,P_ab) + d[2,2]*self.P00(2,2,P_ab) + d[2,3]*self.P00(2,3,P_ab)
                    + d[3,0]*self.P00(3,0,P_ab) + d[3,1]*self.P00(3,1,P_ab) + d[3,2]*self.P00(3,2,P_ab) + d[3,3]*self.P00(3,3,P_ab)
                    + c[0]*self.PA0(0,P_ab) + c[1]*self.PA0(1,P_ab) + c[2]*self.PA0(2,P_ab) + c[3]*self.PA0(3,P_ab)
                    + e[0]*self.PB0(0,P_ab) + e[1]*self.PB0(1,P_ab) + e[2]*self.PB0(2,P_ab) + e[3]*self.PB0(3,P_ab)
                )

            problem.set_objective('max', I4422)
            problem.solve(solver=self.solver) 
            print("Maximum value of I4422:", I4422.value)
     
    

#%%
if __name__ == "__main__":
    #print(f"Solving inequality with index {i}...")
    tw = TW_AB_E(ma=2, mb=2, me=2, kx=2, ky=2, kz=2, solver = 'mosek', BR = 'False')
    #tw.solve_choose(index=26)
    #tw = TW_AB_E(ma=2, mb=2, me=2, kx=4, ky=4, kz=4, solver = 'mosek', BR = 'True')
    #tw.solve_choose(index=i)
    P_t_sol, P_ab_sol, P_sol = tw.solve_CHSH()
    #tw.solve_choose(index=1)
#%%
x, y , xt, yt = 0, 0, 0, 0
print(x,y)
contx = 0
conty = 0
for z in range(tw.kz):
    print(f"z=({z}):")
    xt = contx
    yt = conty
    print(f"z=({xt}, {yt}), z = {tw.pos_z(xt, yt)}:")
    conty+=1
    if conty == tw.ky:
        conty = 0
        contx +=1
    for a_target in range(tw.ma):
        for b_target in range(tw.mb):
            sum_result = 0
            sum_result_aux = 0
            for e in range(tw.me):
                P_E = 0
                for a in range(tw.ma):
                    for b in range(tw.mb):
                            P_E +=P_sol[tw.pos(a,b,e,x,y,z)].value
                P_BE = 0
                for a in range(tw.ma):
                    P_BE += P_sol[tw.pos(a,b_target,e,x,y,z)].value
                P_AE = 0
                for b in range(tw.mb):
                        P_AE += P_sol[tw.pos(a_target,b,e,x,y,z)].value
                P_ABE = P_sol[tw.pos(a_target,b_target,e,x,y,z)].value
                if P_E > 0:
                    sum_result += P_BE*P_AE/P_E
                    sum_result_aux += P_ABE

        

            P_AB = 0
            for e in range(tw.me):
                P_AB += P_sol[tw.pos(a_target,b_target,e,x,y,z)].value
            
            #print(f"a={a_target}, b={b_target}: P_AB - P_E = {P_AB - P_E}")
            print(f"a={a_target}, b={b_target}: P_AB - sum = {P_AB - sum_result}")
            print(f"a={a_target}, b={b_target}: P_AB - sum_aux = {P_AB - sum_result_aux}")
            print(f"a={a_target}, b={b_target}: sum - sum_aux= {sum_result - sum_result_aux}")
#%%
for x in range(tw.kx):
    for y in range(tw.ky):
        z = tw.pos_z(x,y)
        for a in range(tw.ma):
            for b in range(tw.mb):
                e = tw.pos_e(a,b)
                P_AB = P_ab_sol[tw.pos_ab(a_target,e,x,y)].value
                PE = 0
                for ap in range(tw.ma):
                    for bp in range(tw.mb):
                        PE += P_sol[tw.pos(ap,bp,e,x,y,z)].value
                print(P_AB-PE)
#%%
for x in range(tw.kx):
    for y in range(tw.ky):
        for zx in range(tw.kx):
            for zy in range(tw.ky):
                z = tw.pos_z(zx,zy)
                for a in range(tw.ma):
                    for b in range(tw.mb):
                        for ea in range(tw.ma):
                            for eb in range(tw.mb):
                                e = tw.pos_e (ea,eb)
                                print(P_sol[tw.pos(a,b,e,x,y,z)].value- P_sol[tw.pos(ea,eb,tw.pos_e(a,b),zx,zy,tw.pos_z(x,y))].value)

#%%
x, y , yt = 0, 0, 0
xt = x
z = tw.pos_z(xt,yt)
for a_target in range(tw.ma):
    for b_target in range(tw.mb):
        sum_result = 0
        sum_result_aux = 0
        for e in range(tw.me):
            for at in range(tw.ma):
                for bt in range(tw.mb):
                    P_AtBtE = 0
                    for a in range(tw.ma):
                        for b in range(tw.mb):
                            P_AtBtE +=tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,e,x,xt,y,yt,z).value
                    P_B = 0
                    for a in range(tw.ma):
                        for e in range(tw.me):
                            P_B += P_sol[tw.pos(a,b_target,e,x,y,z)].value
                    P_A = 0
                    for b in range(tw.mb):
                        for e in range(tw.me):
                            P_A += P_sol[tw.pos(a_target,b,e,x,y,z)].value

                    P_ABE = P_sol[tw.pos(a_target,b_target,e,x,y,z)].value
                    if P_AtBtE > 0:
                        sum_result += P_B*P_A/P_AtBtE
                        sum_result_aux += P_ABE

            

        P_AB = 0
        for e in range(tw.me):
            P_AB += P_sol[tw.pos(a_target,b_target,e,x,y,z)].value
        
        #print(f"a={a_target}, b={b_target}: P_AB - P_E = {P_AB - P_E}")
        print(f"a={a_target}, b={b_target}: P_AB - sum = {P_AB - sum_result}")
        print(f"a={a_target}, b={b_target}: P_AB - sum_aux = {P_AB - sum_result_aux}")
        print(f"a={a_target}, b={b_target}: sum - sum_aux= {sum_result - sum_result_aux}")
#%% CHECK fiNE K=3
x = 0
xt = 1
ex = 2
y = 0
yt = 1
ey = 2
z = tw.pos_z(ex,ey)

for a in range(tw.ma):
    for b in range(tw.mb):
        P00_1 = 0
        P00_2 = P_ab_sol[tw.pos_ab(a,b,x,y)].value
        for at in range(tw.ma):
            for bt in range(tw.mb):
                for e in range(tw.me):
                    P00_1+= tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,e,x,xt,y,yt,z).value
        print(f'a = {a}, b = {b}, x = {x}, y = {y}:',P00_2-P00_1)
for at in range(tw.ma):
    for bt in range(tw.mb):
        P11_1 = 0
        P11_2 = P_ab_sol[tw.pos_ab(at,bt,xt,yt)].value
        for a in range(tw.ma):
            for b in range(tw.mb):
                for e in range(tw.me):
                    P11_1+= tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,e,x,xt,y,yt,z).value
        print(f'a = {at}, b = {bt}, x = {xt}, y = {yt}:',P11_2-P11_1)
for ea in range(tw.ma):
    for eb in range(tw.mb):
        P22_1 = 0
        P22_2 = P_ab_sol[tw.pos_ab(ea,eb,ex,ey)].value
        for a in range(tw.ma):
            for b in range(tw.mb):
                for at in range(tw.ma):
                    for bt in range(tw.mb):
                        P22_1+= tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,tw.pos_e(ea,eb),x,xt,y,yt,z).value
        print(f'a = {ea}, b = {eb}, x = {ex}, y = {ey}:',P22_2-P22_1)
for at in range(tw.ma):
    for b in range(tw.mb):
        P10_1 = 0
        P10_2 = P_ab_sol[tw.pos_ab(at,b,xt,y)].value
        for a in range(tw.ma):
            for bt in range(tw.mb):
                for e in range(tw.me):
                    P10_1+= tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,e,x,xt,y,yt,z).value
        print(f'a = {at}, b = {b}, x = {xt}, y = {y}:',P10_2-P10_1)

for a in range(tw.ma):
    for bt in range(tw.mb):
        P01_1 = 0
        P01_2 = P_ab_sol[tw.pos_ab(a,bt,x,yt)].value
        for at in range(tw.ma):
            for b in range(tw.mb):
                for e in range(tw.me):
                    P01_1+= tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,e,x,xt,y,yt,z).value
        print(f'a = {a}, b = {bt}, x = {x}, y = {yt}:',P01_2-P01_1)

for a in range(tw.ma):
    for eb in range(tw.mb):
        P02_1 = 0
        P02_2 = P_ab_sol[tw.pos_ab(a,eb,x,ey)].value
        for ea in range(tw.ma):
            for b in range(tw.mb):
                for at in range(tw.ma):
                    for bt in range(tw.mb):
                        P02_1+= tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,tw.pos_e(ea,eb),x,xt,y,yt,z).value
        print(f'a = {a}, b = {eb}, x = {x}, y = {ey}:',P02_2-P02_1)

for at in range(tw.ma):
    for eb in range(tw.mb):
        P12_1 = 0
        P12_2 = P_ab_sol[tw.pos_ab(at,eb,xt,ey)].value
        for ea in range(tw.ma):
            for b in range(tw.mb):
                for a in range(tw.ma):
                    for bt in range(tw.mb):
                        P12_1+= tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,tw.pos_e(ea,eb),x,xt,y,yt,z).value
        print(f'a = {at}, b = {eb}, x = {xt}, y = {ey}:',P12_2-P12_1)
for ea in range(tw.ma):
    for b in range(tw.mb):
        P20_1 = 0
        P20_2 = P_ab_sol[tw.pos_ab(ea,b,ex,y)].value
        for a in range(tw.ma):
            for eb in range(tw.mb):
                for at in range(tw.ma):
                    for bt in range(tw.mb):
                        P20_1+= tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,tw.pos_e(ea,eb),x,xt,y,yt,z).value
        print(f'a = {ea}, b = {b}, x = {ex}, y = {y}:',P20_2-P20_1)

for ea in range(tw.ma):
    for bt in range(tw.mb):
        P21_1 = 0
        P21_2 = P_ab_sol[tw.pos_ab(ea,bt,ex,yt)].value
        for a in range(tw.ma):
            for eb in range(tw.mb):
                for at in range(tw.ma):
                    for b in range(tw.mb):
                        P21_1+= tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,tw.pos_e(ea,eb),x,xt,y,yt,z).value
        print(f'a = {ea}, b = {bt}, x = {ex}, y = {yt}:',P21_2-P21_1)

#%% Check Eve twin
x = 0
y = 0
xt = 0
yt = 0
ex = 0
ey = 0
ext = 0
eyt = 0
'''
for x in range(tw.kx):
    for y in range(tw.ky):
        for xt in range(tw.kx):
            for yt in range(tw.ky):
                for ex in range(tw.kx):
                    for ey in range(tw.ky):
                        for ext in range(tw.kx):
                            for eyt in range(tw.ky):
'''
for a in range(tw.ma):
    for b in range(tw.mb):
        Ptot = 0
        for ea in range (tw.ma):
            for eb in range (tw.mb):
                e = tw.pos_e(ea,eb)
                PE = 0
                for a_sum in range (tw.ma):
                    for b_sum in range(tw.mb):
                        PE+=P_sol[tw.pos(a_sum,b_sum,e,x,y,tw.pos_z(ex,ey))].value
                for at in range(tw.ma):
                    for bt in range(tw.mb):
                        for eat in range (tw.ma):
                            for ebt in range (tw.mb):
                                P1 = tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,e,x,xt,y,yt,tw.pos_z(ex,ey)).value
                                P2 = tw.get_P_twin(P_t_sol,index_map_sol,ea,eat,eb,ebt,e,ex,ext,ey,eyt,tw.pos_z(ex,ey)).value
                                Ptot += P1*P2/PE
        PAB = P_ab_sol[tw.pos_ab(a,b,x,y)].value
        
        #print(Ptot)
        #print(PAB)
        if abs(Ptot-PAB)>10e-9:
            print(f'a = {a}, b= {b}, x= {x}, y = {y}')
            print(Ptot-PAB)
#%%
#%% Check Eve twin
x = 0
y = 0
xt = 0
yt = 0

for a in range(tw.ma):
    for b in range(tw.mb):
        Ptot = 0
        for ea in range (tw.ma):
            for eb in range (tw.mb):
                e = tw.pos_e(ea,eb)
                PE = 0
                for a_sum in range (tw.ma):
                    for b_sum in range(tw.mb):
                        PE+=P_sol[tw.pos(a_sum,b_sum,e,x,y,tw.pos_z(ex,ey))].value
                for at in range(tw.ma):
                    for bt in range(tw.mb):
                        for eat in range (tw.ma):
                            for ebt in range (tw.mb):
                                P1 = 0
                                for e_sum in range(tw.me):
                                    P1 += tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,e_sum,x,xt,y,yt,tw.pos_z(ex,ey)).value
                                P2 = 0
                                for e_sum in range(tw.me):
                                    P2 += tw.get_P_twin(P_t_sol,index_map_sol,ea,eat,eb,ebt,e_sum,ex,ext,ey,eyt,tw.pos_z(ex,ey)).value
                                Ptot += P1*P2/PE
        PAB = P_ab_sol[tw.pos_ab(a,b,x,y)].value
        print(Ptot)
        print(PAB)
        print(Ptot-PAB)
                
#%%
if __name__ == "__main__":
    #print(f"Solving inequality with index {i}...")
    for i in [129,139]:
        print(f"Solving inequality with index {i}...")
        tw = TW_AB_E(ma=2, mb=2, me=2, kx=4, ky=4, kz=4, solver = 'mosek', BR = 'True')
        tw.solve_choose(index=i)
    #tw = TW_AB_E(ma=2, mb=2, me=2, kx=4, ky=4, kz=4, solver = 'mosek', BR = 'True')
    #tw.solve_choose(index=i)
    #P_t_sol, P_ab_sol, P_sol, index_map_sol = tw.solve_CHSH()
    #tw.solve_choose(index=1)
#%%
xt, yt = 0, 0
z = tw.pos_z(xt,yt)
    
for x in range(tw.kx):
    for y in range(tw.ky):
        for a in range (tw.ma):
            for b in range(tw.mb):
                PAB = 0
                PAB_aux = 0
                for at in range (tw.ma):
                    for bt in range(tw.mb):

                        q = 0
                        for a_prime in range(tw.ma):
                            for b_prime in range(tw.mb):
                                for e in range(tw.me):
                                    q+=tw.get_P_twin(P_t_sol,index_map_sol,a_prime,at,b_prime,bt,e,x,xt,y,yt,z).value
                        PA = 0
                        for b_prime in range(tw.mb):
                            for bt_aux in range (tw.mb):
                                for e in range(tw.me):
                                    PA+=tw.get_P_twin(P_t_sol,index_map_sol,a,at,b_prime,bt_aux,e,x,xt,y,yt,z).value
                        PA = PA/q
                        PB = 0
                        for a_prime in range(tw.ma):
                            for at_aux in range (tw.ma):
                                for e in range(tw.me):
                                    PB+=tw.get_P_twin(P_t_sol,index_map_sol,a_prime,at_aux,b,bt,e,x,xt,y,yt,z).value
                        PB = PB/q

                        PAABB = 0
     
                        for e in range(tw.me):
                            PAABB+=tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,e,x,xt,y,yt,z).value
                        PAABB = PAABB/q

                        PAB +=q*PA*PB
                        PAB_aux +=q*PAABB


                    

                P_AB = 0
                for e in range(tw.me):
                    P_AB += P_sol[tw.pos(a,b,e,x,y,z)].value
                
                #print(f"a={a_target}, b={b_target}: P_AB - P_E = {P_AB - P_E}")
                print(f"a={a}, b={b}: P_AB - sum = {P_AB - PAB}")
                print(f"a={a}, b={b}: P_AB - sum = {P_AB - PAB_aux}")
                print(f"a={a}, b={b}: P_AB - sum_aux = {PAB - PAB_aux}")
                print("\n")
#%%
# Ajustes de referencia fijos
xt_ref, yt_ref = 0, 0
z_ref = tw.pos_z(xt_ref, yt_ref)   # 0 si kx=ky=4

# 1. Calcular q(at,bt) una sola vez usando x=0,y=0 (por NS es independiente)
q = np.zeros((tw.ma, tw.mb))
for at in range(tw.ma):
    for bt in range(tw.mb):
        suma = 0
        for a in range(tw.ma):
            for b in range(tw.mb):
                for e in range(tw.me):
                    suma += tw.get_P_twin(P_t_sol, index_map_sol,
                                          a, at, b, bt, e,
                                          0, xt_ref, 0, yt_ref, z_ref).value
        q[at, bt] = suma

# 2. Funciones de respuesta (se pueden precalcular para cada x, at, etc.)
PA = np.zeros((tw.ma, tw.kx, tw.ma))   # PA[a, x, at]
for x in range(tw.kx):
    for at in range(tw.ma):
        if q[at, :].sum() == 0:   # si la combinación no aparece
            continue
        # marginal de A_x y A'_0 (con B en setting 0)
        for a in range(tw.ma):
            suma = 0
            for b in range(tw.mb):
                for bt in range(tw.mb):
                    for e in range(tw.me):
                        suma += tw.get_P_twin(P_t_sol, index_map_sol,
                                              a, at, b, bt, e,
                                              x, xt_ref, 0, yt_ref, z_ref).value
            # dividir por P(A'_0 = at) = sum_{bt} q(at,bt)
            PA[a, x, at] = suma / q[at, :].sum()

PB = np.zeros((tw.mb, tw.ky, tw.mb))   # PB[b, y, bt]
for y in range(tw.ky):
    for bt in range(tw.mb):
        if q[:, bt].sum() == 0:
            continue
        for b in range(tw.mb):
            suma = 0
            for a in range(tw.ma):
                for at in range(tw.ma):
                    for e in range(tw.me):
                        suma += tw.get_P_twin(P_t_sol, index_map_sol,
                                              a, at, b, bt, e,
                                              0, xt_ref, y, yt_ref, z_ref).value
            PB[b, y, bt] = suma / q[:, bt].sum()

# 3. Verificar para cada x,y,a,b
for x in range(tw.kx):
    for y in range(tw.ky):
        for a in range(tw.ma):
            for b in range(tw.mb):
                PAB_LHV = 0
                for at in range(tw.ma):
                    for bt in range(tw.mb):
                        PAB_LHV += q[at, bt] * PA[a, x, at] * PB[b, y, bt]
                # Valor “oficial” de P_ab
                P_AB = 0
                for e in range(tw.me):
                    P_AB += P_sol[tw.pos(a, b, e, x, y, z_ref)].value
                diff = abs(P_AB - PAB_LHV)
                if diff > 1e-10:
                    print(f"x={x},y={y},a={a},b={b}: diff={diff}")
#%%
for x in range(tw.kx):
    for y in range(tw.ky):
        #for y_prime in range(tw.ky):
        z = tw.pos_z(x,y)
        for a_target in range (tw.ma):
            for b_target in range (tw.mb):
                et = tw.pos_e(a_target,b_target)
                P_E = 0
                for a in range(tw.ma):
                    for b in range(tw.mb):
                            P_E +=P_sol[tw.pos(a,b,et,x,y,z)].value
                P_AB = 0
                for e in range(tw.me):
                    P_AB += P_sol[tw.pos(a_target,b_target,e,x,y,z)].value
                print(P_AB)
                #print(P_sol[tw.pos(a_target,b_target,et,x,y,z)].value - P_AB)
#%%
x = 1
xt = 0
y = 1
yt = 1
z = 2
for a in range (tw.ma):
    for at in range (tw.ma):
        for b in range (tw.mb):
            for bt in range (tw.mb):
                for e in range (tw.me):
                    P = tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,e,x,xt,y,yt,z).value
                    if P<10e-10: 
                        P=0
                    print(f'P(a = {a}, at = {at}, b= {b}, bt = {bt}, e = {e}) = ',P)
#%%
for x in range(tw.kx):
    for y in range(tw.ky):
        for z in range(tw.kz):
            for e in range(tw.me):
                P_E = 0
                for a in range(tw.ma):
                    for b in range(tw.mb):
                        P_E +=P_sol[tw.pos(a,b,e,x,y,z)].value
                print(f'P(e = {e}|z = {z}) = {P_E}')
#%%
xref = 0
yref = 0
for a_target in range (tw.ma):
    for b_target in range (tw.mb):
        for x in range(tw.kx):
            for y in range(tw.ky):
                for z in range (tw.kz):
                    P_ABref = 0
                    for e in range(tw.me):
                        P_ABref += P_sol[tw.pos(a_target,b_target,e,xref,yref,z)].value
                    for x in range(tw.kx):
                        for y in range(tw.ky):
                            P_AB = 0
                            for e in range(tw.me):
                                P_AB += P_sol[tw.pos(a_target,b_target,e,x,y,z)].value
                            print(f'P(a = {a}, b = {b}|x = {x} y = {y} z = {z}) - P(a = {a}, b = {b}|z = {z}) = {P_ABref-P_AB}')
#%%
x, y , xt, yt = 0, 0, 1, 1
contx = 0
conty = 0
for z in range(tw.kz):
    print(f"z=({z}):")
    xt = contx
    yt = conty
    print(f"z=({xt}, {yt}), z = {tw.pos_z(xt, yt)}:")
    conty+=1
    if conty == tw.ky:
        conty = 0
        contx +=1
    for a_target in range(tw.ma):
        for b_target in range(tw.mb):
            sum_result = 0
            sum_result_aux = 0
            for att in range(tw.kx):
                for btt in range(tw.ky):
                    e = tw.pos_e(att,btt)
                    P_E = 0
                    for a in range(tw.ma):
                        for b in range(tw.mb):
                            for at in range (tw.ma):
                                for bt in range (tw.ma):
                                    P_E +=tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,bt,e,x,xt,y,yt,z).value
                    P_BE = 0
                    for a in range(tw.ma):
                        for at in range (tw.ma):
                            for bt in range (tw.ma):
                                P_BE += tw.get_P_twin(P_t_sol,index_map_sol,a,at,b_target,bt,e,x,xt,y,yt,z).value
                    P_AE = 0
                    for b in range(tw.mb):
                        for at in range (tw.ma):
                            for bt in range (tw.ma):
                                P_AE += tw.get_P_twin(P_t_sol,index_map_sol,a_target,at,b,bt,e,x,xt,y,yt,z).value
                    P_BtE = 0
                    for a in range(tw.ma):
                        for at in range (tw.ma):
                            for b in range (tw.ma):
                                P_BE += tw.get_P_twin(P_t_sol,index_map_sol,a,at,b,btt,e,x,xt,y,yt,z).value
                    P_AEt = 0
                    for b in range(tw.mb):
                        for a in range (tw.ma):
                            for bt in range (tw.ma):
                                P_AEt += tw.get_P_twin(P_t_sol,index_map_sol,a,att,b,bt,e,x,xt,y,yt,z).value
                    P_ABE = 0
                    for at in range (tw.ma):
                            for bt in range (tw.ma):
                                P_ABE += tw.get_P_twin(P_t_sol,index_map_sol,a_target,at,b_target,bt,e,x,xt,y,yt,z).value
                    P_ABttE = tw.get_P_twin(P_t_sol,index_map_sol,a_target,att,b_target,btt,e,x,xt,y,yt,z).value
                    if P_E > 0:
                        sum_result += P_ABE*P_BtE*P_AEt/(P_E**2)
                        sum_result_aux += P_ABttE

        

            P_AB = 0
            for e in range(tw.me):
                P_AB += P_sol[tw.pos(a_target,b_target,e,x,y,z)].value
            print(f"a={a_target}, b={b_target}: P_AB - sum = {P_AB - sum_result}")
            print(f"a={a_target}, b={b_target}: P_AB - sum_aux = {P_AB - sum_result_aux}")
            print(f"a={a_target}, b={b_target}: sum - sum_aux= {sum_result - sum_result_aux}")

#%%
x, y = 0, 0
for xt in range(tw.kx):
    for yt in range(tw.ky):
        z = tw.pos_z(xt, yt)
        print(f"z=({xt}, {yt}), z = {z}:")
        for a_target in range(tw.ma):
            for b_target in range(tw.mb):
                sum_result = 0
                sum_result_aux = 0
                for at in range(tw.ma):
                    for bt in range(tw.mb):
                        P_E = 0
                        e = tw.pos_e(at, bt)
                        for a in range(tw.ma):
                            for b in range(tw.mb):
                                P_E = tw.get_P_twin(P_t_sol, index_map_sol, a, at, b, bt, e, x, xt, y, yt, z).value
                        P_BE = 0
                        for a in range(tw.ma):
                            P_BE += tw.get_P_twin(P_t_sol, index_map_sol, a, at, b_target, bt, e, x, xt, y, yt, z).value
                        P_AE = 0
                        for b in range(tw.mb):
                            P_AE += tw.get_P_twin(P_t_sol, index_map_sol, a_target, at, b, bt, e, x, xt, y, yt, z).value
                        P_ABE = tw.get_P_twin(P_t_sol, index_map_sol, a_target, at, b_target, bt, e, x, xt, y, yt, z).value
                        if P_E > 0:
                            sum_result += P_BE*P_AE/P_E
                            sum_result_aux += P_ABE

            

                P_AB = 0
                for e in range(tw.me):
                    P_AB += P_sol[tw.pos(a_target,b_target,e,x,y,z)].value
                print(f"a={a_target}, b={b_target}: P_AB - sum = {P_AB - sum_result}")
                print(f"a={a_target}, b={b_target}: P_AB - sum_aux = {P_AB - sum_result_aux}")
                print(f"a={a_target}, b={b_target}: sum - sum_aux= {sum_result - sum_result_aux}")

#%%

print(tw.get_P_twin(P_t_sol, index_map_sol, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2).value)
print(tw.get_P_twin(P_t_sol, index_map_sol, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2).value)

norm = 0
for x in range(tw.kx):
    for y in range(tw.ky):
        for z in range(tw.kz):
            for a in range(tw.ma):
                for b in range(tw.mb):
                    Pe=0
                    for e in range(tw.me):
                        Pe += P_sol[tw.pos(a,b,e,x,y,z)].value
                    print(f"x={x}, y={y}, z={z}, a={a}, b={b}, e={e}: {P_ab_sol[tw.pos_ab(a,b,x,y)].value - Pe} ")

#tw.solve_choose(index=5)
#tw = TW_AB_E(ma=2, mb=2, me=2, kx=4, ky=4, kz=4, solver = 'mosek', BR = 'True')
#tw.solve_I4422()

#tw = TW_AB_E(ma=2, mb=2, me=2, kx=4, ky=4, kz=4, solver = 'mosek', BR = 'True')
#tw.solve_I4422()
    
# %%Comprobar TW swapping
for x in tqdm(range(tw.kx), desc="Verificando TW"):
    for xt in range(tw.kx):
        for y in range(tw.ky):
            for yt in range(tw.ky):
               for z in range(tw.kz):
                    for a in range(tw.ma):
                        for atwin in range(tw.ma):
                            for b in range(tw.mb):
                                for btwin in range(tw.mb):
                                    for e in range(tw.me):
                                        P1 = tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x, xt, y, yt, z).value
                                        P2 = tw.get_P_twin(P_t_sol, index_map_sol, atwin, a, btwin, b, e, xt, x, yt, y, z).value
                                        #print(P1-P2)
                                        if np.abs(P1 - P2) > 1e-10:
                                            print(f"TW violated for x={x}, xt={xt}, y={y}, yt={yt}, z={z}, a={a}, atwin={atwin}, b={b}, btwin={btwin}, e={e}: P1 = {P1}, P2 = {P2}")
# %% Comprobar NS 
for x in range(tw.kx):
    for x2 in range(tw.kx):
        for xt in range(tw.kx):
            for xt2 in range(tw.kx):
                for y in range(tw.ky):
                    for y2 in range(tw.ky):
                        for yt in range(tw.ky):
                            for yt2 in range(tw.ky):
                                for z in range(tw.kz):
                                    for z2 in range(tw.kz):
                                        for a in range(tw.ma):
                                            P1 = 0
                                            P2 = 0
                                            for atwin in range(tw.ma):
                                                for b in range(tw.mb):
                                                    for btwin in range(tw.mb):
                                                        for e in range(tw.me):
                                                            P1 += tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x, xt, y, yt, z).value
                                                            P2 += tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x, xt2, y2, yt2, z2).value
                                                    # print(P1-P2)
                                            if np.abs(P1 - P2) > 1e-10:
                                                print('a=' + str(a) + ', P1-P2=' + str(P1 - P2))
                                        for atwin in range(tw.ma):
                                            P1 = 0
                                            P2 = 0
                                            for a in range(tw.ma):
                                                for b in range(tw.mb):
                                                    for btwin in range(tw.mb):
                                                        for e in range(tw.me):
                                                            P1 += tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x, xt, y, yt, z).value
                                                            P2 += tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x2, xt, y2, yt2, z2).value
                                                    # print(P1-P2)
                                            if np.abs(P1 - P2) > 1e-10:
                                                print('atwin=' + str(atwin) + ', P1-P2=' + str(P1 - P2))
                                        for b in range(tw.mb):
                                            P1 = 0
                                            P2 = 0
                                            for atwin in range(tw.ma):
                                                for a in range(tw.ma):
                                                    for btwin in range(tw.mb):
                                                        for e in range(tw.me):
                                                            P1 += tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x, xt, y, yt, z).value
                                                            P2 += tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x2, xt2, y, yt2, z2).value
                                                        # print(P1-P2)
                                            if np.abs(P1 - P2) > 1e-10:
                                                print('b=' + str(b) + ', P1-P2=' + str(P1 - P2))
                                        for btwin in range(tw.mb):
                                            P1 = 0
                                            P2 = 0
                                            for atwin in range(tw.ma):
                                                for a in range(tw.ma):
                                                    for b in range(tw.mb):
                                                        for e in range(tw.me):
                                                            P1 += tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x, xt, y, yt, z).value
                                                            P2 += tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x2, xt2, y2, yt, z2).value
                                                        # print(P1-P2)
                                            if np.abs(P1 - P2) > 1e-10:
                                                print('btwin=' + str(btwin) + ', P1-P2=' + str(P1 - P2))
                                        for e in range(tw.me):
                                            P1 = 0
                                            P2 = 0
                                            for atwin in range(tw.ma):
                                                for a in range(tw.ma):
                                                    for btwin in range(tw.mb):
                                                        for b in range(tw.mb):
                                                            P1 += tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x, xt, y, yt, z).value
                                                            P2 += tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x2, xt2, y2, yt2, z).value
                                                        # print(P1-P2)
                                            if np.abs(P1 - P2) > 1e-10:
                                                print('e=' + str(e) + ', P1-P2=' + str(P1 - P2))
# %%
print('Min value: ',pc.min(P_t_sol))
for x in range(tw.kx):
    for xt in range(tw.kx):
        for y in range(tw.ky):
            for yt in range(tw.ky):
               for z in range(tw.kz):
                    P = 0
                    for a in range(tw.ma):
                        for atwin in range(tw.ma):
                            for b in range(tw.mb):
                                for btwin in range(tw.mb):
                                    for e in range(tw.me):
                                        P+= tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x, xt, y, yt, z).value
                    print(P)
# %% Relation P, Ptwin
for x in range(tw.kx):
    for xt in range(tw.kx):
        for y in range(tw.ky):
            for yt in range(tw.ky):
               for z in range(tw.kz):
                    for a in range(tw.ma):
                        for b in range(tw.mb): 
                            for e in range(tw.me):
                                P = 0
                                for atwin in range(tw.ma):
                                    for btwin in range(tw.mb):
                                        P+= tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x, xt, y, yt, z).value
                                P2 = P_sol[tw.pos(a,b,e,x,y,z)].value
                                if abs(P-P2)>1e-10:
                                    print(P-P2)
                                    
# %% Check BR
for x in range(tw.kx):
    for y in range(tw.ky):
        for z in range(tw.kz):
            for a in range(tw.ma):
                for b in range(tw.mb): 
                    for e in range(tw.me):
                        #print(f'p(a = {a}, b= {b}, e = {e}|x = {x}, y = {y}, z = ({z})) = ', P_sol[tw.pos(a,b,e,x,y,z)].value)
                        if (z == tw.pos_z(x,y)):
                            #print(f'p(a = {a}, b= {b}, e = {e}|x = {x}, y = {y}, z = ({z})) = ', P_sol[tw.pos(a,b,e,x,y,z)].value)
                            if (e != tw.pos_e(a,b)):
                                print('Must be 0')
                                if (abs(P_sol[tw.pos(a,b,e,x,y,z)].value) < 10e-10):
                                    print('ok')
                                else:
                                    print('bad')

# %% BR1
for x in range(tw.kx):
    for y in range(tw.ky):
        for z in range(tw.kz):
            P1 = 0
            P2 = P_ab_sol[tw.pos_ab(a,b,x,y)].value
            for e in range (tw.me):
                P1+= P_sol[tw.pos(a,b,e,x,y,z)].value
            if abs(P2-P1)>10e-10:
                print(P-P1)


# %% NS
for x in range (tw.kx):
    for x2 in range (tw.kx):
        for y in range (tw.ky):
            for y2 in range (tw.ky):
                for a in range(tw.ma):
                    P1= 0
                    P2 = 0
                    for b in range (tw.mb):
                        P1 +=P_ab_sol[tw.pos_ab(a,b,x,y)].value
                        P2+= P_ab_sol[tw.pos_ab(a,b,x,y2)].value
                    if abs(P1-P2)>10e-10:
                        print(P1-P2)
        
                for b in range (tw.mb):
                    P1= 0
                    P2 = 0
                    for a in range (tw.ma):
                        P1 +=P_ab_sol[tw.pos_ab(a,b,x,y)].value
                        P2+= P_ab_sol[tw.pos_ab(a,b,x2,y)].value
                    if abs(P1-P2)>10e-10:
                        print(P1-P2)
    

# %%
for x in tqdm(range(tw.kx), desc="Verificando TW"):
    for xt in range(tw.kx):
        for y in range(tw.ky):
            for yt in range(tw.ky):
               for z in range(tw.kz):
                    for a in range(tw.ma):
                        for atwin in range(tw.ma):
                            for b in range(tw.mb):
                                for btwin in range(tw.mb):
                                    P1 = 0
                                    P2 = 0
                                    P3 = 0
                                    P4 = 0
                                    for e in range(tw.me):
                                        P3= tw.get_P_twin(P_t_sol, index_map_sol, a, atwin, b, btwin, e, x, xt, y, yt, z).value
                                        P4 = tw.get_P_twin(P_t_sol, index_map_sol, atwin, a, btwin, b, e, xt, x, yt, y, z).value
                                        P1 += P3
                                        P2 += P4
                                        if np.abs(P3 - P4) > 1e-10:
                                            print(f"TW violated for x={x}, xt={xt}, y={y}, yt={yt}, z={z}, a={a}, atwin={atwin}, b={b}, btwin={btwin}, e={e}: P1 = {P1}, P2 = {P2}")
                                    if np.abs(P1 - P2) > 1e-10:
                                        print(f"TW violated for x={x}, xt={xt}, y={y}, yt={yt}, z={z}, a={a}, atwin={atwin}, b={b}, btwin={btwin}, e={e}: P1 = {P1}, P2 = {P2}")
# %%
# Elegir configuraciones de referencia (pueden ser cualesquiera, por ejemplo 0,0)
x0, y0 = 0, 0
z0 = tw.pos_z(x0, y0)   # z0 = 0 en tu caso (kx=2,ky=2 => z0=0)

# Inicializar
rho = np.zeros((tw.ma, tw.mb))
pA = np.zeros((tw.ma, tw.ma, tw.mb))   # pA[a][a0][b0]
pB = np.zeros((tw.mb, tw.ma, tw.mb))   # pB[b][a0][b0]

# Calcular la distribución conjunta R(a,b,a0,b0) = P_twin(a,a0,b,b0, f(a0,b0) | x,x0,y,y0,z0)
# Para un x,y cualesquiera (por ejemplo, los que usaste en tu comprobación: x=1, y=0)
x, y = 1, 1   # o cualquier otro

for a0,b0 in product(range(tw.ma), range(tw.mb)):
    e0 = tw.pos_e(a0, b0)   # f(a0,b0)
    # Sumar sobre a,b para obtener rho(a0,b0)
    for a,b in product(range(tw.ma), range(tw.mb)):
        rho[a0,b0] += tw.get_P_twin(P_t_sol, index_map_sol, a, a0, b, b0, e0, x, x0, y, y0, z0).value
    # Calcular las condicionales
    for a,b in product(range(tw.ma), range(tw.mb)):
        val = tw.get_P_twin(P_t_sol, index_map_sol, a, a0, b, b0, e0, x, x0, y, y0, z0).value
        pA[a, a0, b0] += val   # sumando sobre b
        pB[b, a0, b0] += val   # sumando sobre a
    # Normalizar
    if rho[a0,b0] > 0:
        pA[:, a0, b0] /= rho[a0,b0]
        pB[:, a0, b0] /= rho[a0,b0]

# Reconstruir P(a,b|x,y)
P_recon = np.zeros((tw.ma, tw.mb))
for a,b in product(range(tw.ma), range(tw.mb)):
    for a0,b0 in product(range(tw.ma), range(tw.mb)):
        P_recon[a,b] += rho[a0,b0] * pA[a, a0, b0] * pB[b, a0, b0]

# Comparar con el valor original (de P_ab)
P_original = np.zeros((tw.ma, tw.mb))
for a,b in product(range(tw.ma), range(tw.mb)):
    P_original[a,b] = P_ab_sol[tw.pos_ab(a,b,x,y)].value

print("Diferencia máxima:", np.max(np.abs(P_recon - P_original)))
# %%
# Comprobar simetría twin para un caso típico
a, a0, b, b0, e = 0,1,0,1,0
x, x0, y, y0, z0 = 1,0,1,0,0
p1 = tw.get_P_twin(P_t_sol, index_map_sol, a, a0, b, b0, e, x, x0, y, y0, z0)
p2 = tw.get_P_twin(P_t_sol, index_map_sol, a0, a, b0, b, e, x0, x, y0, y, z0)
print(f"Diferencia simetría: {abs(p1-p2)}")
# %%
import numpy as np
from itertools import combinations

print("=" * 80)
print("VERIFICACIÓN EXHAUSTIVA DE LOCALIDAD")
print("=" * 80)

# 1. Extraer P_AB de los datos (marginal sobre Eve)
P_AB = np.zeros((tw.ma, tw.mb, tw.kx, tw.ky))
for x in range(tw.kx):
    for y in range(tw.ky):
        for a in range(tw.ma):
            for b in range(tw.mb):
                prob = 0
                for e in range(tw.me):
                    prob += P_sol[tw.pos(a, b, e, x, y, 0)].value
                P_AB[a, b, x, y] = prob

# 2. Verificar no-señalización de P_AB
print("\n1. Verificando no-señalización de P_AB:")
ns_violations = False
for x1, x2 in combinations(range(tw.kx), 2):
    for y in range(tw.ky):
        for a in range(tw.ma):
            marg1 = sum(P_AB[a, b, x1, y] for b in range(tw.mb))
            marg2 = sum(P_AB[a, b, x2, y] for b in range(tw.mb))
            if abs(marg1 - marg2) > 1e-10:
                print(f"  Violación NS-A: x1={x1}, x2={x2}, y={y}, a={a}: {marg1:.10f} vs {marg2:.10f}")
                ns_violations = True

for y1, y2 in combinations(range(tw.ky), 2):
    for x in range(tw.kx):
        for b in range(tw.mb):
            marg1 = sum(P_AB[a, b, x, y1] for a in range(tw.ma))
            marg2 = sum(P_AB[a, b, x, y2] for a in range(tw.ma))
            if abs(marg1 - marg2) > 1e-10:
                print(f"  Violación NS-B: y1={y1}, y2={y2}, x={x}, b={b}: {marg1:.10f} vs {marg2:.10f}")
                ns_violations = True

if not ns_violations:
    print("  ✓ P_AB es no-señalizante")

# 3. Verificar normalización
print("\n2. Verificando normalización:")
norm_violations = False
for x in range(tw.kx):
    for y in range(tw.ky):
        total = sum(P_AB[a, b, x, y] for a in range(tw.ma) for b in range(tw.mb))
        if abs(total - 1.0) > 1e-10:
            print(f"  Violación normalización: x={x}, y={y}: {total:.10f}")
            norm_violations = True
if not norm_violations:
    print("  ✓ P_AB está normalizada")

# 4. Calcular valores de expectación
print("\n3. Valores de expectación E(x,y) = P(00) - P(01) - P(10) + P(11):")
E = np.zeros((tw.kx, tw.ky))
for x in range(tw.kx):
    for y in range(tw.ky):
        E[x, y] = P_AB[0, 0, x, y] - P_AB[0, 1, x, y] - P_AB[1, 0, x, y] + P_AB[1, 1, x, y]

print("   ", end="")
for y in range(tw.ky):
    print(f"   y={y}   ", end="")
print()
for x in range(tw.kx):
    print(f"x={x} ", end="")
    for y in range(tw.ky):
        print(f" {E[x, y]:+.6f}", end="")
    print()

# 5. Verificar TODAS las desigualdades CHSH
print("\n4. Verificando CHSH para todos los pares de ajustes:")
chsh_violations = False
for x1, x2 in combinations(range(tw.kx), 2):
    for y1, y2 in combinations(range(tw.ky), 2):
        chsh1 = E[x1, y1] + E[x1, y2] + E[x2, y1] - E[x2, y2]
        chsh2 = E[x1, y1] + E[x1, y2] - E[x2, y1] + E[x2, y2]
        chsh3 = E[x1, y1] - E[x1, y2] + E[x2, y1] + E[x2, y2]
        chsh4 = -E[x1, y1] + E[x1, y2] + E[x2, y1] + E[x2, y2]
        
        max_chsh = max(abs(chsh1), abs(chsh2), abs(chsh3), abs(chsh4))
        
        if max_chsh > 2 + 1e-10:
            print(f"  ¡VIOLACIÓN! CHSH({x1},{x2};{y1},{y2}) = {max_chsh:.10f}")
            chsh_violations = True

if not chsh_violations:
    print("  ✓ Todas las desigualdades CHSH se satisfacen (|CHSH| ≤ 2)")

# 6. Verificar I3322 (si kx,ky ≥ 3)
if tw.kx >= 3 and tw.ky >= 3:
    print("\n5. Verificando I3322:")
    # I3322 = P(00|00) + P(00|01) + P(00|02) + P(00|10) + P(00|11) - P(00|12) + P(00|20) - P(00|21)
    #         - PA(0|0) - 2PB(0|0) - PB(0|1)
    I3322 = (P_AB[0,0,0,0] + P_AB[0,0,0,1] + P_AB[0,0,0,2] + 
             P_AB[0,0,1,0] + P_AB[0,0,1,1] - P_AB[0,0,1,2] + 
             P_AB[0,0,2,0] - P_AB[0,0,2,1])
    
    PA_0_0 = sum(P_AB[0, b, 0, y] for b in range(tw.mb) for y in range(tw.ky)) / tw.ky
    PB_0_0 = sum(P_AB[a, 0, x, 0] for a in range(tw.ma) for x in range(tw.kx)) / tw.kx
    PB_0_1 = sum(P_AB[a, 0, x, 1] for a in range(tw.ma) for x in range(tw.kx)) / tw.kx
    
    I3322 -= (PA_0_0 + 2*PB_0_0 + PB_0_1)
    print(f"  I3322 = {I3322:.10f} (cota local = 0)")
    if I3322 > 1e-10:
        print("  ¡VIOLACIÓN de I3322!")

# 7. Verificar I4422 (si kx,ky ≥ 4)
if tw.kx >= 4 and tw.ky >= 4:
    print("\n6. Verificando I4422:")
    I4422 = (P_AB[0,0,0,0] + P_AB[0,0,0,1] + P_AB[0,0,0,2] + P_AB[0,0,0,3] +
             P_AB[0,0,1,0] + P_AB[0,0,1,1] + P_AB[0,0,1,2] - P_AB[0,0,1,3] +
             P_AB[0,0,2,0] + P_AB[0,0,2,1] - P_AB[0,0,2,2] +
             P_AB[0,0,3,0] - P_AB[0,0,3,1])
    
    PA_0_0 = sum(P_AB[0, b, 0, y] for b in range(tw.mb) for y in range(tw.ky)) / tw.ky
    PB_0_0 = sum(P_AB[a, 0, x, 0] for a in range(tw.ma) for x in range(tw.kx)) / tw.kx
    PB_0_1 = sum(P_AB[a, 0, x, 1] for a in range(tw.ma) for x in range(tw.kx)) / tw.kx
    PB_0_2 = sum(P_AB[a, 0, x, 2] for a in range(tw.ma) for x in range(tw.kx)) / tw.kx
    
    I4422 -= (PA_0_0 + 3*PB_0_0 + 2*PB_0_1 + PB_0_2)
    print(f"  I4422 = {I4422:.10f} (cota local = 0)")
    if I4422 > 1e-10:
        print("  ¡VIOLACIÓN de I4422!")
# %%
