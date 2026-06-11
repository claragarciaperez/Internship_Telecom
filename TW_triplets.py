#%%
import picos as pc
import numpy as np
from itertools import product, combinations
import scipy.sparse as sp
import gurobipy
from tqdm import tqdm
import time

#%%
class TW3_AB_E():
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
        self.outputs_twin = list(product(range(self.ma),range(self.ma),range(self.ma), range(self.mb), range(self.mb), range(self.mb),range(self.me)))
        self.inputs_twin = list(product(range(self.kx), range(self.kx), range(self.kx),range(self.ky), range(self.ky),range(self.ky), range(self.kz)))
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

    def canonicalize(self, a, at, att,b, bt,btt, e, x, xt, xtt,y, yt, ytt,z):
        """
        Canonicalizes the tuple of outcomes and settings for a tripartite system.
    
        Parameters
        ----------
        a : int
            Outcome of party A
        at : int
            Outcome Alice twin
        att: int
            Outcome second Alice twin
        b : int
            Outcome of party B
        bt : int
            Outcome Bob twin
        btt: int
            Outcome second Bob twin
        e : int
            Outcome of party E 

        x : int
            Setting of party A
        xt : int
            Seeting of A''
        xtt: int 
            Setting of A'''
        y : int
            Setting of party B
        yt : int
            Seeting of B''
        ytt: int 
            Setting of B'''
        z : int
            Setting of party E 
    
        Returns
        -------
        tuple of int
            Canonical ordering of (a, at, b, bt, e, x, xt, y, yt, z)
        """
        if (a, x) > (at, xt):
            a, at = at, a
            x, xtt = xtt, x
        if (a, x) > (att, xtt):
            a, att = att, a
            x, xtt = xtt, x
        if (at, xt) > (att, xtt):
            at, att = att, at
            xt, xtt = xtt, xt

        if (b, y) > (bt, yt):
            b, bt = bt, b
            y, ytt = ytt, y
        if (b, y) > (btt, ytt):
            b, btt = btt, b
            y, ytt = ytt, y
        if (bt, bt) > (btt, btt):
            bt, btt = btt, bt
            yt, ytt = ytt, yt
        
            
        return (a, at, att,b, bt, btt, e, x, xt, xtt, y, yt, ytt, z)
    
    def build_index_map(self):
        """
        Builds a mapping from raw outcome/setting tuples to canonical indices.
    
        This method iterates over all combinations of outputs and inputs 
        for a tripartite system (or "twin" system), canonicalizes each 
        combination using `self.canonicalize`, and assigns a unique index 
        to each distinct canonical form.
    
        Returns
        -------
        index_map : dict
            Dictionary mapping each original tuple
            `(a, at,att, b, bt,btt, e, x, xt,xtt, y, yt,ytt, z) to a unique integer index.
        counter : int
            Total number of distinct canonical tuples.
        """
        index_map = {}
        canon_map = {}
        counter = 0

        for (a, at,att, b, bt, btt, e) in self.outputs_twin:
            for (x, xt, xtt, y, yt, ytt, z) in self.inputs_twin:

                key = (a, at, att, b, bt, btt, e, x, xt, xtt,y, yt, ytt, z)
                canon = self.canonicalize(*key)

                if canon not in canon_map:
                    canon_map[canon] = counter
                    counter += 1

                index_map[key] = canon_map[canon]

        return index_map, counter
    
    def get_P_twin(self, P_twin, index_map,
               a, at, att, b, bt, btt, e,
               x, xt, xtt, y, yt, ytt, z):
        """
        Retrieves the probability from P_twin corresponding to a specific
        combination of outcomes and settings using the index map.
    
        Parameters
        ----------
        P_twin : array-like
            1D array or list containing probabilities for all canonical tuples.
        index_map : dict
            Dictionary mapping tuples `(a, at, att, b, bt,btt, e, x, xt,xtt, y, yt, ytt, z)`
            to indices in `P_twin`.
        a, at, att : int
            Outcome and alternate labeling for party A
        b, bt, btt : int
            Outcome and alternate labeling for party B
        e : int
            Outcome labeling for party E
        x, xt, xtt : int
            Setting and alternate labeling for party A
        y, yt, ytt : int
            Setting and alternate labeling for party B
        z : int
            Setting labeling for party E
    
        Returns
        -------
        float
            Probability corresponding to the given outcomes and settings in `P_twin`.
        """
        return P_twin[index_map[(a, at, att, b, bt, btt, e, x, xt, xtt, y, yt, ytt, z)]]
    
    

    def add_ns_constraints(self, P_twin, problem, index_map):
        """
        Adds No-Signaling constrains to P_twin
    
        Parameters
        ----------
        P_twin : pc.RealVariable
            Vector of reduced probability variables representing canonical twin tuples.
            Each element corresponds to a unique tuple (a, at, att, b, bt, btt, e, x, xt, xtt, y, yt, ytt,z).
        problem : pc.Problem
            Linear program object where constraints will be added. Must support 
            `add_constraint` method.
        index_map : dict
            Dictionary mapping each tuple (a, at, att, b, bt, btt, e, x, xt, xtt, y, yt, ytt,z) to its index in P_twin.
    
        Returns
        -------
        none
        """
          # -------- No-signalling A --------
        for x in range(self.kx):
            for a in range(self.ma):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,e,
                                     x,0, 0,0, 0)
                    for a_twin, b, b_twin, e in product(range(self.ma), range(self.mb), range(self.mb), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for xt,y, yt, z in product(range(self.kx),range(self.ky), range(self.ky), range(self.kz)):
                    terms = []
                    for a_twin,b, b_twin, e in product(range(self.ma),range(self.mb), range(self.mb),range(self.me) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, e,
                            x, xt, y, yt, z
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling A' --------
        for a_twin in range(self.ma):
            for xt in range(self.kx):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,e,
                                     0,xt, 0,0, 0)
                    for a, b, b_twin, e in product(range(self.ma), range(self.mb), range(self.mb), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for x,y, yt, z in product(range(self.kx),range(self.ky), range(self.ky), range(self.kz)):
                    terms = []
                    for a,b, b_twin, e in product(range(self.ma),range(self.mb), range(self.mb),range(self.me)):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, e,
                            x, xt, y, yt, z
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)


        # -------- No-signalling B --------
        for y in range(self.ky):
            for b in range(self.mb):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,e,
                                     0,0, y,0, 0)
                    for a,a_twin, b_twin, e in product(range(self.ma), range(self.ma), range(self.mb), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for x, xt,yt, z in product(range(self.kx), range(self.kx), range(self.ky),range(self.kz)):
                    terms = []
                    for a, a_twin, b_twin,e in product(range(self.ma), range(self.ma),range(self.mb),range(self.me) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, e,
                            x, xt, y, yt, z
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)
        
        # -------- No-signalling B' --------
        for yt in range(self.ky):
            for b_twin in range(self.mb):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,e,
                                     0,0, 0,yt, 0)
                    for a,a_twin, b, e in product(range(self.ma), range(self.ma), range(self.mb), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for x, xt,y, z in product(range(self.kx), range(self.kx), range(self.ky),range(self.kz)):
                    terms = []
                    for a, a_twin, b,e in product(range(self.ma), range(self.ma),range(self.mb),range(self.me)):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, e,
                            x, xt, y, yt, z
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling C --------
        for e in range(self.me):
            for z in range(self.kz):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,e,
                                     0,0, 0,0, z)
                    for a,a_twin, b, b_twin in product(range(self.ma), range(self.ma), range(self.mb), range(self.mb))
                ]
                ref = pc.sum(ref_terms)
                for x, xt, y, yt in product(range(self.kx), range(self.kx), range(self.ky), range(self.ky)):
                    terms = []
                    for a, a_twin, b, b_twin in product(range(self.ma), range(self.ma),range(self.mb), range(self.mb)):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, e,
                            x, xt, y, yt, z
                        )
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
    

   
    def normalization_twin(self, P_twin, problem, index_map):
        """
        Adds Normalization constrains to P_twin. sum_{a,a',a'',b,b',b''e}P_twin(aa'a''bb'b''e|xx'x''yy'y''z) = 1 forall x,x',x'',y,y',y'',z
    
        Parameters
        ----------
        P_twin : pc.RealVariable
            Vector of reduced probability variables representing canonical twin tuples.
            Each element corresponds to a unique tuple (a, at, att, b, bt, btt, e, x, xt, xtt, y, yt, ytt,z).
        problem : pc.Problem
            Linear program object where constraints will be added. Must support 
            `add_constraint` method.
        index_map : dict
            Dictionary mapping each tuple (a, at, att, b, bt, btt, e, x, xt, xtt, y, yt, ytt,z) to its index in P_twin.
    
        Returns
        -------
        none
        """
        for x, x_twin, xtt, y, y_twin, ytt, z in self.inputs_twin:

            terms = []

            for a, a_twin, att, b, b_twin, btt, e in self.outputs_twin:
                val = self.get_P_twin(
                    P_twin, index_map,
                    a, a_twin, att, b, b_twin, btt, e,
                    x, x_twin, xtt, y, y_twin, ytt, z
                )
                terms.append(val)

            problem.add_constraint(pc.sum(terms) == 1)
    
    def normalization_P(self, P, problem):
        for x, y, z in self.inputs:

            terms = []

            for a, b, e in self.outputs:
                val = P[self.pos(a, b, e, x, y, z)]
                terms.append(val)

            problem.add_constraint(pc.sum(terms) == 1)


    def relate_P_twin_P(self, P_twin, P, problem, index_map):
        """
        Adds constrains relating P with Ptwin such that sum_{a'a''b'b''e}P_twin(aa'a''bb'b''e|xx'x''yy'y''z) = P(abe|xyz) forall xx'x''yy'y''z
    
        Parameters
        ----------
        P_twin : pc.RealVariable
            Vector of reduced probability variables representing canonical twin tuples.
            Each element corresponds to a unique tuple (a, at, att, b, bt, btt, e, x, xt, xtt, y, yt, ytt,z).
        P : pc.RealVariable
            Matrix of size (ma*mb*me)x(kx*ky*kz) with the probabilities P(abc|xy)
        problem : pc.Problem
            Linear program object where constraints will be added. Must support 
            `add_constraint` method.
        index_map : dict
            Dictionary mapping each tuple (a, at, att, b, bt, btt, e, x, xt, xtt, y, yt, ytt,z) to its index in P_twin.
    
        Returns
        -------
        none
        """
        for x, x_twin, xtt, y, y_twin, ytt, z in self.inputs_twin:
            for a, b, e in self.outputs:

                terms = []

                for a_twin, att, b_twin, btt in product(range(self.ma),range(self.ma), range(self.mb), range(self.mb)):
                    val = self.get_P_twin(
                        P_twin, index_map,
                        a, a_twin, att, b, b_twin, btt,e,
                        x, x_twin, xtt, y, y_twin, ytt, z
                    )
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
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.add_ns_constraints( P_twin, problem, index_map)
            self.normalization_twin(P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
            
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
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.add_ns_constraints( P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
            self.normalization_twin(P_twin, problem, index_map)
    
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
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.add_ns_constraints( P_twin, problem, index_map)
            self.normalization_twin(P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
    
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
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            #self.add_ns_constraints( P_twin, problem, index_map)
            self.add_ns_constraints_P( P, problem)
            self.normalization_twin(P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
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
            return P_twin, P_ab, P, index_map
            
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
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.normalization_twin(P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
            self.constrain_BR1(P, P_ab, problem)
            if self.BR == 'True':
                self.constrain_BR2(P, problem)
            #self.add_ns_constraints( P_twin, problem, index_map)
            self.add_ns_constraints_P( P, problem)
            print(n_vars)
    
    
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

            return P_twin, P_ab, P, index_map
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
            return P_twin, P_ab, P, index_map
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
            #self.add_ns_constraints( P_twin, problem, index_map)
            self.normalization_twin(P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
            self.add_ns_constraints_P( P, problem)
            self.constrain_BR1(P, P_ab, problem)
            if self.BR == 'True':
                self.constrain_BR2(P, problem)

            
            CHSH = d[0,0]*self.P00(0,0,P_ab) + d[0,1]*self.P00(0,1,P_ab) + d[1,0]*self.P00(1,0,P_ab) + d[1,1]*self.P00(1,1,P_ab) + c[0]*self.PA0(0,P_ab) + c[1]*self.PA0(1,P_ab) + e[0]*self.PB0(0,P_ab) + e[1]*self.PB0(1,P_ab)

    
            problem.set_objective('max', CHSH)
            problem.solve(solver=self.solver)  # SCS/ECOS también funcionan
            print("Max value of CHSH:", CHSH.value)
            return P_twin, P_ab, P, index_map


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
            if self.BR == 'True':
                self.constrain_BR2(P, problem)
            #self.add_ns_constraints( P_twin, problem, index_map)
            self.add_ns_constraints_P( P, problem)

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

            return P_twin, P_ab, P, index_map

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

            return P_twin, P_ab, P, index_map

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

            return P_twin, P_ab, P, index_map
     
    

#%%
if __name__ == "__main__":
    tw = TW_AB_E(ma=2, mb=2, me=2, kx=3, ky=3, kz=3, solver = 'mosek', BR = 'False')
    P_t_sol, P_ab_sol, P_sol, index_map_sol =  tw.solve_I3322()
   
