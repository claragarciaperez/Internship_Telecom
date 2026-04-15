#%%
import picos as pc
import numpy as np
from itertools import product, combinations
import scipy.sparse as sp
import gurobipy
#%%
class TW_AB_E():
    """
    Class to solve inequalities given the TW.
    
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
        if BR:
            self.me = me**2
        else: 
            self.me = me
        self.kx = kx
        self.ky = ky
        if BR:
            self.kz = kz**2
        else: 
            self.kz = kz
        self.outputs = list(product(range(ma), range(mb), range(me)))
        self.inputs = list(product(range(kx), range(ky), range(kz)))
        self.outputs_twin = list(product(range(ma),range(ma), range(mb), range(mb),range(me)))
        self.inputs_twin = list(product(range(kx), range(kx),range(ky), range(ky), range(kz)))
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

    def canonicalize(self, a, at, b, bt, e, x, xt, y, yt, z):
        """
        Canonicalizes the tuple of outcomes and settings for a tripartite system.
    
        Parameters
        ----------
        a : int
            Outcome of party A
        at : int
            Alternate labeling of outcome a (for canonical ordering)
        b : int
            Outcome of party B
        bt : int
            Alternate labeling of outcome b
        e : int
            Outcome of party E (or C)

        x : int
            Setting of party A
        xt : int
            Alternate labeling of setting x
        y : int
            Setting of party B
        yt : int
            Alternate labeling of setting y
        z : int
            Setting of party E (or C)
    
        Returns
        -------
        tuple of int
            Canonical ordering of (a, at, b, bt, e, x, xt, y, yt, z)
        """
        if (a, x) > (at, xt):
            a, at = at, a
            x, xt = xt, x
        if (b, y) > (bt, yt):
            b, bt = bt, b
            y, yt = yt, y
            
        return (a, at, b, bt, e, x, xt, y, yt, z)
    
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
            `(a, at, b, bt, e, x, xt, y, yt, z) to a unique integer index.
        counter : int
            Total number of distinct canonical tuples.
        """
        index_map = {}
        canon_map = {}
        counter = 0

        for (a, at, b, bt, e) in self.outputs_twin:
            for (x, xt, y, yt, z) in self.inputs_twin:

                key = (a, at, b, bt, e, x, xt, y, yt, z)
                canon = self.canonicalize(*key)

                if canon not in canon_map:
                    canon_map[canon] = counter
                    counter += 1

                index_map[key] = canon_map[canon]

        return index_map, counter
    
    def get_P_twin(self, P_twin, index_map,
               a, at, b, bt, e,
               x, xt, y, yt, z):
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
        return P_twin[index_map[(a, at, b, bt, e, x, xt, y, yt, z)]]


    def add_ns_constraints(self, P_twin, problem, index_map):
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

   
    def normalization_twin(self, P_twin, problem, index_map):
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
        for x, x_twin, y, y_twin, z in self.inputs_twin:

            terms = []

            for a, a_twin, b, b_twin, e in self.outputs_twin:
                val = self.get_P_twin(
                    P_twin, index_map,
                    a, a_twin, b, b_twin, e,
                    x, x_twin, y, y_twin, z
                )
                terms.append(val)

            problem.add_constraint(pc.sum(terms) == 1)


    def relate_P_twin_P(self, P_twin, P, problem, index_map):
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
        for x, x_twin, y, y_twin, z in self.inputs_twin:
            for a, b, e in self.outputs:

                terms = []

                for a_twin, b_twin in product(range(self.ma), range(self.mb)):
                    val = self.get_P_twin(
                        P_twin, index_map,
                        a, a_twin, b, b_twin, e,
                        x, x_twin, y, y_twin, z
                    )
                    terms.append(val)

                problem.add_constraint(
                    P[self.pos(a, b, e, x, y, z)] == pc.sum(terms)
                )
    def constrain_BR1(self,P, P_ab, problem):
        for x,y,e in self.inputs:
            for a in range(self.ma):
                for b in range (self.mb):
                    terms = []
                    for e in range(self.me):
                        terms.append(P[self.pos(a,b,e,x,y,z)])
                    problem.add_constraint(pc.sum(terms) == P_ab[self.pos_ab(a,b,x,y)])
    def pos_z(self,x,y):
        return (x*self.ky + y)
    
    def pos_e(self,a,b):
        return (a*self.mb + b)

    def constrain_BR2(self,P, P_ab, problem):
        for x in range(self.kx):
            for y in range(self.ky):
                z = self.pos_z(x,y)
                for a ,b,e in self.outputs:
                    if (self.pos_e(a,b) != e):
                        problem.add_constraint(P[self.pos(a,b,e,x,y,z)] == 0)

    def E(self,x, y, z, P):
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
            sum(P[self.pos(0, b, x, y)] for b in range(self.mb))
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
            sum(P[self.pos(1, b, x, y)] for b in range(self.mb))
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
            sum(P[self.pos(a, 0, x, y)] for a in range(self.ma))
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
            sum(P[self.pos(a, 1, x, y)] for a in range(self.ma))
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
        return P[self.pos(0, 0, x, y)]

    
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
        if (self.kx ==2 and self.ky == 2 and self.ma ==2 and self.mb ==2 and BR = 'True'):
            problem = pc.Problem (verbosity =1)
            P = pc.RealVariable("P", (self.ma*self.mb*self.mc, self.kx*self.ky*self.kz), lower=0, upper=1)
            P_ab = pc.RealVariable("P_ab", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.add_ns_constraints( P_twin, problem, index_map)
            self.normalization_twin(P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
            self.constrain_BR1(P, P_ab, problem)
            self.constrain_BR2(P, P_ab, problem)
            
            CHSH = self.P00(0,0,P_ab) + self.P00(0,1,P_ab) + self.P00(1,0,P_ab) - self.P00(1,1,P_ab) - self.PA0(0,P_ab) - self.PB0(0,P_ab)

            problem.set_objective('max', CHSH)
            problem.solve(solver=self.solver)  # SCS/ECOS también funcionan
            print("Max value of CHSH:", CHSH.value)
            
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
            P = pc.RealVariable("P", (self.ma*self.mb*self.mc, self.kx*self.ky*self.kz), lower=0, upper=1)
            P_ab = pc.RealVariable("P_ab", (self.ma*self.mb, self.kx*self.ky), lower=0, upper=1)
            index_map, n_vars = self.build_index_map()
            P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
            self.normalization_twin(P_twin, problem, index_map)
            self.relate_P_twin_P(P_twin, P, problem, index_map)
            self.constrain_BR1(P, P_ab, problem)
            self.constrain_BR2(P, P_ab, problem)
            self.add_ns_constraints( P_twin, problem, index_map)
    
    
            I3322= (
                    + self.P00(0,0,P) + self.P00(0,1,P) + self.P00(0,2,P)
                    + self.P00(1,0,P) + self.P00(1,1,P) - self.P00(1,2,P)
                    + self.P00(2,0,P) - self.P00(2,1,P)
                    - self.PA0(0,P) 
                    - 2*self.PB0(0,P) - self.PB0(1,P)
                )
    
            problem.set_objective('max', I3322)
            problem.solve(solver=self.solver) 
            print("Max value of I3322:", I3322.value)
        else:
            print("I3322 for TW with A,B,E only defined for ma=mb=2 and kx=ky=3 and Bound Randomness constrain")
            
    
#%%
if __name__ == "__main__":
    #tw = TW_AB_E(ma=2, mb=2, me=2, kx=3, ky=3, kz=3, solver = 'mosek')
    #tw.solve_I3_3()
    tw = TW_AB_E(ma=2, mb=2, me=2, kx=3, ky=3, kz=3, solver = 'mosek', BR = 'True')
    tw.solve_I3322()
    
# %%
