#%%
import picos as pc
import numpy as np
from itertools import product, combinations
import scipy.sparse as sp
import gurobipy
#%%
class TW_abe ():
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
    
    """
    def __init__(self, ma=2, mb=2, me=2, kx=3, ky=3, kz=3, solver = 'mosek'):
        self.ma = ma
        self.mb = mb
        self.me = me
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.outputs = list(product(range(ma), range(mb), range(me)))
        self.inputs = list(product(range(kx), range(ky), range(kz)))
        self.outputs_twin = list(product(range(ma),range(ma), range(mb), range(mb),range(me),range(me)))
        self.inputs_twin = list(product(range(kx), range(kx),range(ky), range(ky), range(kz), range(kz)))
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

    def canonicalize(self, a, at, b, bt, e, et, x, xt, y, yt, z, zt):
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
        et : int
            Alternate labeling of outcome e
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
        zt : int
            Alternate labeling of setting z
    
        Returns
        -------
        tuple of int
            Canonical ordering of (a, at, b, bt, e, et, x, xt, y, yt, z, zt)
        """
        if (a, x) > (at, xt):
            a, at = at, a
            x, xt = xt, x
        if (b, y) > (bt, yt):
            b, bt = bt, b
            y, yt = yt, y
        if (e, z) > (et, zt):
            e, et = et, e
            z, zt = zt, z

        return (a, at, b, bt, e, et, x, xt, y, yt, z, zt)
    
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
            `(a, at, b, bt, e, et, x, xt, y, yt, z, zt)` to a unique integer index.
        counter : int
            Total number of distinct canonical tuples.
        """
        index_map = {}
        canon_map = {}
        counter = 0

        for (a, at, b, bt, e, et) in self.outputs_twin:
            for (x, xt, y, yt, z, zt) in self.inputs_twin:

                key = (a, at, b, bt, e, et, x, xt, y, yt, z, zt)
                canon = self.canonicalize(*key)

                if canon not in canon_map:
                    canon_map[canon] = counter
                    counter += 1

                index_map[key] = canon_map[canon]

        return index_map, counter
    
    def get_P_twin(self, P_twin, index_map,
               a, at, b, bt, e, et,
               x, xt, y, yt, z, zt):
        """
        Retrieves the probability from P_twin corresponding to a specific
        combination of outcomes and settings using the index map.
    
        Parameters
        ----------
        P_twin : array-like
            1D array or list containing probabilities for all canonical tuples.
        index_map : dict
            Dictionary mapping tuples `(a, at, b, bt, c, ct, x, xt, y, yt, z, zt)`
            to indices in `P_twin`.
        a, at : int
            Outcome and alternate labeling for party A
        b, bt : int
            Outcome and alternate labeling for party B
        c, ct : int
            Outcome and alternate labeling for party C
        x, xt : int
            Setting and alternate labeling for party A
        y, yt : int
            Setting and alternate labeling for party B
        z, zt : int
            Setting and alternate labeling for party C
    
        Returns
        -------
        float
            Probability corresponding to the given outcomes and settings in `P_twin`.
        """
        return P_twin[index_map[(a, at, b, bt, e, et, x, xt, y, yt, z, zt)]]


    def add_ns_constraints(self, P_twin, problem, index_map):
        """
        Adds No-Signaling constrains to P_twin
    
        Parameters
        ----------
        P_twin : pc.RealVariable
            Vector of reduced probability variables representing canonical twin tuples.
            Each element corresponds to a unique tuple (a, at, b, bt,e,et, x, xt, y, yt,z,zt).
        problem : pc.Problem
            Linear program object where constraints will be added. Must support 
            `add_constraint` method.
        index_map : dict
            Dictionary mapping each tuple (a, at, b, bt,e,et x, xt, y, yt,z,zt) to its index in P_twin.
    
        Returns
        -------
        none
        """
        # -------- No-signalling A --------
        for x in range(self.kx):
            for a in range(self.ma):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,e,e_twin,
                                     x,0, 0,0, 0, 0)
                    for a_twin, b, b_twin, e,e_twin in product(range(self.ma), range(self.mb), range(self.mb), range(self.me), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for xt,y, yt, z, zt in product(range(self.kx),range(self.ky), range(self.ky), range(self.kz), range(self.kz)):
                    terms = []
                    for a_twin,b, b_twin, e, e_twin in product(range(self.ma),range(self.mb), range(self.mb),range(self.me), range(self.me) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, e, e_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling A' --------
        for a_twin in range(self.ma):
            for xt in range(self.kx):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,e,e_twin,
                                     0,xt, 0,0, 0, 0)
                    for a, b, b_twin, e,e_twin in product(range(self.ma), range(self.mb), range(self.mb), range(self.me), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for x,y, yt, z, zt in product(range(self.kx),range(self.ky), range(self.ky), range(self.kz), range(self.kz)):
                    terms = []
                    for a,b, b_twin, e, e_twin in product(range(self.ma),range(self.mb), range(self.mb),range(self.me), range(self.me) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, e, e_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)


        # -------- No-signalling B --------
        for y in range(self.ky):
            for b in range(self.mb):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,e,e_twin,
                                     0,0, y,0, 0, 0)
                    for a,a_twin, b_twin, e,e_twin in product(range(self.ma), range(self.ma), range(self.mb), range(self.me), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for x, xt,yt, z, zt in product(range(self.kx), range(self.kx), range(self.ky),range(self.kz), range(self.kz)):
                    terms = []
                    for a, a_twin, b_twin,e, e_twin in product(range(self.ma), range(self.ma),range(self.mb),range(self.me), range(self.me) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, e, e_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)
        
        # -------- No-signalling B' --------
        for yt in range(self.ky):
            for b_twin in range(self.mb):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,e,e_twin,
                                     0,0, 0,yt, 0, 0)
                    for a,a_twin, b, e,e_twin in product(range(self.ma), range(self.ma), range(self.mb), range(self.me), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for x, xt,y, z, zt in product(range(self.kx), range(self.kx), range(self.ky),range(self.kz), range(self.kz)):
                    terms = []
                    for a, a_twin, b,e, e_twin in product(range(self.ma), range(self.ma),range(self.mb),range(self.me), range(self.me) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, e, e_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling C --------
        for e in range(self.me):
            for z in range(self.kz):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,e,e_twin,
                                     0,0, 0,0, z, 0)
                    for a,a_twin, b, b_twin, e_twin in product(range(self.ma), range(self.ma), range(self.mb), range(self.mb), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for x, xt, y, yt, zt in product(range(self.kx), range(self.kx), range(self.ky), range(self.ky), range(self.kz)):
                    terms = []
                    for a, a_twin, b, b_twin, e_twin in product(range(self.ma), range(self.ma),range(self.mb), range(self.mb), range(self.me) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, e, e_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling C' --------
        for zt in range(self.kz):
            for e_twin in range(self.me):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,e,e_twin,
                                     0,0, 0,0, 0, zt)
                    for a,a_twin, b, b_twin, e in product(range(self.ma), range(self.ma), range(self.mb), range(self.mb), range(self.me))
                ]
                ref = pc.sum(ref_terms)
                for x, xt, y, yt, z in product(range(self.kx), range(self.kx), range(self.ky), range(self.ky), range(self.kz)):
                    terms = []
                    for a, a_twin, b, b_twin, e in product(range(self.ma), range(self.ma),range(self.mb), range(self.mb), range(self.me) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, e, e_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)


   
    def normalization_twin(self, P_twin, problem, index_map):
        """
        Adds Normalization constrains to P_twin. sum_{a,a',b,b',e,e'}P_twin(aa'bb'ee'|xx'yy'zz') = 1 forall x,x',y,y',z,z'
    
        Parameters
        ----------
        P_twin : pc.RealVariable
            Vector of reduced probability variables representing canonical twin tuples.
            Each element corresponds to a unique tuple (a, at, b, bt,e,et, x, xt, y, yt,z,zt).
        problem : pc.Problem
            Linear program object where constraints will be added. Must support 
            `add_constraint` method.
        index_map : dict
            Dictionary mapping each tuple (a, at, b, bt,e,et, x, xt, y, yt,z,zt) to its index in P_twin.
    
        Returns
        -------
        none
        """
        for x, x_twin, y, y_twin, z, z_twin in self.inputs_twin:

            terms = []

            for a, a_twin, b, b_twin, e, e_twin in self.outputs_twin:
                val = self.get_P_twin(
                    P_twin, index_map,
                    a, a_twin, b, b_twin, e, e_twin,
                    x, x_twin, y, y_twin, z, z_twin
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
            Each element corresponds to a unique tuple (a, at, b, bt,e,et, x, xt, y, yt,z,zt).
        P : pc.RealVariable
            Matrix of size (ma*mb*mc)x(kx*ky*kz) with the probabilities P(ab|xy)
        problem : pc.Problem
            Linear program object where constraints will be added. Must support 
            `add_constraint` method.
        index_map : dict
            Dictionary mapping each tuple (a, at, b, bt, e, et, x, xt, y, yt, z, zt) to its index in P_twin.
    
        Returns
        -------
        none
        """
        for x, x_twin, y, y_twin, z, z_twin in self.inputs_twin:
            for a, b, e in self.outputs:

                terms = []

                for a_twin, b_twin, e_twin in self.outputs:
                    val = self.get_P_twin(
                        P_twin, index_map,
                        a, a_twin, b, b_twin, e, e_twin,
                        x, x_twin, y, y_twin, z, z_twin
                    )
                    terms.append(val)

                problem.add_constraint(
                    P[self.pos(a, b, e, x, y, z)] == pc.sum(terms)
                )
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
    
        
    def mod_expectation(self, x, y, z, P, d):
        """
        Computes the modular expectation value of the combination [A_x - B_y + E_z].
    
        Mathematically, this is:
        
            < [A_x - B_y + E_z] > = sum_k k * P( (a - b + e) mod d = k | x, y, z )
    
        Parameters
        ----------
        x : int
            Measurement setting for party A
        y : int
            Measurement setting for party B
        z : int
            Measurement setting for party E
        P : array-like
            Probability array indexed by `self.pos(a,b,e,x,y,z)`.
        d : int
            Number of outcomes
    
        Returns
        -------
        float
            Modular expectation value of [A_x - B_y + E_z].
        """
        expr = 0
        
        for a in range(self.ma):
            for b in range(self.mb):
                for e in range(self.me):
                    
                    k = (a - b + e) % d   
                    
                    expr += k * P[self.pos(a,b,e,x,y,z)]
                    
        return expr
    
    def mod_expectation_shiftA(self, x, y, z, P, d):
        """
        Computes the modular expectation value of the combination [B_y - A_{x+1} - C_z].
    
        The measurement setting of party A is shifted by +1 (modulo the number of settings).
        Mathematically, this is:
    
            < [B_y - A_{x+1} - E_z] > = sum_k k * P( (b - a - e) mod d = k | x+1, y, z )
    
        Parameters
        ----------
        x : int
            Measurement setting for party A
        y : int
            Measurement setting for party B
        z : int
            Measurement setting for party E
        P : array-like
            Probability array indexed by `self.pos(a,b,e,x,y,z)`.
        d : int
            Modulus for the combination (usually the number of outcomes)
    
        Returns
        -------
        float
            Modular expectation value of [B_y - A_{x+1} - E_z].
        """
        expr = 0
        x_shift = (x + 1) % self.kx
        
        for a in range(self.ma):
            for b in range(self.mb):
                for e in range(self.me):
                    
                    k = (b - a - e) % d
                    
                    expr += k * P[self.pos(a,b,e,x_shift,y,z)]
                    
        return expr
    
    def tripartite_inequality(self, P, d):
        """
        Computes the tripartite modular Bell-like inequality for a given probability distribution.
    
        The inequality is computed as a sum over shifted modular expectation values
        for parties A, B, and E. Specifically, it sums over alpha and beta settings,
        combining the standard and A-shifted modular expectation values:
    
            I = sum_{alpha, beta} < [A_x - B_y + E_z] > + < [B_y - A_{x+1} - E_z] >
    
        where x = alpha-1, y = (alpha+beta-1-1) % kx, z = beta-1.
    
        Parameters
        ----------
        P : array-like
            Probability array indexed by `self.pos(a,b,e,x,y,z)`.
        d : int
            Modulus for the combination (usually the number of outcomes).
    
        Returns
        -------
        float
            Value of the tripartite inequality I for the given probability distribution P.
        """
        
        I = 0
        
        for alpha in range(1,self.kx+1):
            for beta in range(1,self.kx+1):
                
                x = alpha
                y = (alpha + beta-1) % self.kx
                z = beta

                x = x-1
                y = y-1
                z = z-1
                
                I += self.mod_expectation(x, y, z, P, d)
                
                I += self.mod_expectation_shiftA(x, y, z, P, d)
        
        return I
    
    def solve_I3_3(self):
        """
        Solves the tripartite I3_3 modular inequality optimization problem. Local bound (min) is 3.
    
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

            d = self.ma
    
            I = self.tripartite_inequality(P, d=d)
    
            problem.set_objective('min', I) 
            problem.solve(solver=self.solver)  
            print("Min value of the inequality:", I.value)
        else:
            print("I^3_3 only defined for ma=mb=me=2 and kx=ky=kz=3")
            
    
#%%
if __name__ == "__main__":
    tw = TW_abe(ma=2, mb=2, me=2, kx=3, ky=3, kz=3, solver = 'gurobi')
    tw.solve_I3_3()
    
# %%
