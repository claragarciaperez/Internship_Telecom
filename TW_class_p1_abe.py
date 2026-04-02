#%%
import picos as pc
import numpy as np
from itertools import product, combinations
import scipy.sparse as sp
import gurobipy
#%%
class TW_abe ():
    def __init__(self, ma=2, mb=2, mc=2, kx=3, ky=3, kz=3, solver = 'osqp'):
        self.ma = ma
        self.mb = mb
        self.mc = mc
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.outputs = list(product(range(ma), range(mb), range(mc)))
        self.inputs = list(product(range(kx), range(ky), range(kz)))
        self.outputs_twin = list(product(range(ma),range(ma), range(mb), range(mb),range(mc),range(mc)))
        self.inputs_twin = list(product(range(kx), range(kx),range(ky), range(ky), range(kz), range(kz)))
        self.solver = solver
    
    def pos(self, a, b, c, x, y, z):
        return (a*(self.mb*self.mc) + b*self.mc + c, x*(self.ky*self.kz) + y*self.kz + z)

    def canonicalize(self, a, at, b, bt, c, ct, x, xt, y, yt, z, zt):
        if (a, x) > (at, xt):
            a, at = at, a
            x, xt = xt, x
        if (b, y) > (bt, yt):
            b, bt = bt, b
            y, yt = yt, y
        if (c, z) > (ct, zt):
            c, ct = ct, c
            z, zt = zt, z

        return (a, at, b, bt, c, ct, x, xt, y, yt, z, zt)
    
    def build_index_map(self):
        index_map = {}
        canon_map = {}
        counter = 0

        for (a, at, b, bt, c, ct) in self.outputs_twin:
            for (x, xt, y, yt, z, zt) in self.inputs_twin:

                key = (a, at, b, bt, c, ct, x, xt, y, yt, z, zt)
                canon = self.canonicalize(*key)

                if canon not in canon_map:
                    canon_map[canon] = counter
                    counter += 1

                index_map[key] = canon_map[canon]

        return index_map, counter
    
    def get_P_twin(self, P_twin, index_map,
               a, at, b, bt, c, ct,
               x, xt, y, yt, z, zt):
        return P_twin[index_map[(a, at, b, bt, c, ct, x, xt, y, yt, z, zt)]]


    def add_ns_constraints_optimized(self, P_twin, problem, index_map):
        # -------- No-signalling A --------
        for x, xt in product(range(self.kx), range(self.kx)):
            for a, a_twin in product(range(self.ma), range(self.ma)):
                ref = None
                for y, yt, z, zt in product(range(self.ky), range(self.ky), range(self.kz), range(self.kz)):
                    terms = []
                    for b, b_twin, c, c_twin in product(range(self.mb), range(self.mb),range(self.mc), range(self.mc) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, c, c_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    if ref is None:
                        ref = expr
                    else:
                        problem.add_constraint(expr == ref)

        # -------- No-signalling B --------
        for y, yt in product(range(self.ky), range(self.ky)):
            for b, b_twin in product(range(self.mb), range(self.mb)):

                ref = None
                for x, xt, z, zt in product(range(self.kx), range(self.kx), range(self.kz), range(self.kz)):
                    terms = []
                    for a, a_twin, c, c_twin in product(range(self.ma), range(self.ma),range(self.mc), range(self.mc) ):

                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, c, c_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    if ref is None:
                        ref = expr
                    else:
                        problem.add_constraint(expr == ref)

        # -------- No-signalling C --------
        for z, zt in product(range(self.kz), range(self.kz)):
            for c, c_twin in product(range(self.mc), range(self.mc)):

                ref = None
                for x, xt, y, yt in product(range(self.kx), range(self.kx), range(self.ky), range(self.ky)):
                    terms = []
                    for a, a_twin, b, b_twin in product(range(self.ma), range(self.ma),range(self.mb), range(self.mb) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, c, c_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    if ref is None:
                        ref = expr
                    else:
                        problem.add_constraint(expr == ref)


    def add_ns_constraints(self, P_twin, problem, index_map):
        # -------- No-signalling A --------
        for x in range(self.kx):
            for a in range(self.ma):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,c,c_twin,
                                     x,0, 0,0, 0, 0)
                    for a_twin, b, b_twin, c,c_twin in product(range(self.ma), range(self.mb), range(self.mb), range(self.mc), range(self.mc))
                ]
                ref = pc.sum(ref_terms)
                for xt,y, yt, z, zt in product(range(self.kx),range(self.ky), range(self.ky), range(self.kz), range(self.kz)):
                    terms = []
                    for a_twin,b, b_twin, c, c_twin in product(range(self.ma),range(self.mb), range(self.mb),range(self.mc), range(self.mc) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, c, c_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling A' --------
        for a_twin in range(self.ma):
            for xt in range(self.kx):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,c,c_twin,
                                     0,xt, 0,0, 0, 0)
                    for a, b, b_twin, c,c_twin in product(range(self.ma), range(self.mb), range(self.mb), range(self.mc), range(self.mc))
                ]
                ref = pc.sum(ref_terms)
                for x,y, yt, z, zt in product(range(self.kx),range(self.ky), range(self.ky), range(self.kz), range(self.kz)):
                    terms = []
                    for a,b, b_twin, c, c_twin in product(range(self.ma),range(self.mb), range(self.mb),range(self.mc), range(self.mc) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, c, c_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)


        # -------- No-signalling B --------
        for y in range(self.ky):
            for b in range(self.mb):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,c,c_twin,
                                     0,0, y,0, 0, 0)
                    for a,a_twin, b_twin, c,c_twin in product(range(self.ma), range(self.ma), range(self.mb), range(self.mc), range(self.mc))
                ]
                ref = pc.sum(ref_terms)
                for x, xt,yt, z, zt in product(range(self.kx), range(self.kx), range(self.ky),range(self.kz), range(self.kz)):
                    terms = []
                    for a, a_twin, b_twin,c, c_twin in product(range(self.ma), range(self.ma),range(self.mb),range(self.mc), range(self.mc) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, c, c_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)
        
        # -------- No-signalling B' --------
        for yt in range(self.ky):
            for b_twin in range(self.mb):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,c,c_twin,
                                     0,0, 0,yt, 0, 0)
                    for a,a_twin, b, c,c_twin in product(range(self.ma), range(self.ma), range(self.mb), range(self.mc), range(self.mc))
                ]
                ref = pc.sum(ref_terms)
                for x, xt,y, z, zt in product(range(self.kx), range(self.kx), range(self.ky),range(self.kz), range(self.kz)):
                    terms = []
                    for a, a_twin, b,c, c_twin in product(range(self.ma), range(self.ma),range(self.mb),range(self.mc), range(self.mc) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, c, c_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling C --------
        for c in range(self.mc):
            for z in range(self.kz):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,c,c_twin,
                                     0,0, 0,0, z, 0)
                    for a,a_twin, b, b_twin, c_twin in product(range(self.ma), range(self.ma), range(self.mb), range(self.mb), range(self.mc))
                ]
                ref = pc.sum(ref_terms)
                for x, xt, y, yt, zt in product(range(self.kx), range(self.kx), range(self.ky), range(self.ky), range(self.kz)):
                    terms = []
                    for a, a_twin, b, b_twin, c_twin in product(range(self.ma), range(self.ma),range(self.mb), range(self.mb), range(self.mc) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, c, c_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)

        # -------- No-signalling C' --------
        for zt in range(self.kz):
            for c_twin in range(self.mc):
                ref_terms = [
                    self.get_P_twin(P_twin, index_map, a, a_twin, b, b_twin,c,c_twin,
                                     0,0, 0,0, 0, zt)
                    for a,a_twin, b, b_twin, c in product(range(self.ma), range(self.ma), range(self.mb), range(self.mb), range(self.mc))
                ]
                ref = pc.sum(ref_terms)
                for x, xt, y, yt, z in product(range(self.kx), range(self.kx), range(self.ky), range(self.ky), range(self.kz)):
                    terms = []
                    for a, a_twin, b, b_twin, c in product(range(self.ma), range(self.ma),range(self.mb), range(self.mb), range(self.mc) ):
                        val = self.get_P_twin(
                            P_twin, index_map,
                            a, a_twin, b, b_twin, c, c_twin,
                            x, xt, y, yt, z, zt
                        )
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)


    def add_ns_constraints_P(self,P, problem):
        # -------- No-signalling A --------
        for x in range(self.kx):
            for a in range(self.ma):
                ref_terms = [
                    P[self.pos(a, b, c, x, 0, 0)]
                    for  b, c in product( range(self.mb), range(self.mc))
                ]
                ref = pc.sum(ref_terms)
                for y,  z in product(range(self.ky),range(self.kz)):
                    terms = []
                    for b, c in product(range(self.mb), range(self.mc) ):
                        val = P[self.pos(a, b, c, x, y, z)]
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)


        # -------- No-signalling B --------
        for y in range(self.ky):
            for b in range(self.mb):
                ref_terms = [
                    P[self.pos(a, b, c, 0, y, 0)]
                    for  a, c in product( range(self.ma), range(self.mc))
                ]
                ref = pc.sum(ref_terms)
                for x,  z in product(range(self.kx),range(self.kz)):
                    terms = []
                    for a, c in product(range(self.ma), range(self.mc) ):
                        val = P[self.pos(a, b, c, x, y, z)]
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)
        

        # -------- No-signalling C --------
        for z in range(self.kz):
            for c in range(self.mc):
                ref_terms = [
                    P[self.pos(a, b, c, 0, 0, z)]
                    for  a, b in product( range(self.ma), range(self.mb))
                ]
                ref = pc.sum(ref_terms)
                for x,  y in product(range(self.kx),range(self.ky)):
                    terms = []
                    for a, b in product(range(self.ma), range(self.mb) ):
                        val = P[self.pos(a, b, c, x, y, z)]
                        terms.append(val)
                    expr = pc.sum(terms)
                    problem.add_constraint(expr == ref)
    def normalization_twin(self, P_twin, problem, index_map):
        for x, x_twin, y, y_twin, z, z_twin in self.inputs_twin:

            terms = []

            for a, a_twin, b, b_twin, c, c_twin in self.outputs_twin:
                val = self.get_P_twin(
                    P_twin, index_map,
                    a, a_twin, b, b_twin, c, c_twin,
                    x, x_twin, y, y_twin, z, z_twin
                )
                terms.append(val)

            problem.add_constraint(pc.sum(terms) == 1)

    
    def normalization_P(self, P, problem):
        for x, y, z in self.inputs:

            terms = []

            for a, b, c in self.outputs:
                val = P[self.pos(a, b, c, x, y, z)]
                terms.append(val)

            problem.add_constraint(pc.sum(terms) == 1)

    def relate_P_twin_P(self, P_twin, P, problem, index_map):
        for x, x_twin, y, y_twin, z, z_twin in self.inputs_twin:
            for a, b, c in self.outputs:

                terms = []

                for a_twin, b_twin, c_twin in self.outputs:
                    val = self.get_P_twin(
                        P_twin, index_map,
                        a, a_twin, b, b_twin, c, c_twin,
                        x, x_twin, y, y_twin, z, z_twin
                    )
                    terms.append(val)

                problem.add_constraint(
                    P[self.pos(a, b, c, x, y, z)] == pc.sum(terms)
                )
        

    def solve_Mermin(self):
        def correlacion(x, y, z, P):
            return sum(((-1)**(a+b+c)) * P[self.pos(a, b, c, x, y, z)] for a in range(self.ma) for b in range(self.mb) for c in range(self.mc))        
        problem = pc.Problem (verbosity =1)
        problem = pc.Problem (verbosity =1)
        P = pc.RealVariable("P", (self.ma*self.mb*self.mc, self.kx*self.ky*self.kz), lower=0, upper=1)
        index_map, n_vars = self.build_index_map()
        P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
        self.add_ns_constraints( P_twin, problem, index_map)
        self.normalization_twin(P_twin, problem, index_map)
        self.relate_P_twin_P(P_twin, P, problem, index_map)
        #self.add_ns_constraints_P(P, problem)
        #self.normalization_P(P, problem)
        
        #M = correlacion(0,0,1, P) + correlacion(0,1,0, P) + correlacion(1,0,0, P) - correlacion(1,1,1, P )
        M = (correlacion(0,1,1, P) + correlacion(1,0,1, P) + correlacion(1,1,0, P) - correlacion(0,0,0, P))
        problem.set_objective('max', M)
        problem.solve(solver=self.solver)  # SCS/ECOS también funcionan
        print("Max value of Bell-Mermin:", M.value)
    
    def solve_Svetlichny(self):
        def correlacion(x,y,z, P):
            return sum(((-1)**(a+b+c)) * P[self.pos(a,b,c,x,y,z)] 
               for a in range(self.ma) for b in range(self.mb) for c in range(self.mc))
        
        problem = pc.Problem (verbosity =1)
        P = pc.RealVariable("P", (self.ma*self.mb*self.mc, self.kx*self.ky*self.kz), lower=0, upper=1)
        index_map, n_vars = self.build_index_map()
        P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
        #self.add_ns_constraints( P_twin, problem, index_map)
        #self.relate_P_twin_P(P_twin, P, problem, index_map)
        #self.normalization_twin(P_twin, problem, index_map)
        self.add_ns_constraints_P(P, problem)
        self.normalization_P(P, problem)
       # S = (
        #correlacion(0,0,0, P) + correlacion(0,0,1, P) + correlacion(0,1,0, P) - correlacion(0,1,1, P)
        #+ correlacion(1,0,0, P) - correlacion(1,0,1, P) - correlacion(1,1,0, P) - correlacion(1,1,1, P)
        #)

        S = (correlacion(0,0,1,P)
             + correlacion(0,1,0,P)
             + correlacion(1,0,0,P)
             -correlacion(1,1,1,P)
             +correlacion(1,1,0,P)
             +correlacion(1,0,1,P)
             +correlacion(0,1,1,P)
             -correlacion(0,0,0,P)
        )
        problem.set_objective( 'max', S )
        problem.solve(solver=self.solver)  # SCS/ECOS también funcionan
        print("Max value of Svetlichny:", S.value)# %%
    
    def mod_expectation(self, x, y, z, P, d):
        """
        Computes < [A_x - B_y + C_z] > 
        = sum_k k * P( (a - b + c mod d) = k | x,y,z )
        """
        expr = 0
        
        for a in range(self.ma):
            for b in range(self.mb):
                for c in range(self.mc):
                    
                    k = (a - b + c) % d   
                    
                    expr += k * P[self.pos(a,b,c,x,y,z)]
                    
        return expr
    
    def mod_expectation_shiftA(self, x, y, z, P, d):
        expr = 0
        x_shift = (x + 1) % self.kx
        
        for a in range(self.ma):
            for b in range(self.mb):
                for c in range(self.mc):
                    
                    k = (b - a - c) % d
                    
                    expr += k * P[self.pos(a,b,c,x_shift,y,z)]
                    
        return expr
    
    def tripartite_inequality(self, P, d):
        
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
    
    def solve_k3(self):
        problem = pc.Problem (verbosity =1)
        P = pc.RealVariable("P", (self.ma*self.mb*self.mc, self.kx*self.ky*self.kz), lower=0, upper=1)
        index_map, n_vars = self.build_index_map()
        P_twin = pc.RealVariable("P_twin_reduced", n_vars, lower=0, upper=1)
        self.add_ns_constraints( P_twin, problem, index_map)
        self.normalization_twin(P_twin, problem, index_map)
        self.relate_P_twin_P(P_twin, P, problem, index_map)
        self.add_ns_constraints_P(P, problem)
        self.normalization_P(P, problem)
        d = self.ma   # number of outcomes

        I = self.tripartite_inequality(P, d=d)

        problem.set_objective('min', I)   # inequality is ≥ M(d-1)
        problem.solve(solver=self.solver)  # SCS/ECOS también funcionan
        print("Min value of the inequality:", I.value)# %%
    
#%%
if __name__ == "__main__":
    tw = TW_abe(ma=2, mb=2, mc=2, kx=3, ky=3, kz=3, solver = 'gurobi')
    tw.solve_k3()
    
# %%
